"""Four-example LoRA memorization run and explicit manual approval gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset, RandomSampler

from voxcpm.training.data import VoxCPMCollator

from .artifacts import atomic_json, fingerprint, sha256_file
from .asr import GigaAMRNNT
from .checkpoint import CheckpointManager
from .schedule import EpochGeometry, TrainingProgress
from .selection import SelectedSample
from .tracking import create_run_manager, memorization_pair_payload
from .trainer import (
    _accelerator,
    _default_batch_processor_factory,
    _default_optimizer_factory,
    _default_scheduler_factory,
    _forward,
    _optimizer_step_was_skipped,
    _weighted_loss,
    build_model,
)

_APPROVAL_FINGERPRINT_FIELDS = frozenset(
    {
        "base_revision",
        "data_fingerprint",
        "selection_fingerprint",
        "lora_fingerprint",
        "generation_fingerprint",
        "checkpoint",
        "wandb_run_id",
        "wandb_completion",
        "result",
    }
)


class MemorizationError(RuntimeError):
    """The fixed four-item experiment is incomplete or unsafe to run."""


class ApprovalMismatch(MemorizationError):
    """A manual approval does not match its artifacts or expected identity."""


@dataclass(frozen=True)
class ApprovalRecord:
    """A verified, artifact-bound operator approval."""

    path: Path
    wandb_run_id: str
    approver: str
    approved_at: str
    fingerprints: Mapping[str, str]
    artifacts: tuple[Mapping[str, str], ...]
    approval_fingerprint: str


@dataclass(frozen=True)
class MemorizationResult:
    """Completed four-item diagnostic run; never a stage-training launch."""

    dir: Path
    result_path: Path
    status: str
    seen_sample_ids: tuple[str, ...]
    attempted_sample_ids: tuple[str, ...]
    contributing_sample_ids: tuple[str, ...]
    reference_audio: tuple[Path, ...]
    generated_audio: tuple[Path, ...]
    diagnostics: tuple[str, ...]
    checkpoint: Path
    wandb_run_id: str
    wandb_completion_path: Path
    fingerprints: Mapping[str, str]
    large_training_started: bool = False


class _RepeatDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: Sequence[SelectedSample], repeats: int, tokenizer: Callable[[str], Sequence[int]]):
        self.samples = tuple(samples)
        self.repeats = repeats
        self.tokenizer = tokenizer
        self._audio = tuple(_load_selected_wav(sample) for sample in samples)

    def __len__(self) -> int:
        return self.repeats

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index = index % len(self.samples)
        sample = self.samples[sample_index]
        waveform, sample_rate = self._audio[sample_index]
        return {
            "text_ids": list(self.tokenizer(sample.text)),
            "audio_array": waveform,
            "audio_sampling_rate": sample_rate,
            "dataset_id": sample_index,
            "is_prompt": False,
        }


def run_memorization(config: Any, runtime: Any) -> MemorizationResult:
    """Overfit the published four-item selection, log diagnostics, and stop."""
    settings = _settings(config, runtime)
    generation_settings = _memorization_generation_settings(config)
    selection_path, selection_fingerprint, samples = _load_selection(config)
    global_samples = settings["updates"] * settings["accumulation"] * settings["batch_size"] * runtime.world_size
    if global_samples < len(samples):
        raise MemorizationError("memorization geometry cannot cover all four selected identities")
    result_dir = Path(config.output_dir).resolve() / "memorization"
    result_path = result_dir / "memorization-result.json"
    if result_path.exists():
        raise MemorizationError(f"completed memorization result already exists: {result_path}")

    run_manager_holder: list[Any] = []

    def start_tracking() -> None:
        run_manager_holder.append(
            create_run_manager(
                is_main_process=True,
                config=config,
                job_type="memorization",
                run_state_path=result_dir / "wandb-run.json",
            )
        )

    _rank_zero_phase(runtime, result_dir, "tracking-start", start_tracking)
    run_manager = (
        run_manager_holder[0] if runtime.rank == 0 else create_run_manager(is_main_process=False, config=config)
    )
    _rank_zero_phase(
        runtime,
        result_dir,
        "tracking-mode",
        lambda: _require_online_tracking(run_manager),
    )
    run_id = getattr(run_manager, "run_id", None)
    if not isinstance(run_id, str) or not run_id:
        run_id = _wandb_run_id(result_dir / "wandb-run.json")
    identity = _identity(config, selection_fingerprint, settings, run_id, run_manager)

    model, audio_vae, tokenizer = build_model(config, 1, adapter_checkpoint=None)
    trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not trainable or any(
        "lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad
    ):
        raise MemorizationError("memorization may optimize only LoRA parameters")
    optimizer = _default_optimizer_factory(
        trainable,
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = _default_scheduler_factory(optimizer, warmup_steps=0, total_steps=settings["updates"])
    dataset = _RepeatDataset(samples, global_samples, tokenizer)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(settings["seed"])
    loader = DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        sampler=RandomSampler(dataset, generator=sampler_generator),
        drop_last=True,
        collate_fn=VoxCPMCollator(),
    )
    processor = _default_batch_processor_factory(model, audio_vae, runtime)
    prepared = runtime.prepare(model, optimizer, loader, scheduler)
    if not isinstance(prepared, tuple) or len(prepared) != 4:
        raise TypeError("runtime.prepare(model, optimizer, loader, scheduler) must return four objects")
    model, optimizer, loader, scheduler = prepared

    geometry_steps = max(8, settings["updates"])
    geometry = EpochGeometry.from_counts(
        geometry_steps * runtime.world_size * settings["batch_size"] * settings["accumulation"],
        runtime.world_size,
        settings["batch_size"],
        settings["accumulation"],
    )
    progress = TrainingProgress(stage="stage1", sampler_seed=settings["seed"])
    losses: list[float] = []
    attempted_sample_ids: list[str] = []
    contributing_sample_ids: list[str] = []
    pending_contribution_ids: list[str] = []
    accumulation_cursor = 0
    iterator = iter(loader)
    while progress.optimizer_step < settings["updates"]:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise MemorizationError("prepared memorization loader is empty") from error
        with runtime.accumulate(model):
            batch_sample_ids = _batch_sample_ids(batch, samples)
            attempted_sample_ids.extend(batch_sample_ids)
            pending_contribution_ids.extend(batch_sample_ids)
            processed = processor(batch)
            outputs = _forward(model, processed, progress.optimizer_step / settings["updates"])
            loss = _weighted_loss(outputs, settings["loss_weights"])
            runtime.backward(loss)
            accumulation_cursor += 1
            if not runtime.sync_gradients:
                continue
            if accumulation_cursor != settings["accumulation"]:
                raise MemorizationError("runtime accumulation does not match memorization configuration")
            runtime.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                settings["max_grad_norm"],
            )
            optimizer.step()
            skipped = _optimizer_step_was_skipped(runtime, optimizer)
            if not skipped:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_cursor = 0
            if skipped:
                pending_contribution_ids.clear()
                continue
            contributing_sample_ids.extend(pending_contribution_ids)
            pending_contribution_ids.clear()
            progress.complete_optimizer_step(geometry)
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            _rank_zero_phase(
                runtime,
                result_dir,
                "train-log",
                lambda: run_manager.log_train({"memorization/loss": loss_value}, progress.global_step),
            )
    if accumulation_cursor != 0 or pending_contribution_ids:
        raise MemorizationError("memorization loader ended before the configured updates completed")

    audit_dir = result_dir / "batch-audit"
    audit_path = audit_dir / f"rank-{runtime.rank:05d}.json"

    def write_batch_audit() -> None:
        atomic_json(
            audit_path,
            {
                "version": 1,
                "rank": runtime.rank,
                "selection_fingerprint": selection_fingerprint,
                "attempted_sample_ids": attempted_sample_ids,
                "contributing_sample_ids": contributing_sample_ids,
                "completed_optimizer_steps": progress.optimizer_step,
            },
        )

    _all_rank_phase(runtime, result_dir, "batch-audit", write_batch_audit)
    combined_attempted: list[str] = []
    combined_contributing: list[str] = []
    audit_paths: list[Path] = []

    def validate_coverage() -> None:
        audit_paths.extend(audit_dir / f"rank-{rank:05d}.json" for rank in range(runtime.world_size))
        for path in audit_paths:
            audit = _json_mapping(path, "memorization batch audit")
            if audit.get("selection_fingerprint") != selection_fingerprint:
                raise MemorizationError(f"batch audit selection identity does not match: {path}")
            combined_attempted.extend(_identity_sequence(audit.get("attempted_sample_ids"), "attempted samples"))
            combined_contributing.extend(
                _identity_sequence(audit.get("contributing_sample_ids"), "contributing samples")
            )
        _require_exact_coverage(samples, combined_attempted, combined_contributing, MemorizationError)

    _rank_zero_phase(runtime, result_dir, "coverage-validation", validate_coverage)

    checkpoint_manager = CheckpointManager(result_dir / "checkpoints")
    checkpoint = checkpoint_manager.save_recovery(
        _accelerator(runtime),
        model,
        progress,
        _checkpoint_metadata(identity, settings, runtime, geometry_steps),
        name="memorization-final",
    )
    checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint)
    generated_audio: list[Path] = []
    diagnostics: list[str] = []
    pairs: list[dict[str, object]] = []

    def generate_and_diagnose() -> None:
        generated_dir = result_dir / "generated"
        target_model = runtime.unwrap(model)
        model_was_training = target_model.training
        audio_vae_was_training = audio_vae.training
        setattr(target_model, "audio_vae", audio_vae)
        try:
            target_model.eval()
            audio_vae.eval()
            for index, sample in enumerate(samples):
                generated = target_model.generate(
                    target_text=sample.text,
                    prompt_text=None,
                    prompt_wav_path=None,
                    seed=settings["seed"] + index,
                    cfg_value=generation_settings["cfg_value"],
                    inference_timesteps=generation_settings["inference_timesteps"],
                    max_len=generation_settings["max_length"],
                )
                generated_path = generated_dir / f"item-{index:02d}.wav"
                _atomic_wav(generated_path, generated, _sample_rate(target_model))
                generated_audio.append(generated_path)
        finally:
            delattr(target_model, "audio_vae")
            target_model.train(model_was_training)
            audio_vae.train(audio_vae_was_training)

        diagnostics.extend(_diagnose_generated(config, runtime, generated_audio))
        pairs.extend(
            {
                "sample_id": sample.source_relative_path,
                "text": sample.text,
                "reference_audio": sample.wav_path,
                "generated_audio": generated_path,
                "asr_hypothesis": hypothesis,
            }
            for sample, generated_path, hypothesis in zip(samples, generated_audio, diagnostics, strict=True)
        )

    _rank_zero_phase(runtime, result_dir, "generation", generate_and_diagnose)
    upload_receipts: list[dict[str, Any]] = []

    def upload_pairs() -> None:
        receipt = run_manager.log_memorization(pairs, global_step=progress.global_step)
        upload_receipts.append(_validated_upload_receipt(receipt, pairs, progress.global_step, run_id))

    _rank_zero_phase(runtime, result_dir, "pair-upload", upload_pairs)

    def finish_tracking() -> None:
        run_manager.finish()
        receipt = upload_receipts[0]
        wandb_completion_path = result_dir / "wandb-complete.json"
        atomic_json(
            wandb_completion_path,
            {
                "version": 1,
                "status": "complete",
                "job_type": "memorization",
                "run_id": run_id,
                "mode": "online",
                "global_step": receipt["global_step"],
                "pair_count": receipt["pair_count"],
                "pair_payload_fingerprint": receipt["pair_payload_fingerprint"],
                "client_log_returned": True,
                "remote_receipt_verified": False,
                "verification_scope": (
                    "local W&B client log and finish calls returned; "
                    "W&B provides no server-side receipt in this workflow"
                ),
            },
        )

    _rank_zero_phase(runtime, result_dir, "wandb-finish", finish_tracking)

    def publish_result() -> None:
        wandb_completion_path = result_dir / "wandb-complete.json"
        reference_audio = tuple(sample.wav_path.resolve() for sample in samples)
        generated_tuple = tuple(path.resolve() for path in generated_audio)
        core_fingerprints = {
            "base_revision": identity["base_revision"],
            "data_fingerprint": identity["data_fingerprint"],
            "selection_fingerprint": identity["selection_fingerprint"],
            "lora_fingerprint": identity["lora_fingerprint"],
            "generation_fingerprint": fingerprint(generation_settings),
            "checkpoint": checkpoint_fingerprint,
            "wandb_run_id": run_id,
            "wandb_completion": sha256_file(wandb_completion_path),
        }
        result_value = {
            "version": 1,
            "status": "complete",
            "large_training_started": False,
            "updates": settings["updates"],
            "seen_sample_ids": combined_attempted,
            "attempted_sample_ids": combined_attempted,
            "contributing_sample_ids": combined_contributing,
            "batch_audits": [_artifact_value(path) for path in audit_paths],
            "wandb_pair_payload": upload_receipts[0]["pair_payload"],
            "reference_audio": [_artifact_value(path) for path in reference_audio],
            "generated_audio": [_artifact_value(path) for path in generated_tuple],
            "diagnostics": diagnostics,
            "losses": losses,
            "generation_settings": generation_settings,
            "checkpoint": str(Path(checkpoint).resolve()),
            "selection_manifest": str(selection_path.resolve()),
            "wandb_completion": str(wandb_completion_path.resolve()),
            "fingerprints": core_fingerprints,
        }
        atomic_json(result_path, result_value)

    _rank_zero_phase(runtime, result_dir, "result-publication", publish_result)
    return _load_result(result_path)


def approve_memorization(result_dir: str | Path, wandb_run_id: str, approver: str) -> Path:
    """Atomically record a human decision without launching any training."""
    if not isinstance(wandb_run_id, str) or not wandb_run_id:
        raise ValueError("W&B run ID must be a non-empty string")
    if not isinstance(approver, str) or not approver.strip():
        raise ValueError("approver must be a non-empty string")
    result_dir = Path(result_dir).resolve()
    result_path = result_dir / "memorization-result.json"
    result = _json_mapping(result_path, "memorization result")
    if result.get("status") != "complete" or result.get("large_training_started") is not False:
        raise ApprovalMismatch("memorization result is not a completed stopped diagnostic run")
    _validate_result_coverage(result)
    fingerprints = _string_mapping(result.get("fingerprints"), "result fingerprints")
    if fingerprints.get("wandb_run_id") != wandb_run_id:
        raise ApprovalMismatch("W&B run identity does not match the completed memorization result")
    completion_path = Path(str(result.get("wandb_completion", "")))
    completion = _json_mapping(completion_path, "W&B completion")
    if completion.get("status") != "complete" or completion.get("run_id") != wandb_run_id:
        raise ApprovalMismatch("W&B completion does not match the requested run")
    _validate_wandb_evidence(result, completion, wandb_run_id)
    if sha256_file(completion_path) != fingerprints.get("wandb_completion"):
        raise ApprovalMismatch("W&B completion artifact hash does not match the result")

    artifacts = _approval_artifacts(result_path, result)
    bound_fingerprints = {**fingerprints, "result": sha256_file(result_path)}
    approval: dict[str, Any] = {
        "version": 1,
        "status": "approved",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "approver": approver.strip(),
        "wandb_run_id": wandb_run_id,
        "fingerprints": bound_fingerprints,
        "artifacts": artifacts,
    }
    approval["approval_fingerprint"] = fingerprint(approval)
    path = result_dir / "memorization-approval.json"
    atomic_json(path, approval)
    verify_approval(path, bound_fingerprints)
    return path


def verify_approval(path: str | Path, expected_fingerprints: Mapping[str, Any]) -> ApprovalRecord:
    """Verify approval integrity, every bound file, and the caller identity."""
    path = Path(path).resolve()
    value = _json_mapping(path, "memorization approval")
    stored_fingerprint = value.get("approval_fingerprint")
    payload = {key: item for key, item in value.items() if key != "approval_fingerprint"}
    if not isinstance(stored_fingerprint, str) or fingerprint(payload) != stored_fingerprint:
        raise ApprovalMismatch("approval fingerprint does not match the record")
    if value.get("status") != "approved":
        raise ApprovalMismatch("memorization approval status is not approved")
    fingerprints = _string_mapping(value.get("fingerprints"), "approval fingerprints")
    missing = sorted(_APPROVAL_FINGERPRINT_FIELDS - set(expected_fingerprints))
    extra = sorted(set(expected_fingerprints) - _APPROVAL_FINGERPRINT_FIELDS)
    if missing or extra:
        raise ApprovalMismatch(f"expected fingerprints are incomplete or invalid; missing={missing}, extra={extra}")
    for name, expected in expected_fingerprints.items():
        if not isinstance(expected, str) or fingerprints.get(str(name)) != expected:
            raise ApprovalMismatch(f"approval {name} fingerprint does not match")
    artifacts_value = value.get("artifacts")
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise ApprovalMismatch("approval has no artifact bindings")
    artifacts: list[Mapping[str, str]] = []
    for item in artifacts_value:
        record = _string_mapping(item, "approval artifact")
        artifact_path = Path(record.get("path", ""))
        if not artifact_path.is_file() or sha256_file(artifact_path) != record.get("sha256"):
            raise ApprovalMismatch(f"approval artifact does not match: {artifact_path}")
        artifacts.append(MappingProxyType(record))
    approver = value.get("approver")
    run_id = value.get("wandb_run_id")
    approved_at = value.get("approved_at")
    if not all(isinstance(item, str) and item for item in (approver, run_id, approved_at)):
        raise ApprovalMismatch("approval operator, W&B run, or timestamp is missing")
    return ApprovalRecord(
        path=path,
        wandb_run_id=run_id,
        approver=approver,
        approved_at=approved_at,
        fingerprints=MappingProxyType(fingerprints),
        artifacts=tuple(artifacts),
        approval_fingerprint=stored_fingerprint,
    )


def _settings(config: Any, runtime: Any) -> dict[str, Any]:
    value = getattr(config, "memorization", None)
    if value is None:
        raise MemorizationError("configuration must define memorization settings")
    settings = {
        "updates": getattr(value, "updates", None),
        "learning_rate": getattr(value, "learning_rate", None),
        "batch_size": getattr(value, "batch_size", 1),
        "accumulation": getattr(value, "accumulation", 1),
        "seed": getattr(value, "seed", 0),
        "weight_decay": getattr(value, "weight_decay", 0.01),
        "max_grad_norm": getattr(value, "max_grad_norm", 1.0),
        "loss_weights": dict(getattr(value, "loss_weights", {"loss/diff": 1.0, "loss/stop": 1.0})),
    }
    for name in ("updates", "batch_size", "accumulation"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] <= 0:
            raise MemorizationError(f"memorization {name} must be a positive integer")
    if isinstance(settings["seed"], bool) or not isinstance(settings["seed"], int):
        raise MemorizationError("memorization seed must be an integer")
    for name in ("learning_rate", "max_grad_norm"):
        if not isinstance(settings[name], (int, float)) or settings[name] <= 0:
            raise MemorizationError(f"memorization {name} must be positive")
        settings[name] = float(settings[name])
    if not isinstance(settings["weight_decay"], (int, float)) or settings["weight_decay"] < 0:
        raise MemorizationError("memorization weight_decay must be non-negative")
    if settings["accumulation"] != getattr(runtime, "accumulation", settings["accumulation"]):
        raise MemorizationError("runtime and memorization accumulation must match")
    return settings


def _memorization_generation_settings(config: Any) -> dict[str, object]:
    value = getattr(config, "generation", None)
    if value is None:
        raise MemorizationError("configuration must define generation settings")
    if callable(getattr(value, "model_dump", None)):
        settings = dict(value.model_dump(mode="json"))
    else:
        settings = {
            "cfg_value": getattr(value, "cfg_value", None),
            "inference_timesteps": getattr(value, "inference_timesteps", None),
            "max_length": getattr(value, "max_length", None),
        }
    if not isinstance(settings["cfg_value"], (int, float)):
        raise MemorizationError("generation cfg_value must be numeric")
    for name in ("inference_timesteps", "max_length"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] <= 0:
            raise MemorizationError(f"generation {name} must be a positive integer")
    return settings


def _require_online_tracking(run_manager: Any) -> None:
    mode = getattr(getattr(run_manager, "_settings", None), "mode", None)
    if mode != "online":
        raise MemorizationError("memorization approval requires online W&B tracking")


def _validated_upload_receipt(
    receipt: Any,
    pairs: Sequence[Mapping[str, object]],
    global_step: int,
    run_id: str,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise MemorizationError("online W&B pair upload returned no verifiable local receipt")
    expected_payload = memorization_pair_payload(
        pairs,
        global_step=global_step,
        run_id=run_id,
        mode="online",
    )
    payload = receipt.get("pair_payload")
    expected_fingerprint = fingerprint(expected_payload)
    if (
        not isinstance(payload, Mapping)
        or dict(payload) != expected_payload
        or receipt.get("run_id") != run_id
        or receipt.get("mode") != "online"
        or receipt.get("global_step") != global_step
        or receipt.get("pair_count") != 4
        or receipt.get("pair_payload_fingerprint") != expected_fingerprint
    ):
        raise MemorizationError("online W&B receipt does not bind the exact four-pair payload and run")
    return {
        "run_id": run_id,
        "mode": "online",
        "global_step": global_step,
        "pair_count": 4,
        "pair_payload": expected_payload,
        "pair_payload_fingerprint": expected_fingerprint,
    }


def _load_selection(config: Any) -> tuple[Path, str, tuple[SelectedSample, ...]]:
    selection_dir = Path(getattr(config, "selection_dir", Path(config.output_dir) / "selection"))
    path = selection_dir / "memorization.json"
    value = _json_mapping(path, "memorization selection")
    selection_fingerprint = value.get("fingerprint")
    raw_samples = value.get("memorization")
    if not isinstance(selection_fingerprint, str) or not selection_fingerprint or not isinstance(raw_samples, list):
        raise MemorizationError("published memorization selection is malformed")
    try:
        samples = tuple(SelectedSample.model_validate(item) for item in raw_samples)
    except Exception as error:
        raise MemorizationError("published memorization selection is malformed") from error
    identities = {sample.source_relative_path for sample in samples}
    if len(samples) != 4 or len(identities) != 4:
        raise MemorizationError("memorization requires exactly four distinct selected sample identities")
    for sample in samples:
        if not sample.wav_path.is_file() or sha256_file(sample.wav_path) != sample.wav_sha256:
            raise MemorizationError(f"selected reference artifact changed: {sample.source_relative_path}")
    return path.resolve(), selection_fingerprint, samples


def _load_selected_wav(sample: SelectedSample) -> tuple[np.ndarray, int]:
    waveform, sample_rate = torchaudio.load(sample.wav_path)
    if waveform.ndim != 2 or waveform.size(0) == 0 or sample_rate <= 0:
        raise MemorizationError(f"selected WAV is invalid: {sample.wav_path}")
    mono = waveform.mean(dim=0).numpy().astype(np.float32, copy=False)
    return mono, int(sample_rate)


def _identity(
    config: Any,
    selection_fingerprint: str,
    settings: Mapping[str, Any],
    run_id: str,
    run_manager: Any,
) -> dict[str, Any]:
    pins = _json_mapping(Path(config.hub.local_dir) / "hub-pins.json", "Hub pins")
    model_pin = pins.get("model")
    asr_pin = pins.get("gigaam")
    if not isinstance(model_pin, Mapping) or not isinstance(asr_pin, Mapping):
        raise MemorizationError("Hub pins must include model and GigaAM identities")
    base_revision = model_pin.get("revision")
    evaluator_revision = asr_pin.get("revision")
    if (
        not isinstance(base_revision, str)
        or not base_revision
        or not isinstance(evaluator_revision, str)
        or not evaluator_revision
    ):
        raise MemorizationError("Hub pins must contain immutable revisions")
    index_path = Path(config.data.index_dir) / "balalaika-index.sqlite3"
    try:
        with sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True) as database:
            row = database.execute("SELECT value FROM metadata WHERE key = 'fingerprint'").fetchone()
    except sqlite3.Error as error:
        raise MemorizationError(f"cannot read index identity: {index_path}") from error
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise MemorizationError("index has no data fingerprint")
    lora_value = _object_mapping(config.lora)
    group = getattr(getattr(run_manager, "_settings", None), "group", None)
    if group is None:
        group = getattr(getattr(config, "wandb", None), "group", None)
    return {
        "base_revision": base_revision,
        "evaluator_revision": evaluator_revision,
        "data_fingerprint": row[0],
        "selection_fingerprint": selection_fingerprint,
        "lora_fingerprint": fingerprint(lora_value),
        "optimization_fingerprint": fingerprint(_jsonable(dict(settings))),
        "wandb_run_id": run_id,
        "wandb_group": group,
    }


def _checkpoint_metadata(
    identity: Mapping[str, Any], settings: Mapping[str, Any], runtime: Any, geometry_steps: int
) -> dict[str, Any]:
    return {
        "stage_epochs": 1,
        "world_size": runtime.world_size,
        "microbatch": settings["batch_size"],
        "accumulation": settings["accumulation"],
        "global_batch_size": runtime.world_size * settings["batch_size"] * settings["accumulation"],
        "optimizer_steps_per_epoch": geometry_steps,
        "optimizer_count": 1,
        "scheduler_count": 1,
        "checkpointable_count": 1,
        "stage_start_global_step": 0,
        **identity,
    }


def _create_asr(config: Any, runtime: Any) -> GigaAMRNNT:
    pins = _json_mapping(Path(config.hub.local_dir) / "hub-pins.json", "Hub pins")
    pin = pins.get("gigaam")
    if not isinstance(pin, Mapping):
        raise MemorizationError("Hub pins have no GigaAM model")
    model_dir = Path(str(pin.get("local_dir", "")))
    device_id = runtime.device.index if runtime.device.index is not None else runtime.rank
    return GigaAMRNNT(model_dir, device_id, fingerprint(pin))


def _diagnose_generated(config: Any, runtime: Any, generated_audio: Sequence[Path]) -> list[str]:
    diagnostics: list[str] = []
    try:
        asr = _create_asr(config, runtime)
        with asr:
            for path in generated_audio:
                try:
                    hypothesis = asr.transcribe(path)
                    if not isinstance(hypothesis, str):
                        raise TypeError("GigaAM diagnostic transcript must be a string")
                except Exception as error:
                    hypothesis = _diagnostic_error(error)
                diagnostics.append(hypothesis)
    except Exception as error:
        diagnostics = [_diagnostic_error(error)] * len(generated_audio)
    return diagnostics


def _diagnostic_error(error: Exception) -> str:
    return f"ERROR: {type(error).__name__}: {error}"


def _rank_zero_phase(runtime: Any, result_dir: Path, phase: str, operation: Callable[[], None]) -> None:
    local_error: BaseException | None = None
    if runtime.rank == 0:
        try:
            operation()
        except BaseException as error:
            local_error = error
            try:
                atomic_json(
                    result_dir / "memorization-failed.json",
                    {
                        "version": 1,
                        "status": "failed",
                        "phase": phase,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    },
                )
            except BaseException:
                pass
    status = torch.tensor([0 if local_error is not None else 1], dtype=torch.int8, device=runtime.device)
    try:
        gathered = runtime.gather(status).reshape(-1)
    except BaseException as error:
        raise MemorizationError(f"memorization {phase} outcome collective failed") from (local_error or error)
    if bool((gathered == 1).all().item()):
        return
    failure = _json_mapping(result_dir / "memorization-failed.json", "memorization failure")
    error = failure.get("error")
    if failure.get("status") != "failed" or failure.get("phase") != phase or not isinstance(error, Mapping):
        raise MemorizationError(f"memorization {phase} failed on rank zero without matching failure evidence")
    error_type = str(error.get("type", "RuntimeError"))
    message = str(error.get("message", "memorization phase failed"))
    raise MemorizationError(f"memorization {phase} failed on rank zero: {error_type}: {message}") from local_error


def _all_rank_phase(runtime: Any, result_dir: Path, phase: str, operation: Callable[[], None]) -> None:
    local_error: BaseException | None = None
    failure_path = result_dir / f"memorization-failed-rank-{runtime.rank:05d}.json"
    try:
        operation()
    except BaseException as error:
        local_error = error
        try:
            atomic_json(
                failure_path,
                {
                    "version": 1,
                    "status": "failed",
                    "phase": phase,
                    "rank": runtime.rank,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        except BaseException:
            pass
    status = torch.tensor([0 if local_error is not None else 1], dtype=torch.int8, device=runtime.device)
    try:
        gathered = runtime.gather(status).reshape(-1)
    except BaseException as error:
        raise MemorizationError(f"memorization {phase} outcome collective failed") from (local_error or error)
    failed_ranks = [rank for rank, item in enumerate(gathered.tolist()) if item == 0]
    if not failed_ranks:
        return
    failed_rank = failed_ranks[0]
    failure = _json_mapping(
        result_dir / f"memorization-failed-rank-{failed_rank:05d}.json",
        "memorization rank failure",
    )
    error = failure.get("error")
    if (
        failure.get("status") != "failed"
        or failure.get("phase") != phase
        or failure.get("rank") != failed_rank
        or not isinstance(error, Mapping)
    ):
        raise MemorizationError(f"memorization {phase} failed on rank {failed_rank} without matching evidence")
    error_type = str(error.get("type", "RuntimeError"))
    message = str(error.get("message", "memorization phase failed"))
    raise MemorizationError(
        f"memorization {phase} failed on rank {failed_rank}: {error_type}: {message}"
    ) from local_error


def _batch_sample_ids(batch: Mapping[str, Any], samples: Sequence[SelectedSample]) -> list[str]:
    values = batch.get("dataset_ids")
    if isinstance(values, torch.Tensor):
        raw_ids = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        raw_ids = list(values)
    else:
        raise MemorizationError("prepared memorization batch has no dataset identity audit")
    identities: list[str] = []
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(samples):
            raise MemorizationError("prepared memorization batch has an invalid dataset identity")
        identities.append(samples[value].source_relative_path)
    if not identities:
        raise MemorizationError("prepared memorization batch has an empty dataset identity audit")
    return identities


def _identity_sequence(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ApprovalMismatch(f"{label} must be a non-empty identity sequence")
    return list(value)


def _require_exact_coverage(
    samples: Sequence[SelectedSample] | Sequence[str],
    attempted: Sequence[str],
    contributing: Sequence[str],
    error_type: type[MemorizationError],
) -> None:
    selected = {sample.source_relative_path if isinstance(sample, SelectedSample) else sample for sample in samples}
    if len(selected) != 4:
        raise error_type("memorization coverage requires four selected identities")
    attempted_set = set(attempted)
    if attempted_set != selected:
        raise error_type(
            "observed attempt coverage does not match the four selected identities; "
            f"missing={sorted(selected - attempted_set)}, unexpected={sorted(attempted_set - selected)}"
        )
    contributing_set = set(contributing)
    if contributing_set != selected:
        raise error_type(
            "observed contribution coverage does not match the four selected identities; "
            f"missing={sorted(selected - contributing_set)}, unexpected={sorted(contributing_set - selected)}"
        )


def _validate_result_coverage(result: Mapping[str, Any]) -> None:
    selection_path = Path(str(result.get("selection_manifest", "")))
    selection = _json_mapping(selection_path, "memorization selection")
    raw_samples = selection.get("memorization")
    if not isinstance(raw_samples, list):
        raise ApprovalMismatch("memorization selection has no selected identities")
    selected: list[str] = []
    for item in raw_samples:
        if not isinstance(item, Mapping):
            raise ApprovalMismatch("memorization selection identity is malformed")
        identity = item.get("source_relative_path")
        if not isinstance(identity, str) or not identity:
            raise ApprovalMismatch("memorization selection identity is malformed")
        selected.append(identity)
    attempted = _identity_sequence(result.get("attempted_sample_ids"), "attempted samples")
    contributing = _identity_sequence(result.get("contributing_sample_ids"), "contributing samples")
    _require_exact_coverage(selected, attempted, contributing, ApprovalMismatch)
    seen = _identity_sequence(result.get("seen_sample_ids"), "seen samples")
    if seen != attempted:
        raise ApprovalMismatch("seen sample audit does not match actual attempted batches")

    audits = result.get("batch_audits")
    if not isinstance(audits, list) or not audits:
        raise ApprovalMismatch("memorization result has no per-rank batch audits")
    audited_attempted: list[str] = []
    audited_contributing: list[str] = []
    for item in audits:
        if not isinstance(item, Mapping):
            raise ApprovalMismatch("memorization batch audit artifact is malformed")
        path = Path(str(item.get("path", "")))
        expected = item.get("sha256")
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise ApprovalMismatch(f"memorization batch audit artifact does not match: {path}")
        audit = _json_mapping(path, "memorization batch audit")
        audited_attempted.extend(_identity_sequence(audit.get("attempted_sample_ids"), "attempted samples"))
        audited_contributing.extend(_identity_sequence(audit.get("contributing_sample_ids"), "contributing samples"))
    if audited_attempted != attempted or audited_contributing != contributing:
        raise ApprovalMismatch("memorization result identity coverage does not match its batch audits")


def _validate_wandb_evidence(result: Mapping[str, Any], completion: Mapping[str, Any], run_id: str) -> None:
    if completion.get("mode") != "online":
        raise ApprovalMismatch("memorization approval requires an online W&B completion")
    expected_scope = (
        "local W&B client log and finish calls returned; " "W&B provides no server-side receipt in this workflow"
    )
    if (
        completion.get("job_type") != "memorization"
        or completion.get("run_id") != run_id
        or completion.get("pair_count") != 4
        or completion.get("client_log_returned") is not True
        or completion.get("remote_receipt_verified") is not False
        or completion.get("verification_scope") != expected_scope
    ):
        raise ApprovalMismatch("online W&B completion evidence is incomplete or unverifiable")
    payload = result.get("wandb_pair_payload")
    if not isinstance(payload, Mapping):
        raise ApprovalMismatch("online W&B completion has no exact four-pair payload binding")
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 4:
        raise ApprovalMismatch("online W&B completion has no exact four-pair payload binding")
    reconstructed: list[dict[str, object]] = []
    try:
        for pair in raw_pairs:
            if not isinstance(pair, Mapping):
                raise TypeError("pair must be a mapping")
            reference = pair.get("reference_audio")
            generated = pair.get("generated_audio")
            if not isinstance(reference, Mapping) or not isinstance(generated, Mapping):
                raise TypeError("pair audio must be a mapping")
            for artifact in (reference, generated):
                path = Path(str(artifact.get("path", "")))
                expected_hash = artifact.get("sha256")
                if not path.is_file() or not isinstance(expected_hash, str) or sha256_file(path) != expected_hash:
                    raise ValueError(f"pair audio artifact does not match: {path}")
            reconstructed.append(
                {
                    "sample_id": pair.get("sample_id"),
                    "text": pair.get("text"),
                    "asr_hypothesis": pair.get("asr_hypothesis"),
                    "reference_audio": reference.get("path"),
                    "generated_audio": generated.get("path"),
                }
            )
        canonical = memorization_pair_payload(
            reconstructed,
            global_step=int(completion.get("global_step")),
            run_id=run_id,
            mode="online",
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise ApprovalMismatch("online W&B four-pair payload is malformed or unverifiable") from error
    if (
        dict(payload) != canonical
        or completion.get("pair_payload_fingerprint") != fingerprint(canonical)
        or completion.get("global_step") != canonical["global_step"]
    ):
        raise ApprovalMismatch("online W&B four-pair payload fingerprint does not match")


def _atomic_wav(path: Path, generated: Any, sample_rate: int) -> None:
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    waveform = np.asarray(generated, dtype=np.float32).squeeze()
    if waveform.ndim != 1 or waveform.size == 0 or sample_rate <= 0:
        raise MemorizationError("generated audio must be a non-empty mono waveform")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp.wav", delete=False) as file:
            temporary_path = Path(file.name)
        sf.write(temporary_path, waveform, sample_rate, format="WAV", subtype="PCM_16")
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _sample_rate(model: Any) -> int:
    value = getattr(model, "sample_rate", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MemorizationError("VoxCPM2 model has no valid output sample rate")
    return value


def _checkpoint_fingerprint(checkpoint: Path) -> str:
    metadata = _json_mapping(Path(checkpoint) / "metadata.json", "memorization checkpoint metadata")
    value = metadata.get("checkpoint_fingerprint")
    if isinstance(value, str) and value:
        return value
    return fingerprint(_tree_hashes(checkpoint))


def _wandb_run_id(path: Path) -> str:
    state = _json_mapping(path, "memorization W&B run state")
    run_id = state.get("run_id")
    if state.get("job_type") != "memorization" or not isinstance(run_id, str) or not run_id:
        raise MemorizationError("memorization requires a distinct persisted W&B run identity")
    return run_id


def _load_result(path: Path) -> MemorizationResult:
    value = _json_mapping(path, "memorization result")
    if value.get("status") != "complete" or value.get("large_training_started") is not False:
        raise MemorizationError("memorization result is not complete")
    _validate_result_coverage(value)
    seen = _identity_sequence(value.get("seen_sample_ids"), "seen samples")
    attempted = _identity_sequence(value.get("attempted_sample_ids"), "attempted samples")
    contributing = _identity_sequence(value.get("contributing_sample_ids"), "contributing samples")
    diagnostics = value.get("diagnostics")
    references = value.get("reference_audio")
    generated = value.get("generated_audio")
    if (
        not isinstance(diagnostics, list)
        or len(diagnostics) != 4
        or not all(isinstance(item, str) for item in diagnostics)
        or not isinstance(references, list)
        or len(references) != 4
        or not isinstance(generated, list)
        or len(generated) != 4
    ):
        raise MemorizationError("memorization result has malformed examples")
    reference_paths = tuple(Path(str(item["path"])).resolve() for item in references if isinstance(item, Mapping))
    generated_paths = tuple(Path(str(item["path"])).resolve() for item in generated if isinstance(item, Mapping))
    if len(reference_paths) != 4 or len(generated_paths) != 4:
        raise MemorizationError("memorization result has malformed artifact paths")
    core_fingerprints = _string_mapping(value.get("fingerprints"), "result fingerprints")
    completion_path = Path(str(value.get("wandb_completion", ""))).resolve()
    completion = _json_mapping(completion_path, "W&B completion")
    _validate_wandb_evidence(value, completion, core_fingerprints.get("wandb_run_id", ""))
    if sha256_file(completion_path) != core_fingerprints.get("wandb_completion"):
        raise ApprovalMismatch("W&B completion artifact hash does not match the result")
    return MemorizationResult(
        dir=path.parent.resolve(),
        result_path=path.resolve(),
        status="complete",
        seen_sample_ids=tuple(seen),
        attempted_sample_ids=tuple(attempted),
        contributing_sample_ids=tuple(contributing),
        reference_audio=reference_paths,
        generated_audio=generated_paths,
        diagnostics=tuple(diagnostics),
        checkpoint=Path(str(value.get("checkpoint", ""))).resolve(),
        wandb_run_id=core_fingerprints["wandb_run_id"],
        wandb_completion_path=completion_path,
        fingerprints=MappingProxyType({**core_fingerprints, "result": sha256_file(path)}),
    )


def _approval_artifacts(result_path: Path, result: Mapping[str, Any]) -> list[dict[str, str]]:
    artifacts = [{"role": "result", "path": str(result_path), "sha256": sha256_file(result_path)}]
    for role in ("selection_manifest", "wandb_completion"):
        candidate = Path(str(result.get(role, "")))
        if not candidate.is_file():
            raise ApprovalMismatch(f"required {role} artifact is missing: {candidate}")
        artifacts.append({"role": role, "path": str(candidate.resolve()), "sha256": sha256_file(candidate)})
    for role in ("reference_audio", "generated_audio"):
        values = result.get(role)
        if not isinstance(values, list) or len(values) != 4:
            raise ApprovalMismatch(f"result must bind exactly four {role} artifacts")
        for number, item in enumerate(values):
            if not isinstance(item, Mapping):
                raise ApprovalMismatch(f"result {role} artifact is malformed")
            candidate = Path(str(item.get("path", "")))
            expected = item.get("sha256")
            if not candidate.is_file() or not isinstance(expected, str) or sha256_file(candidate) != expected:
                raise ApprovalMismatch(f"result {role} artifact does not match: {candidate}")
            artifacts.append({"role": f"{role}:{number}", "path": str(candidate.resolve()), "sha256": expected})
    audits = result.get("batch_audits")
    if not isinstance(audits, list) or not audits:
        raise ApprovalMismatch("result must bind the per-rank batch audit artifacts")
    for number, item in enumerate(audits):
        if not isinstance(item, Mapping):
            raise ApprovalMismatch("result batch audit artifact is malformed")
        candidate = Path(str(item.get("path", "")))
        expected = item.get("sha256")
        if not candidate.is_file() or not isinstance(expected, str) or sha256_file(candidate) != expected:
            raise ApprovalMismatch(f"result batch audit artifact does not match: {candidate}")
        artifacts.append({"role": f"batch_audit:{number}", "path": str(candidate.resolve()), "sha256": expected})
    checkpoint = Path(str(result.get("checkpoint", "")))
    if not checkpoint.is_dir():
        raise ApprovalMismatch(f"required checkpoint artifact is missing: {checkpoint}")
    result_fingerprints = _string_mapping(result.get("fingerprints"), "result fingerprints")
    if _checkpoint_fingerprint(checkpoint) != result_fingerprints.get("checkpoint"):
        raise ApprovalMismatch("checkpoint fingerprint does not match the completed memorization result")
    checkpoint_files = [path for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    if not checkpoint_files:
        raise ApprovalMismatch("memorization checkpoint contains no files")
    for candidate in checkpoint_files:
        artifacts.append(
            {
                "role": f"checkpoint:{candidate.relative_to(checkpoint).as_posix()}",
                "path": str(candidate.resolve()),
                "sha256": sha256_file(candidate),
            }
        )
    return artifacts


def _artifact_value(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _tree_hashes(path: Path) -> dict[str, str]:
    return {
        candidate.relative_to(path).as_posix(): sha256_file(candidate)
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
    }


def _json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ApprovalMismatch(f"{label} is missing or unreadable: {path}") from error
    if not isinstance(value, dict):
        raise ApprovalMismatch(f"{label} must be a JSON object: {path}")
    return value


def _string_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) or not item for key, item in value.items()
    ):
        raise ApprovalMismatch(f"{label} must contain non-empty string values")
    return dict(value)


def _object_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    return {key: item for key, item in vars(value).items() if not key.startswith("_")}


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "ApprovalMismatch",
    "ApprovalRecord",
    "MemorizationError",
    "MemorizationResult",
    "approve_memorization",
    "run_memorization",
    "verify_approval",
]
