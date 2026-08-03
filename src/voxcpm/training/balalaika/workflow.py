"""Typed operator workflows behind the thin Balalaika CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sqlite3
import subprocess
from subprocess import CompletedProcess
from typing import Any

from .artifacts import sha256_file
from .artifacts import fingerprint
from .asr import GigaAMRNNT
from .checkpoint import CheckpointManager
from .cli import CommandResult
from .config import BalalaikaConfig, DataConfig
from .hub import CommandRunner, HubPin, pin_hub_inputs
from .index import BuildExpectations, IndexAudit, build_index
from .evaluation import DistributedEvaluator
from .ledger import ValidationLedger
from .memorization import approve_memorization, run_memorization, verify_approval
from .metrics import BenchmarkRow
from .runtime import AccelerateRuntime
from .selection import PromptSample, SelectedSample, SelectionBundle, create_selection_manifests
from .tracking import create_run_manager
from .trainer import BalalaikaTrainer, EvaluationBoundary

_JSON = Mapping[str, Any]


class SubprocessRunner(CommandRunner):
    """Execute an argv sequence without a shell, preserving the operator's auth."""

    def run(self, argv: Sequence[str]) -> CompletedProcess[str]:
        return subprocess.run(list(argv), check=False, capture_output=True, text=True)


def load_build_expectations(config: DataConfig) -> BuildExpectations:
    """Derive immutable index expectations from the trusted corpus manifests."""
    if config.verification_manifest is None or config.combined_metadata is None:
        raise ValueError("data verification_manifest and combined_metadata are required for preparation")
    verification = _read_json(config.verification_manifest, "corpus verification manifest")
    combined = _read_json(config.combined_metadata, "combined-sidecar metadata")
    if verification.get("status") != "ok":
        raise ValueError("corpus verification manifest is not successful")
    stats = _mapping(verification.get("stats"), "corpus verification stats")
    shards = verification.get("shards")
    if not isinstance(shards, list) or len(shards) != config.expected_shards:
        raise ValueError(
            f"corpus verification must contain {config.expected_shards} shards; found "
            f"{len(shards) if isinstance(shards, list) else 'malformed'}"
        )
    source_hashes: dict[str, str] = {}
    source_rows = 0
    for item in shards:
        row = _mapping(item, "corpus shard verification")
        name = row.get("shard")
        digest = row.get("sha256")
        samples = row.get("samples")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("shard_")
            or not name.endswith(".tar")
        ):
            raise ValueError("corpus verification contains an invalid shard name")
        _require_sha256(digest, f"source shard {name}")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError(f"source shard {name} has an invalid sample count")
        relative = f"train/{name}"
        if relative in source_hashes:
            raise ValueError(f"corpus verification contains duplicate shard {name}")
        source_hashes[relative] = digest
        source_rows += samples
    declared_rows = stats.get("samples")
    if declared_rows != config.expected_rows or source_rows != config.expected_rows:
        raise ValueError(
            f"corpus row expectation mismatch: expected {config.expected_rows}, "
            f"manifest={declared_rows}, shards={source_rows}"
        )
    if combined.get("complete") is not True or combined.get("rows") != config.expected_rows:
        raise ValueError("combined-sidecar metadata is incomplete or has the wrong row count")
    provenance = _mapping(
        _mapping(combined.get("inputs"), "combined-sidecar inputs").get("provenance_binding"),
        "ROVER provenance binding",
    )
    if provenance.get("trusted_release") is not True:
        raise ValueError("combined-sidecar metadata does not bind a trusted ROVER release")
    rover_hash = provenance.get("rover_archive_sha256")
    combined_hash = _mapping(combined.get("output"), "combined-sidecar output").get("sha256")
    _require_sha256(rover_hash, "ROVER archive")
    _require_sha256(combined_hash, "combined sidecar")
    return BuildExpectations(
        source_shard_count=config.expected_shards,
        source_row_count=config.expected_rows,
        rover_row_count=config.expected_rows,
        combined_row_count=config.expected_rows,
        rover_archive_sha256=rover_hash,
        combined_sidecar_sha256=combined_hash,
        source_tar_sha256=source_hashes,
    )


@dataclass
class ProductionCommands:
    """Production orchestration using the pipeline's typed module interfaces."""

    runner: CommandRunner = field(default_factory=SubprocessRunner)
    hub_pinner: Callable[[Any, CommandRunner], dict[str, HubPin]] = pin_hub_inputs
    expectation_loader: Callable[[DataConfig], BuildExpectations] = load_build_expectations
    index_builder: Callable[[DataConfig, BuildExpectations], IndexAudit] = build_index
    selection_builder: Callable[[str | Path, str | Path, str | Path, int], SelectionBundle] = create_selection_manifests
    runtime_factory: Callable[[Any], Any] = AccelerateRuntime.create
    memorization_runner: Callable[[Any, Any], Any] = run_memorization
    approval_writer: Callable[[str | Path, str, str], Path] = approve_memorization
    training_runner: Callable[..., Path] | None = None
    validation_runner: Callable[..., Mapping[str, Any]] | None = None

    def pin(self, config: BalalaikaConfig) -> CommandResult:
        pins = self.hub_pinner(config.hub, self.runner)
        return CommandResult(
            "pin",
            {
                "status": "complete",
                "pins": {name: pin.model_dump(mode="json") for name, pin in pins.items()},
            },
        )

    def prepare(self, config: BalalaikaConfig) -> CommandResult:
        expectations = self.expectation_loader(config.data)
        audit = self.index_builder(config.data, expectations)
        benchmark_path = config.hub.local_dir / "benchmark" / config.hub.benchmark_file
        selection = self.selection_builder(
            audit.index_path,
            benchmark_path,
            config.selection_dir,
            config.runtime.seed,
        )
        return CommandResult(
            "prepare",
            {
                "status": "complete",
                "index_path": audit.index_path,
                "audit_path": audit.audit_path,
                "fingerprint": audit.fingerprint,
                "joined_rows": audit.total_rows,
                "eligible_rows": audit.eligible_rows,
                "stage1_rows": audit.stage1_rows,
                "stage2_rows": audit.stage2_rows,
                "excluded_null_agreement": audit.excluded_null_agreement,
                "selection_fingerprint": selection.fingerprint,
                "memorization_samples": len(selection.memorization),
                "validation_prompts": len(selection.prompts),
                "benchmark_assignments": len(selection.benchmark_prompt_by_id),
            },
        )

    def memorize(self, config: BalalaikaConfig, *, smoke: bool) -> CommandResult:
        del smoke
        runtime = self.runtime_factory(config.memorization)
        result = self.memorization_runner(config, runtime)
        return CommandResult(
            "memorize",
            {
                "status": result.status,
                "result_path": result.result_path,
                "checkpoint": result.checkpoint,
                "wandb_run_id": result.wandb_run_id,
                "large_training_started": result.large_training_started,
            },
        )

    def approve(self, config: BalalaikaConfig, *, wandb_run_id: str) -> CommandResult:
        approver = os.environ.get("USER") or os.environ.get("LOGNAME")
        if not approver:
            raise ValueError("approval requires USER or LOGNAME to identify the operator")
        path = self.approval_writer(config.output_dir / "memorization", wandb_run_id, approver)
        return CommandResult("approve", {"status": "approved", "approval_path": path})

    def train(
        self,
        config: BalalaikaConfig,
        *,
        stage: int,
        smoke: bool,
        resume: Path | None,
        stage1_checkpoint: Path | None,
    ) -> CommandResult:
        runner = self.training_runner or _run_configured_training
        checkpoint = runner(
            config,
            stage=stage,
            smoke=smoke,
            resume=resume,
            stage1_checkpoint=stage1_checkpoint,
        )
        return CommandResult("train", {"status": "complete", "stage": stage, "checkpoint": checkpoint})

    def validate(self, config: BalalaikaConfig, *, checkpoint: Path, smoke: bool) -> CommandResult:
        runner = self.validation_runner or _run_configured_validation
        return CommandResult("validate", runner(config, checkpoint=checkpoint, smoke=smoke))

    def audit(self, config: BalalaikaConfig) -> CommandResult:
        audit_path = config.data.index_dir / "balalaika-index-audit.json"
        index_path = config.data.index_dir / "balalaika-index.sqlite3"
        audit = _read_json(audit_path, "index audit")
        if not index_path.is_file() or audit.get("index_sha256") != sha256_file(index_path):
            raise ValueError("prepared index is absent or does not match its audit hash")
        expected = {
            "total_rows": config.data.expected_rows,
            "excluded_null_agreement": config.data.expected_null_agreement,
        }
        for name, value in expected.items():
            if audit.get(name) != value:
                raise ValueError(f"production audit {name} mismatch: expected {value}, found {audit.get(name)}")
        selections = _load_selection_summary(config.selection_dir)
        _verified_pins(config)
        return CommandResult(
            "audit",
            {
                "status": "complete",
                "source_shards": config.data.expected_shards,
                "joined_rows": audit["total_rows"],
                "eligible_rows": audit["eligible_rows"],
                "stage1_rows": audit["stage1_rows"],
                "stage2_rows": audit["stage2_rows"],
                "excluded_null_agreement": audit["excluded_null_agreement"],
                "index_bytes": index_path.stat().st_size,
                "fingerprint": audit["fingerprint"],
                **selections,
            },
        )


def _run_configured_training(
    config: BalalaikaConfig,
    *,
    stage: int,
    smoke: bool,
    resume: Path | None,
    stage1_checkpoint: Path | None,
) -> Path:
    if smoke:
        raise ValueError("production --smoke uses scripts/smoke_balalaika_accelerate.py, not corpus training")
    pins = _verified_pins(config)
    data_fingerprint = _index_fingerprint(config)
    selection = _load_selection(config.selection_dir)
    lora_fingerprint = fingerprint(config.lora.model_dump(mode="json"))
    approval_path: Path | None = None
    approval_expected: Mapping[str, Any] | None = None
    if stage == 1:
        approval_path = config.output_dir / "memorization" / "memorization-approval.json"
        approval_expected = _approval_expectations(
            config,
            pins=pins,
            data_fingerprint=data_fingerprint,
            selection_fingerprint=selection.fingerprint,
            lora_fingerprint=lora_fingerprint,
        )
        # The operator gate is verified before runtime, tracking, model, or dataset setup.
        verify_approval(approval_path, approval_expected)

    runtime = AccelerateRuntime.create(config.runtime)
    stage_name = f"stage{stage}"
    stage_root = config.output_dir / stage_name
    run_state_path = stage_root / "wandb-run.json"
    run_manager = create_run_manager(
        is_main_process=runtime.rank == 0,
        config=config,
        job_type=stage_name,
        run_state_path=run_state_path,
    )
    runtime.barrier()
    run_state = _read_json(run_state_path, f"{stage_name} W&B run state")
    run_id = run_state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{stage_name} W&B run state has no run ID")
    group = config.wandb.group
    if not isinstance(group, str) or not group:
        raise ValueError("production training requires an explicit W&B group")
    stage_config = config.stage1 if stage == 1 else config.stage2
    optimization_fingerprint = fingerprint(
        {
            "stage": stage_config.model_dump(mode="json"),
            "runtime": config.runtime.model_dump(mode="json"),
            "optimization": config.optimization.model_dump(mode="json"),
        }
    )
    identity = {
        "base_revision": pins["model"]["revision"],
        "evaluator_revision": pins["gigaam"]["revision"],
        "data_fingerprint": data_fingerprint,
        "selection_fingerprint": selection.fingerprint,
        "lora_fingerprint": lora_fingerprint,
        "optimization_fingerprint": optimization_fingerprint,
        "wandb_run_id": run_id,
        "wandb_group": group,
    }
    rows = _load_benchmark_rows(config.hub.local_dir / "benchmark" / config.hub.benchmark_file)

    def evaluator_factory(boundary: EvaluationBoundary, checkpoint: Path) -> DistributedEvaluator:
        boundary_root = (
            config.output_dir
            / "validation"
            / boundary.stage
            / f"epoch-{boundary.epoch + 1:02d}-boundary-{boundary.boundary:02d}-{checkpoint.name}"
        )
        ledger = ValidationLedger(
            boundary_root,
            generation_fingerprint=fingerprint(config.generation.model_dump(mode="json")),
            asr_fingerprint=fingerprint(pins["gigaam"]),
            max_attempts=config.validation.retries,
        )
        gigaam_dir = Path(str(pins["gigaam"]["local_dir"]))
        return DistributedEvaluator(
            runtime=runtime,
            rows=rows,
            selection=selection,
            ledger=ledger,
            asr_factory=lambda device_id: GigaAMRNNT(gigaam_dir, device_id, fingerprint(pins["gigaam"])),
            run_manager=run_manager,
            expected_item_count=config.validation.benchmark_size,
            validation_root=config.output_dir / "validation",
        )

    trainer = BalalaikaTrainer(
        config,
        runtime,
        checkpoint_manager=CheckpointManager(stage_root / "checkpoints"),
        evaluator_factory=evaluator_factory,
        identity=identity,
        approval_verifier=verify_approval if stage == 1 else None,
        approval_path=approval_path,
        approval_expected=approval_expected,
        accumulation=config.runtime.accumulation,
        workers=config.runtime.workers,
        sampler_seed=config.runtime.seed,
        warmup_fraction=config.optimization.warmup_fraction,
        weight_decay=config.optimization.weight_decay,
        max_grad_norm=config.optimization.max_grad_norm,
        loss_weights=config.optimization.loss_weights,
        resume_checkpoint=resume,
        stage1_checkpoint=stage1_checkpoint,
    )
    try:
        return trainer.run_stage(stage)
    finally:
        run_manager.finish()


def _run_configured_validation(
    config: BalalaikaConfig,
    *,
    checkpoint: Path,
    smoke: bool,
) -> Mapping[str, Any]:
    del smoke
    pins = _verified_pins(config)
    selection = _load_selection(config.selection_dir)
    expected = {
        "base_revision": pins["model"]["revision"],
        "evaluator_revision": pins["gigaam"]["revision"],
        "data_fingerprint": _index_fingerprint(config),
        "selection_fingerprint": selection.fingerprint,
        "lora_fingerprint": fingerprint(config.lora.model_dump(mode="json")),
    }
    metadata = CheckpointManager(checkpoint.parent).verify(checkpoint, expected)
    return {
        "status": "complete",
        "checkpoint": checkpoint,
        "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
        "stage": metadata["stage"],
        "epoch": metadata["epoch"],
        "boundary": metadata["boundary"],
        "note": "checkpoint integrity and current input identities verified; boundary evaluation resumes in train",
    }


def _verified_pins(config: BalalaikaConfig) -> dict[str, dict[str, Any]]:
    manifest = _read_json(config.hub.local_dir / "hub-pins.json", "Hub pin manifest")
    expected_repos = {
        "model": ("model", config.hub.model_repo_id, config.hub.local_dir / "model"),
        "benchmark": ("dataset", config.hub.benchmark_repo_id, config.hub.local_dir / "benchmark"),
        "gigaam": ("model", config.hub.gigaam_repo_id, config.hub.local_dir / "gigaam"),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (kind, repo_id, local_dir) in expected_repos.items():
        pin = _mapping(manifest.get(name), f"{name} Hub pin")
        if pin.get("kind") != kind or pin.get("repo_id") != repo_id:
            raise ValueError(f"{name} Hub pin does not match configured repository")
        revision = pin.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{name} Hub pin has no immutable revision")
        if Path(str(pin.get("local_dir", ""))).resolve() != local_dir.resolve():
            raise ValueError(f"{name} Hub pin local directory does not match configuration")
        recorded_files = pin.get("files")
        if not isinstance(recorded_files, Mapping) or not recorded_files:
            raise ValueError(f"{name} Hub pin has no file hashes")
        actual_files = {
            path.relative_to(local_dir).as_posix(): sha256_file(path)
            for path in sorted(local_dir.rglob("*"))
            if path.is_file()
        }
        if dict(recorded_files) != actual_files:
            raise ValueError(f"{name} pinned files changed after download")
        result[name] = dict(pin)
    return result


def _index_fingerprint(config: BalalaikaConfig) -> str:
    path = config.data.index_dir / "balalaika-index.sqlite3"
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as database:
            row = database.execute("SELECT value FROM metadata WHERE key = 'fingerprint'").fetchone()
    except sqlite3.Error as error:
        raise ValueError(f"cannot read prepared index identity: {path}") from error
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise ValueError("prepared index has no fingerprint")
    return row[0]


def _load_selection(root: Path) -> SelectionBundle:
    memorization = _read_json(root / "memorization.json", "memorization selection")
    prompts = _read_json(root / "prompts.json", "prompt selection")
    assignments = _read_json(root / "benchmark-prompts.json", "benchmark assignments")
    audio_ids = _read_json(root / "audio-log-ids.json", "audio-log selection")
    fingerprints = {value.get("fingerprint") for value in (memorization, prompts, assignments, audio_ids)}
    seeds = {value.get("seed") for value in (memorization, prompts, assignments, audio_ids)}
    if len(fingerprints) != 1 or None in fingerprints or len(seeds) != 1:
        raise ValueError("selection manifests do not share one fingerprint and seed")
    try:
        return SelectionBundle(
            fingerprint=next(iter(fingerprints)),
            seed=next(iter(seeds)),
            memorization=[SelectedSample.model_validate(item) for item in memorization["memorization"]],
            prompts=[PromptSample.model_validate(item) for item in prompts["prompts"]],
            benchmark_prompt_by_id=assignments["benchmark_prompt_by_id"],
            audio_log_ids=audio_ids["audio_log_ids"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection manifests are malformed") from error


def _load_benchmark_rows(path: Path) -> tuple[BenchmarkRow, ...]:
    rows: list[BenchmarkRow] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    value = json.loads(line)
                    rows.append(BenchmarkRow(**_mapping(value, f"benchmark row {line_number}")))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(f"benchmark row {line_number} is malformed") from error
    except OSError as error:
        raise ValueError(f"cannot read pinned benchmark {path}: {error}") from error
    if len(rows) != 2_000 or len({row.id for row in rows}) != 2_000:
        raise ValueError(f"benchmark must contain 2,000 unique rows; found {len(rows)}")
    return tuple(rows)


def _approval_expectations(
    config: BalalaikaConfig,
    *,
    pins: Mapping[str, Mapping[str, Any]],
    data_fingerprint: str,
    selection_fingerprint: str,
    lora_fingerprint: str,
) -> dict[str, str]:
    result_path = config.output_dir / "memorization" / "memorization-result.json"
    result = _read_json(result_path, "memorization result")
    stored = _mapping(result.get("fingerprints"), "memorization result fingerprints")
    current = {
        "base_revision": pins["model"]["revision"],
        "data_fingerprint": data_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "lora_fingerprint": lora_fingerprint,
    }
    for name, value in current.items():
        if stored.get(name) != value:
            raise ValueError(f"memorization approval {name} does not match current inputs")
    expected = {
        **current,
        "checkpoint": stored.get("checkpoint"),
        "wandb_run_id": stored.get("wandb_run_id"),
        "wandb_completion": stored.get("wandb_completion"),
        "result": sha256_file(result_path),
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise ValueError("memorization result has incomplete approval fingerprints")
    return expected


def _load_selection_summary(root: Path) -> dict[str, Any]:
    memorization = _read_json(root / "memorization.json", "memorization selection")
    prompts = _read_json(root / "prompts.json", "prompt selection")
    assignments = _read_json(root / "benchmark-prompts.json", "benchmark assignments")
    audio_ids = _read_json(root / "audio-log-ids.json", "audio-log selection")
    fingerprints = {value.get("fingerprint") for value in (memorization, prompts, assignments, audio_ids)}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("selection manifests do not share one fingerprint")
    memorization_rows = memorization.get("memorization")
    prompt_rows = prompts.get("prompts")
    assignment_rows = assignments.get("benchmark_prompt_by_id")
    fixed_audio = audio_ids.get("audio_log_ids")
    counts = (
        len(memorization_rows) if isinstance(memorization_rows, list) else -1,
        len(prompt_rows) if isinstance(prompt_rows, list) else -1,
        len(assignment_rows) if isinstance(assignment_rows, dict) else -1,
        len(fixed_audio) if isinstance(fixed_audio, list) else -1,
    )
    if counts != (4, 20, 2_000, 4):
        raise ValueError(f"selection counts are invalid: {counts}")
    return {
        "selection_fingerprint": next(iter(fingerprints)),
        "memorization_samples": counts[0],
        "validation_prompts": counts[1],
        "benchmark_assignments": counts[2],
        "audio_log_ids": counts[3],
    }


def _read_json(path: Path, label: str) -> _JSON:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is malformed: {path}") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> _JSON:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must have a lowercase SHA-256 digest")
