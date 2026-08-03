"""Resumable distributed Balalaika generation, ASR, and publication."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, mkdtemp
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf
import torch

from .artifacts import atomic_json, fingerprint, sha256_file
from .ledger import ValidationClaim, ValidationLedger
from .metrics import AggregateScore, BenchmarkRow, ItemScore, aggregate_scores, score_utterance
from .tracking import ValidationItem, ValidationPayload

if TYPE_CHECKING:
    from .runtime import TrainingRuntime
    from .selection import SelectionBundle


class IncompleteValidation(RuntimeError):
    """A boundary does not contain the exact expected complete item set."""


class EvaluationIntegrityError(RuntimeError):
    """A durable evaluator input or output no longer matches its identity."""


@dataclass(frozen=True)
class _RunContext:
    checkpoint_fingerprint: str
    boundary: Mapping[str, object]
    input_fingerprint: str
    item_inputs: Mapping[int, Mapping[str, object]]
    item_seeds: Mapping[int, int]


@dataclass(frozen=True)
class _RNGState:
    cpu: torch.Tensor
    cuda_device: torch.device | None
    cuda: torch.Tensor | None


@dataclass(frozen=True)
class _RetentionCandidate:
    boundary_dir: Path
    record: Mapping[str, object]
    audio: tuple[Mapping[str, object], ...]
    ordering_key: tuple[int, str, str]
    source_wavs_exist: bool


_COMPLETION_KEYS = frozenset(
    {
        "version",
        "status",
        "input_fingerprint",
        "selection_fingerprint",
        "checkpoint_fingerprint",
        "boundary",
        "item_count",
        "artifacts",
        "audio",
        "snapshot_fingerprint",
        "published_at",
    }
)
_AUDIO_KEYS = frozenset({"id", "source_path", "retained_path", "sha256"})
_ITEM_SNAPSHOT_KEYS = frozenset({"row", "score", "prompt_id", "asr_hypothesis", "wav_path", "wav_sha256"})


def partition_ids(ids: Iterable[int], *, rank: int, world_size: int) -> tuple[int, ...]:
    """Return sorted IDs whose zero-based positions belong to ``rank``."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if rank >= world_size:
        raise ValueError("rank must be smaller than world_size")
    materialized = tuple(ids)
    if any(isinstance(item_id, bool) or not isinstance(item_id, int) for item_id in materialized):
        raise TypeError("validation item IDs must be integers")
    return tuple(item_id for position, item_id in enumerate(sorted(materialized)) if position % world_size == rank)


def require_exact_complete(expected_ids: Iterable[int], complete_ids: Iterable[int]) -> tuple[int, ...]:
    """Validate and return one exact, unique completed boundary."""
    expected = tuple(expected_ids)
    completed = tuple(complete_ids)
    expected_set = set(expected)
    completed_set = set(completed)
    duplicates = sorted(item_id for item_id in completed_set if completed.count(item_id) > 1)
    missing = sorted(expected_set - completed_set)
    unexpected = sorted(completed_set - expected_set)
    if (
        len(expected) != len(expected_set)
        or len(completed) != len(completed_set)
        or missing
        or unexpected
        or len(completed) != len(expected)
    ):
        suffix = f"; missing={missing}; unexpected={unexpected}"
        if duplicates:
            suffix += f"; duplicates={duplicates}"
        raise IncompleteValidation(
            f"Incomplete validation: {len(completed_set & expected_set)}/{len(expected_set)} complete{suffix}."
        )
    return tuple(sorted(completed))


def _generation_settings(value: Mapping[str, object] | Any | None) -> dict[str, object]:
    if value is None:
        raise ValueError("evaluator generation_settings are required")
    if callable(getattr(value, "model_dump", None)):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        raise ValueError("evaluator generation_settings must be a mapping")
    settings = dict(value)
    for name in ("cfg_value", "inference_timesteps", "max_length"):
        if name not in settings:
            raise ValueError(f"evaluator generation_settings are missing {name}")
    return settings


class DistributedEvaluator:
    """Coordinate one immutable full-validation boundary across all ranks.

    The caller supplies a boundary-local ledger.  This keeps Task 11 focused on
    evaluator semantics while the trainer owns checkpoint and directory
    lifecycle.  ``payload_factory`` is a smoke-test seam; production uses the
    strict 2,000-row :class:`ValidationPayload`.
    """

    def __init__(
        self,
        *,
        runtime: TrainingRuntime,
        rows: Sequence[BenchmarkRow],
        selection: SelectionBundle,
        ledger: ValidationLedger,
        asr_factory: Callable[[int], Any],
        run_manager: Any,
        expected_item_count: int = 2_000,
        validation_root: str | Path | None = None,
        payload_factory: Callable[..., ValidationPayload] = ValidationPayload,
        generation_settings: Mapping[str, object] | Any | None = None,
    ) -> None:
        if (
            isinstance(expected_item_count, bool)
            or not isinstance(expected_item_count, int)
            or expected_item_count <= 0
        ):
            raise ValueError("expected_item_count must be a positive integer")
        materialized_rows = tuple(sorted(rows, key=lambda row: row.id))
        if len(materialized_rows) != expected_item_count:
            raise ValueError(
                f"Evaluator requires exactly {expected_item_count:,} benchmark rows, found {len(materialized_rows):,}."
            )
        if any(not isinstance(row, BenchmarkRow) for row in materialized_rows):
            raise TypeError("Evaluator rows must be BenchmarkRow instances")
        row_ids = tuple(row.id for row in materialized_rows)
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Evaluator benchmark row IDs must be unique")
        assignments = dict(getattr(selection, "benchmark_prompt_by_id", {}))
        if set(assignments) != set(row_ids):
            raise ValueError("Selection prompt assignments must cover exactly the evaluator benchmark IDs")
        prompt_list = tuple(getattr(selection, "prompts", ()))
        prompt_by_id = {getattr(prompt, "prompt_id", None): prompt for prompt in prompt_list}
        if len(prompt_by_id) != len(prompt_list) or None in prompt_by_id:
            raise ValueError("Selection prompt IDs must be non-empty and unique")
        if set(assignments.values()) - set(prompt_by_id):
            raise ValueError("Selection prompt assignments reference an unknown prompt")
        audio_log_ids = tuple(getattr(selection, "audio_log_ids", ()))
        if len(audio_log_ids) != 4 or len(set(audio_log_ids)) != 4 or set(audio_log_ids) - set(row_ids):
            raise ValueError("Selection must provide exactly four unique evaluator audio-log IDs")
        selection_fingerprint = getattr(selection, "fingerprint", None)
        if not isinstance(selection_fingerprint, str) or not selection_fingerprint:
            raise ValueError("Selection fingerprint must be a non-empty string")
        if payload_factory is ValidationPayload and expected_item_count != 2_000:
            raise ValueError("The production ValidationPayload requires exactly 2,000 evaluator rows")

        self.runtime = runtime
        self.rows = materialized_rows
        self._row_by_id = MappingProxyType({row.id: row for row in materialized_rows})
        self.selection = selection
        self._prompt_by_id = MappingProxyType(prompt_by_id)
        self.ledger = ledger
        self.asr_factory = asr_factory
        self.run_manager = run_manager
        self.expected_item_count = expected_item_count
        self.validation_root = Path(validation_root) if validation_root is not None else ledger.root.parent
        self.payload_factory = payload_factory
        self.generation_settings = _generation_settings(generation_settings)
        if fingerprint(self.generation_settings) != self.ledger.generation_fingerprint:
            raise ValueError("generation settings do not match the validation ledger fingerprint")

    def item_seed(self, item_id: int, checkpoint: object, boundary: object) -> int:
        """Return the caller-current deterministic seed for one item."""
        context = self._context(checkpoint, boundary)
        try:
            return context.item_seeds[item_id]
        except KeyError as error:
            raise ValueError(f"Unknown evaluator item ID: {item_id}") from error

    def run_rank(
        self, model: torch.nn.Module, audio_vae: torch.nn.Module, checkpoint: object, boundary: object
    ) -> set[int]:
        """Run generation then ASR for this rank without collective operations."""
        context = self._context(checkpoint, boundary)
        local_ids = partition_ids(self._row_by_id, rank=self.runtime.rank, world_size=self.runtime.world_size)
        unwrapped = self.runtime.unwrap(model)
        model_training = _training_mode(unwrapped)
        vae_training = _training_mode(audio_vae)
        had_audio_vae = hasattr(unwrapped, "audio_vae")
        previous_audio_vae = getattr(unwrapped, "audio_vae", None)
        rng_state = _capture_rng(self.runtime)
        claims: dict[int, ValidationClaim] = {}
        try:
            _set_training(unwrapped, False)
            _set_training(audio_vae, False)
            setattr(unwrapped, "audio_vae", audio_vae)
            try:
                claims = self._generate_local(unwrapped, local_ids, context)
            finally:
                _restore_audio_vae(unwrapped, had_audio_vae, previous_audio_vae)
            self._transcribe_local(local_ids, context, claims)
        finally:
            _restore_audio_vae(unwrapped, had_audio_vae, previous_audio_vae)
            _restore_training(audio_vae, vae_training)
            _restore_training(unwrapped, model_training)
            _restore_rng(rng_state)
        return {
            item_id
            for item_id in local_ids
            if self.ledger.is_complete(
                item_id, inputs=context.item_inputs[item_id], attempt_seed=context.item_seeds[item_id]
            )
        }

    def aggregate(self, checkpoint: object, boundary: object, *, write_artifacts: bool = True) -> ValidationPayload:
        """Build the exact deterministic payload from caller-current ledger records."""
        context = self._context(checkpoint, boundary)
        expected = {item_id: (context.item_inputs[item_id], context.item_seeds[item_id]) for item_id in self._row_by_id}
        completed = require_exact_complete(self._row_by_id, self.ledger.complete_ids(expected))
        validation_items: list[ValidationItem] = []
        wav_paths: dict[int, Path] = {}
        wav_hashes: dict[int, str] = {}
        generation_seconds = 0.0
        asr_seconds = 0.0
        generation_failures = 0
        asr_failures = 0
        for item_id in completed:
            record = self._item_record(item_id)
            if record.get("id") != item_id:
                raise EvaluationIntegrityError(f"Ledger record ID does not match filename for item {item_id}")
            generation = _mapping(record.get("generation"), f"generation record for item {item_id}")
            asr = _mapping(record.get("asr"), f"ASR record for item {item_id}")
            attempts = _mapping(record.get("attempts"), f"attempt record for item {item_id}")
            wav_path = Path(str(generation.get("wav_path")))
            wav_hash = str(generation.get("wav_sha256"))
            if not wav_path.is_file() or sha256_file(wav_path) != wav_hash:
                raise EvaluationIntegrityError(f"Generated WAV content changed for item {item_id}")
            hypothesis = asr.get("hypothesis")
            if not isinstance(hypothesis, str):
                raise EvaluationIntegrityError(f"ASR hypothesis is not a successful string for item {item_id}")
            row = self._row_by_id[item_id]
            score = score_utterance(row, hypothesis)
            prompt_id = str(context.item_inputs[item_id]["prompt_id"])
            validation_items.append(
                ValidationItem(row=row, score=score, prompt_id=prompt_id, asr_hypothesis=hypothesis)
            )
            wav_paths[item_id] = wav_path
            wav_hashes[item_id] = wav_hash
            generation_seconds += _nonnegative_time(generation.get("elapsed_seconds"))
            asr_seconds += _nonnegative_time(asr.get("elapsed_seconds"))
            generation_failures += max(0, _attempt_count(attempts.get("generation")) - 1)
            asr_failures += max(0, _attempt_count(attempts.get("asr")) - 1)
        if len(set(wav_paths.values())) != self.expected_item_count:
            raise EvaluationIntegrityError("Complete validation records must identify unique generated WAV files")

        metrics = aggregate_scores([item.score for item in validation_items])
        timings = {
            "generation_seconds": generation_seconds,
            "asr_seconds": asr_seconds,
            "generation_mean_seconds": generation_seconds / self.expected_item_count,
            "asr_mean_seconds": asr_seconds / self.expected_item_count,
        }
        failure_counts = {"generation": generation_failures, "asr": asr_failures}
        audio_paths = {item_id: wav_paths[item_id] for item_id in tuple(getattr(self.selection, "audio_log_ids"))}
        payload = self.payload_factory(
            metrics=metrics,
            category_metrics=metrics.category_metrics,
            items=tuple(validation_items),
            audio_paths=audio_paths,
            artifact_dir=self.ledger.root,
            timings=timings,
            failure_counts=failure_counts,
            stage_progress=float(context.boundary["stage_progress"]),
            input_fingerprint=context.input_fingerprint,
            expected_audio_ids=tuple(getattr(self.selection, "audio_log_ids")),
            selection_fingerprint=str(getattr(self.selection, "fingerprint")),
        )
        if write_artifacts:
            _atomic_jsonl(
                self.ledger.root / "items.jsonl",
                [_item_snapshot(item, wav_paths[item.row.id], wav_hashes[item.row.id]) for item in validation_items],
            )
            atomic_json(
                self.ledger.root / "metrics.json",
                {
                    "version": 1,
                    "input_fingerprint": context.input_fingerprint,
                    "metrics": _aggregate_snapshot(metrics),
                    "category_metrics": {
                        category: _aggregate_snapshot(category_metrics)
                        for category, category_metrics in metrics.category_metrics.items()
                    },
                    "timings": timings,
                    "failure_counts": failure_counts,
                },
            )
        return payload

    def run(
        self, model: torch.nn.Module, audio_vae: torch.nn.Module, checkpoint: object, boundary: object
    ) -> ValidationPayload:
        """Run all phases, publish rank-zero tracking/completion, then verify on every rank."""
        context = self._context(checkpoint, boundary)
        completion_path = self.ledger.root / "validation-complete.json"
        if completion_path.is_file():
            status_path = self.ledger.root / "resume-retention-status.json"
            if self.runtime.rank == 0:
                try:
                    record = self._verify_completion(context, require_retained=False)
                    self._apply_retention()
                    status = {
                        "version": 1,
                        "status": "success",
                        "input_fingerprint": context.input_fingerprint,
                        "completion_snapshot_fingerprint": record["snapshot_fingerprint"],
                    }
                except Exception as error:
                    status = {
                        "version": 1,
                        "status": "failed",
                        "input_fingerprint": context.input_fingerprint,
                        "error": _error_snapshot(error),
                    }
                atomic_json(status_path, status)
            self.runtime.barrier()
            status = _read_json(status_path, "resume retention status")
            if status.get("input_fingerprint") != context.input_fingerprint:
                raise EvaluationIntegrityError(f"Resume retention status belongs to different inputs: {status_path}")
            if status.get("version") != 1 or status.get("status") not in {"success", "failed"}:
                raise EvaluationIntegrityError(f"Resume retention status is malformed: {status_path}")
            if status["status"] == "failed":
                if set(status) != {"version", "status", "input_fingerprint", "error"}:
                    raise EvaluationIntegrityError(f"Resume retention failure status is malformed: {status_path}")
                _raise_error_snapshot(_mapping(status.get("error"), "resume retention error"))
            if set(status) != {
                "version",
                "status",
                "input_fingerprint",
                "completion_snapshot_fingerprint",
            }:
                raise EvaluationIntegrityError(f"Resume retention success status is malformed: {status_path}")
            record = self._verify_completion(context, require_retained=True)
            if status["completion_snapshot_fingerprint"] != record["snapshot_fingerprint"]:
                raise EvaluationIntegrityError(f"Resume retention completion identity changed: {status_path}")
            return self._payload_from_artifacts(context, record)

        local_status_path = self.ledger.root / "rank-status" / f"rank-{self.runtime.rank:02d}.json"
        status_write_error: BaseException | None = None
        run_error: BaseException | None = None
        try:
            completed = self.run_rank(model, audio_vae, checkpoint, boundary)
            local_status = {
                "version": 1,
                "status": "success",
                "rank": self.runtime.rank,
                "input_fingerprint": context.input_fingerprint,
                "completed_ids": sorted(completed),
            }
        except Exception as error:
            run_error = error
            local_status = {
                "version": 1,
                "status": "failed",
                "rank": self.runtime.rank,
                "input_fingerprint": context.input_fingerprint,
                "error": _error_snapshot(error),
            }
        finally:
            run_outcome = torch.tensor([1 if run_error is None else 0], dtype=torch.int8, device=self.runtime.device)
            self.runtime.gather(run_outcome)
        try:
            atomic_json(
                local_status_path,
                local_status,
            )
        except BaseException as error:
            status_write_error = error
        finally:
            outcome = torch.tensor(
                [1 if status_write_error is None else 0], dtype=torch.int8, device=self.runtime.device
            )
            outcomes = self.runtime.gather(outcome).reshape(-1)
        if not bool((outcomes == 1).all().item()):
            if self.runtime.rank == 0:
                try:
                    atomic_json(
                        self.ledger.root / "validation-failed.json",
                        {
                            "version": 1,
                            "status": "failed",
                            "input_fingerprint": context.input_fingerprint,
                            "error": _error_snapshot(
                                status_write_error or RuntimeError("peer rank status publication failed")
                            ),
                        },
                    )
                except BaseException:
                    pass
            self.runtime.barrier()
            raise RuntimeError(
                "validation rank status publication failed: status write failure"
            ) from status_write_error
        self.runtime.barrier()

        if self.runtime.rank == 0:
            try:
                self._require_rank_success(context)
                payload = self.aggregate(checkpoint, boundary)
                self.run_manager.log_validation(payload, global_step=int(context.boundary["global_step"]))
                atomic_json(completion_path, self._completion_record(context, payload))
                self._apply_retention()
            except Exception as error:
                atomic_json(
                    self.ledger.root / "validation-failed.json",
                    {
                        "version": 1,
                        "status": "failed",
                        "input_fingerprint": context.input_fingerprint,
                        "error": _error_snapshot(error),
                    },
                )
        self.runtime.barrier()

        if completion_path.is_file():
            record = self._verify_completion(context, require_retained=True)
            return self._payload_from_artifacts(context, record)
        self._raise_published_failure(context)
        raise RuntimeError("Validation failed without a durable failure record")

    def _generate_local(self, model: Any, local_ids: Sequence[int], context: _RunContext) -> dict[int, ValidationClaim]:
        claims: dict[int, ValidationClaim] = {}
        for item_id in local_ids:
            inputs = context.item_inputs[item_id]
            seed = context.item_seeds[item_id]
            while not self.ledger.is_complete(item_id, inputs=inputs, attempt_seed=seed):
                claim = self.ledger.claim(item_id, rank=self.runtime.rank, inputs=inputs, attempt_seed=seed)
                if claim is None:
                    break
                record = self._item_record(item_id)
                if _mapping(record.get("generation"), "generation").get("status") == "success":
                    claims[item_id] = claim
                    break
                started = time.monotonic()
                try:
                    generated = model.generate(
                        target_text=self._row_by_id[item_id].stressed,
                        prompt_text=str(inputs["prompt_text"]),
                        prompt_wav_path=str(inputs["prompt_wav_path"]),
                        seed=seed,
                        cfg_value=self.generation_settings["cfg_value"],
                        inference_timesteps=self.generation_settings["inference_timesteps"],
                        max_len=self.generation_settings["max_length"],
                    )
                    wav_path = self.ledger.wav_path(item_id, rank=self.runtime.rank)
                    _atomic_wav(wav_path, generated, _sample_rate(model))
                    self.ledger.record_generation(
                        item_id,
                        claim=claim,
                        wav_sha256=sha256_file(wav_path),
                        path=wav_path,
                        elapsed_seconds=time.monotonic() - started,
                    )
                    claims[item_id] = claim
                    break
                except Exception as error:
                    self.ledger.record_failure(item_id, claim=claim, stage="generation", exception=error)
        return claims

    def _transcribe_local(
        self,
        local_ids: Sequence[int],
        context: _RunContext,
        claims: dict[int, ValidationClaim],
    ) -> None:
        pending = [
            item_id
            for item_id in local_ids
            if not self.ledger.is_complete(
                item_id, inputs=context.item_inputs[item_id], attempt_seed=context.item_seeds[item_id]
            )
            and _mapping(self._item_record_if_present(item_id).get("generation"), "generation").get("status")
            == "success"
        ]
        if not pending:
            return
        asr: Any | None = None
        try:
            for item_id in pending:
                inputs = context.item_inputs[item_id]
                seed = context.item_seeds[item_id]
                claim = claims.pop(item_id, None)
                while not self.ledger.is_complete(item_id, inputs=inputs, attempt_seed=seed):
                    replacement = self.ledger.claim(item_id, rank=self.runtime.rank, inputs=inputs, attempt_seed=seed)
                    claim = replacement if replacement is not None else claim
                    if claim is None:
                        raise EvaluationIntegrityError(f"No live validation claim is available for ASR item {item_id}")
                    record = self._item_record(item_id)
                    generation = _mapping(record.get("generation"), "generation")
                    if generation.get("status") != "success":
                        break
                    started = time.monotonic()
                    try:
                        if asr is None:
                            asr = self.asr_factory(_device_id(self.runtime))
                        hypothesis = asr.transcribe(Path(str(generation["wav_path"])))
                        self.ledger.record_asr(
                            item_id,
                            claim=claim,
                            hypothesis=hypothesis,
                            elapsed_seconds=time.monotonic() - started,
                        )
                        claim = None
                        break
                    except Exception as error:
                        self.ledger.record_failure(item_id, claim=claim, stage="asr", exception=error)
                        claim = None
        finally:
            if asr is not None:
                asr.close()

    def _context(self, checkpoint: object, boundary: object) -> _RunContext:
        checkpoint_fingerprint = _checkpoint_fingerprint(checkpoint)
        boundary_value = _boundary_value(boundary)
        prompt_values: dict[str, dict[str, str]] = {}
        for prompt_id, prompt in self._prompt_by_id.items():
            wav_path = Path(getattr(prompt, "wav_path", ""))
            declared_hash = getattr(prompt, "wav_sha256", None)
            if not wav_path.is_file():
                raise FileNotFoundError(f"Validation prompt WAV is missing: {wav_path}")
            actual_hash = sha256_file(wav_path)
            if not isinstance(declared_hash, str) or actual_hash != declared_hash:
                raise EvaluationIntegrityError(f"Validation prompt WAV hash changed: {wav_path}")
            text = getattr(prompt, "text", None)
            if not isinstance(text, str) or not text:
                raise EvaluationIntegrityError(f"Validation prompt text is missing for {prompt_id}")
            prompt_values[str(prompt_id)] = {
                "text": text,
                "wav_path": str(wav_path),
                "wav_sha256": actual_hash,
            }
        assignments = dict(getattr(self.selection, "benchmark_prompt_by_id"))
        identity = {
            "version": 1,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "boundary": boundary_value,
            "selection_fingerprint": str(getattr(self.selection, "fingerprint")),
            "generation_fingerprint": self.ledger.generation_fingerprint,
            "generation_settings": self.generation_settings,
            "asr_fingerprint": self.ledger.asr_fingerprint,
            "rows": [asdict(row) for row in self.rows],
            "assignments": [[item_id, assignments[item_id]] for item_id in self._row_by_id],
            "prompts": prompt_values,
            "audio_log_ids": list(getattr(self.selection, "audio_log_ids")),
        }
        input_fingerprint = fingerprint(identity)
        item_inputs: dict[int, Mapping[str, object]] = {}
        item_seeds: dict[int, int] = {}
        for row in self.rows:
            prompt_id = assignments[row.id]
            prompt = prompt_values[prompt_id]
            inputs: dict[str, object] = {
                "validation_input_fingerprint": input_fingerprint,
                "selection_fingerprint": str(getattr(self.selection, "fingerprint")),
                "checkpoint_fingerprint": checkpoint_fingerprint,
                "boundary": boundary_value,
                "row": asdict(row),
                "prompt_id": prompt_id,
                "prompt_text": prompt["text"],
                "prompt_wav_path": prompt["wav_path"],
                "prompt_wav_sha256": prompt["wav_sha256"],
                "generation_settings": self.generation_settings,
            }
            item_fingerprint = fingerprint(inputs)
            seed = int(fingerprint({"validation": input_fingerprint, "item": item_fingerprint})[:16], 16)
            item_inputs[row.id] = MappingProxyType(inputs)
            item_seeds[row.id] = seed
        return _RunContext(
            checkpoint_fingerprint=checkpoint_fingerprint,
            boundary=MappingProxyType(boundary_value),
            input_fingerprint=input_fingerprint,
            item_inputs=MappingProxyType(item_inputs),
            item_seeds=MappingProxyType(item_seeds),
        )

    def _item_record(self, item_id: int) -> dict[str, object]:
        path = self.ledger.item_path(item_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationIntegrityError(f"Ledger item record is unreadable: {path}") from error
        if not isinstance(value, dict):
            raise EvaluationIntegrityError(f"Ledger item record is malformed: {path}")
        return value

    def _item_record_if_present(self, item_id: int) -> dict[str, object]:
        return self._item_record(item_id) if self.ledger.item_path(item_id).is_file() else {}

    def _require_rank_success(self, context: _RunContext) -> None:
        for rank in range(self.runtime.world_size):
            path = self.ledger.root / "rank-status" / f"rank-{rank:02d}.json"
            record = _read_json(path, "rank status")
            if (
                record.get("status") != "success"
                or record.get("rank") != rank
                or record.get("input_fingerprint") != context.input_fingerprint
            ):
                error = record.get("error")
                raise RuntimeError(f"Validation rank {rank} failed before aggregation: {error}")
            expected = partition_ids(self._row_by_id, rank=rank, world_size=self.runtime.world_size)
            if tuple(record.get("completed_ids", ())) != expected:
                raise IncompleteValidation(
                    f"Validation rank {rank} completed IDs do not equal its deterministic partition"
                )

    def _completion_record(self, context: _RunContext, payload: Any) -> dict[str, object]:
        metrics_path = self.ledger.root / "metrics.json"
        items_path = self.ledger.root / "items.jsonl"
        audio: list[dict[str, object]] = []
        for item_id, path_value in payload.audio_paths.items():
            path = Path(path_value)
            retained = self.validation_root / "retained-audio" / self.ledger.root.name / f"{item_id:05d}.wav"
            audio.append(
                {
                    "id": item_id,
                    "source_path": str(path),
                    "retained_path": str(retained),
                    "sha256": sha256_file(path),
                }
            )
        identity = {
            "version": 1,
            "status": "complete",
            "input_fingerprint": context.input_fingerprint,
            "selection_fingerprint": str(getattr(self.selection, "fingerprint")),
            "checkpoint_fingerprint": context.checkpoint_fingerprint,
            "boundary": dict(context.boundary),
            "item_count": self.expected_item_count,
            "artifacts": {
                "metrics.json": sha256_file(metrics_path),
                "items.jsonl": sha256_file(items_path),
            },
            "audio": audio,
        }
        return {**identity, "snapshot_fingerprint": fingerprint(identity), "published_at": _timestamp()}

    def _verify_completion(self, context: _RunContext, *, require_retained: bool) -> dict[str, object]:
        path = self.ledger.root / "validation-complete.json"
        record = _read_json(path, "validation completion")
        expected = {
            "status": "complete",
            "input_fingerprint": context.input_fingerprint,
            "selection_fingerprint": str(getattr(self.selection, "fingerprint")),
            "checkpoint_fingerprint": context.checkpoint_fingerprint,
            "boundary": dict(context.boundary),
            "item_count": self.expected_item_count,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise EvaluationIntegrityError(f"Validation completion identity does not match current inputs: {path}")
        fingerprint_value = {
            key: value for key, value in record.items() if key not in {"snapshot_fingerprint", "published_at"}
        }
        if record.get("snapshot_fingerprint") != fingerprint(fingerprint_value):
            raise EvaluationIntegrityError(f"Validation completion snapshot fingerprint changed: {path}")
        artifacts = _mapping(record.get("artifacts"), "validation completion artifacts")
        for name in ("metrics.json", "items.jsonl"):
            artifact = self.ledger.root / name
            expected_hash = artifacts.get(name)
            if not artifact.is_file() or not isinstance(expected_hash, str) or sha256_file(artifact) != expected_hash:
                raise EvaluationIntegrityError(f"Completed validation {name} content changed: {artifact}")
        audio = record.get("audio")
        expected_audio_ids = tuple(getattr(self.selection, "audio_log_ids"))
        if (
            not isinstance(audio, list)
            or tuple(item.get("id") for item in audio if isinstance(item, dict)) != expected_audio_ids
        ):
            raise EvaluationIntegrityError("Completed validation audio IDs do not match the fixed selection order")
        if require_retained:
            for item in audio:
                retained = Path(str(item["retained_path"]))
                if not retained.is_file() or sha256_file(retained) != item.get("sha256"):
                    raise EvaluationIntegrityError(f"Retained validation audio content changed: {retained}")
        return record

    def _payload_from_artifacts(self, context: _RunContext, completion: Mapping[str, object]) -> ValidationPayload:
        metrics_record = _read_json(self.ledger.root / "metrics.json", "validation metrics")
        metric_value = _mapping(metrics_record.get("metrics"), "aggregate metrics")
        metrics = _aggregate_from_snapshot(metric_value)
        categories_value = _mapping(metrics_record.get("category_metrics"), "category metrics")
        categories = {
            str(category): _aggregate_from_snapshot(_mapping(value, f"category metrics {category}"))
            for category, value in categories_value.items()
        }
        metrics = AggregateScore(
            **{
                field.name: getattr(metrics, field.name)
                for field in fields(AggregateScore)
                if field.name != "category_metrics"
            },
            category_metrics=categories,
        )
        items: list[ValidationItem] = []
        try:
            with (self.ledger.root / "items.jsonl").open("r", encoding="utf-8") as stream:
                snapshots = [json.loads(line) for line in stream if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationIntegrityError("Completed validation items.jsonl is unreadable") from error
        if len(snapshots) != self.expected_item_count:
            raise EvaluationIntegrityError("Completed validation items.jsonl has the wrong item count")
        for snapshot in snapshots:
            row = BenchmarkRow(**_mapping(snapshot.get("row"), "item row"))
            score_value = dict(_mapping(snapshot.get("score"), "item score"))
            score_value["gold_number_span"] = tuple(score_value["gold_number_span"])
            score_value["hypothesis_number_span"] = tuple(score_value["hypothesis_number_span"])
            items.append(
                ValidationItem(
                    row=row,
                    score=ItemScore(**score_value),
                    prompt_id=str(snapshot["prompt_id"]),
                    asr_hypothesis=str(snapshot["asr_hypothesis"]),
                )
            )
        audio_records = completion["audio"]
        audio_paths = {int(item["id"]): Path(str(item["retained_path"])) for item in audio_records}
        return self.payload_factory(
            metrics=metrics,
            category_metrics=categories,
            items=tuple(items),
            audio_paths=audio_paths,
            artifact_dir=self.ledger.root,
            timings=dict(_mapping(metrics_record.get("timings"), "validation timings")),
            failure_counts=dict(_mapping(metrics_record.get("failure_counts"), "validation failures")),
            stage_progress=float(context.boundary["stage_progress"]),
            input_fingerprint=context.input_fingerprint,
            expected_audio_ids=tuple(getattr(self.selection, "audio_log_ids")),
            selection_fingerprint=str(getattr(self.selection, "fingerprint")),
        )

    def _apply_retention(self) -> None:
        if not self.validation_root.is_dir():
            return
        completed: list[_RetentionCandidate] = []
        for boundary_dir in sorted(self.validation_root.iterdir()):
            if not boundary_dir.is_dir() or boundary_dir.name == "retained-audio":
                continue
            completion_path = boundary_dir / "validation-complete.json"
            if not completion_path.is_file():
                continue
            completed.append(self._validate_retention_completion(boundary_dir))
        if not completed:
            return

        staged: list[tuple[_RetentionCandidate, Path]] = []
        try:
            for candidate in completed:
                retained_dir = self._retained_dir(candidate.boundary_dir)
                if retained_dir.is_symlink():
                    raise EvaluationIntegrityError(f"Retained validation boundary is unsafe: {retained_dir}")
                if retained_dir.exists():
                    self._validate_retained_boundary(candidate, retained_dir)
                    continue
                staged.append((candidate, self._stage_retained_boundary(candidate)))
            for candidate, staging_dir in staged:
                retained_dir = self._retained_dir(candidate.boundary_dir)
                if retained_dir.exists() or retained_dir.is_symlink():
                    raise EvaluationIntegrityError(
                        f"Retained validation boundary appeared concurrently: {retained_dir}"
                    )
                os.replace(staging_dir, retained_dir)
                _fsync_directory(retained_dir.parent)
            for candidate in completed:
                self._validate_retained_boundary(candidate, self._retained_dir(candidate.boundary_dir))
        finally:
            for _, staging_dir in staged:
                if staging_dir.exists():
                    shutil.rmtree(staging_dir)

        latest = max(completed, key=lambda candidate: candidate.ordering_key).boundary_dir
        for candidate in completed:
            wavs = candidate.boundary_dir / "wavs"
            if candidate.boundary_dir != latest and wavs.exists():
                if wavs.is_symlink() or not wavs.is_dir():
                    raise EvaluationIntegrityError(f"Validation WAV retention target is unsafe: {wavs}")
                shutil.rmtree(wavs)

    def _validate_retention_completion(self, boundary_dir: Path) -> _RetentionCandidate:
        completion_path = boundary_dir / "validation-complete.json"
        record = _read_json(completion_path, "retention completion")
        if set(record) != _COMPLETION_KEYS:
            raise EvaluationIntegrityError(f"Completed validation has an invalid schema: {completion_path}")
        if record.get("version") != 1 or record.get("status") != "complete":
            raise EvaluationIntegrityError(f"Completed validation has an invalid identity: {completion_path}")
        for name in ("input_fingerprint", "selection_fingerprint", "checkpoint_fingerprint", "published_at"):
            if not isinstance(record.get(name), str) or not record[name]:
                raise EvaluationIntegrityError(f"Completed validation has an invalid {name}: {completion_path}")
        if record["selection_fingerprint"] != str(getattr(self.selection, "fingerprint")):
            raise EvaluationIntegrityError(f"Completed validation selection identity changed: {completion_path}")
        if record.get("item_count") != self.expected_item_count:
            raise EvaluationIntegrityError(f"Completed validation item count changed: {completion_path}")
        boundary_record = _mapping(record.get("boundary"), "retention boundary")
        if set(boundary_record) != {"stage", "epoch", "boundary", "global_step", "stage_progress"}:
            raise EvaluationIntegrityError(f"Completed validation boundary schema changed: {completion_path}")
        normalized_boundary = _boundary_value(boundary_record)
        if normalized_boundary != boundary_record:
            raise EvaluationIntegrityError(f"Completed validation boundary identity changed: {completion_path}")
        canonical = {key: value for key, value in record.items() if key not in {"snapshot_fingerprint", "published_at"}}
        if record.get("snapshot_fingerprint") != fingerprint(canonical):
            raise EvaluationIntegrityError(f"Completed validation snapshot fingerprint changed: {completion_path}")

        artifacts = _mapping(record.get("artifacts"), "retention artifacts")
        if set(artifacts) != {"metrics.json", "items.jsonl"}:
            raise EvaluationIntegrityError(f"Completed validation artifact schema changed: {completion_path}")
        for name, expected_hash in artifacts.items():
            artifact = boundary_dir / name
            if not isinstance(expected_hash, str) or not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise EvaluationIntegrityError(f"Completed validation {name} content changed: {artifact}")

        snapshots = _read_item_snapshots(boundary_dir / "items.jsonl", self.expected_item_count)
        items: list[ValidationItem] = []
        item_audio: dict[int, tuple[Path, str]] = {}
        wav_dir = boundary_dir / "wavs"
        if wav_dir.is_symlink():
            raise EvaluationIntegrityError(f"Completed validation WAV tree is unsafe: {wav_dir}")
        source_wavs_exist = wav_dir.exists()
        if source_wavs_exist and not wav_dir.is_dir():
            raise EvaluationIntegrityError(f"Completed validation WAV tree is unsafe: {wav_dir}")
        wav_root = wav_dir.resolve()
        for snapshot in snapshots:
            if not isinstance(snapshot, Mapping) or set(snapshot) != _ITEM_SNAPSHOT_KEYS:
                raise EvaluationIntegrityError(f"Completed validation item snapshot schema changed: {completion_path}")
            row = BenchmarkRow(**_mapping(snapshot.get("row"), "retention item row"))
            score_value = dict(_mapping(snapshot.get("score"), "retention item score"))
            score_value["gold_number_span"] = tuple(score_value["gold_number_span"])
            score_value["hypothesis_number_span"] = tuple(score_value["hypothesis_number_span"])
            hypothesis = snapshot.get("asr_hypothesis")
            prompt_id = snapshot.get("prompt_id")
            if not isinstance(hypothesis, str) or not isinstance(prompt_id, str) or not prompt_id:
                raise EvaluationIntegrityError(f"Completed validation item text identity changed: {completion_path}")
            item = ValidationItem(
                row=row,
                score=ItemScore(**score_value),
                prompt_id=prompt_id,
                asr_hypothesis=hypothesis,
            )
            if row.id in item_audio:
                raise EvaluationIntegrityError(f"Completed validation item IDs are not unique: {completion_path}")
            source = Path(str(snapshot.get("wav_path")))
            expected_hash = snapshot.get("wav_sha256")
            if not isinstance(expected_hash, str):
                raise EvaluationIntegrityError(f"Completed validation item WAV hash changed: {completion_path}")
            try:
                source.resolve().relative_to(wav_root)
            except ValueError as error:
                raise EvaluationIntegrityError(
                    f"Completed validation source audio escapes WAV tree: {source}"
                ) from error
            if source_wavs_exist and (
                source.is_symlink() or not source.is_file() or sha256_file(source) != expected_hash
            ):
                raise EvaluationIntegrityError(f"Completed validation source audio content changed: {source}")
            item_audio[row.id] = (source, expected_hash)
            items.append(item)
        if len(item_audio) != self.expected_item_count or len({value[0] for value in item_audio.values()}) != len(
            item_audio
        ):
            raise EvaluationIntegrityError(f"Completed validation WAV identities are not exact: {completion_path}")
        self._validate_retention_metrics(boundary_dir, record, items)

        audio_value = record.get("audio")
        if not isinstance(audio_value, list) or len(audio_value) != 4:
            raise EvaluationIntegrityError(
                f"Completed validation must retain exactly four audio entries: {completion_path}"
            )
        audio = tuple(audio_value)
        expected_ids = tuple(getattr(self.selection, "audio_log_ids"))
        if tuple(item.get("id") for item in audio if isinstance(item, Mapping)) != expected_ids:
            raise EvaluationIntegrityError(f"Completed validation fixed audio order changed: {completion_path}")
        if len({item_id for item_id in expected_ids}) != 4:
            raise EvaluationIntegrityError(f"Completed validation fixed audio IDs are not unique: {completion_path}")
        for item in audio:
            if not isinstance(item, Mapping) or set(item) != _AUDIO_KEYS:
                raise EvaluationIntegrityError(f"Completed validation audio schema changed: {completion_path}")
            item_id = item["id"]
            source, expected_hash = item_audio[item_id]
            retained = Path(str(item["retained_path"]))
            if Path(str(item["source_path"])) != source or item.get("sha256") != expected_hash:
                raise EvaluationIntegrityError(f"Completed validation audio identity changed: {completion_path}")
            if retained != self._retained_dir(boundary_dir) / f"{item_id:05d}.wav":
                raise EvaluationIntegrityError(f"Completed validation retained-audio path changed: {retained}")

        candidate = _RetentionCandidate(
            boundary_dir=boundary_dir,
            record=MappingProxyType(record),
            audio=audio,
            ordering_key=(
                int(normalized_boundary["global_step"]),
                str(record["published_at"]),
                boundary_dir.name,
            ),
            source_wavs_exist=source_wavs_exist,
        )
        if not source_wavs_exist:
            self._validate_retained_boundary(candidate, self._retained_dir(boundary_dir))
        return candidate

    def _validate_retention_metrics(
        self, boundary_dir: Path, completion: Mapping[str, object], items: Sequence[ValidationItem]
    ) -> None:
        metrics_record = _read_json(boundary_dir / "metrics.json", "retention metrics")
        if metrics_record.get("input_fingerprint") != completion["input_fingerprint"]:
            raise EvaluationIntegrityError(f"Completed validation metrics identity changed: {boundary_dir}")
        recomputed = aggregate_scores([item.score for item in items])
        if dict(_mapping(metrics_record.get("metrics"), "retention aggregate metrics")) != _aggregate_snapshot(
            recomputed
        ):
            raise EvaluationIntegrityError(f"Completed validation aggregate metrics changed: {boundary_dir}")
        categories = _mapping(metrics_record.get("category_metrics"), "retention category metrics")
        expected_categories = {
            category: _aggregate_snapshot(value) for category, value in recomputed.category_metrics.items()
        }
        if dict(categories) != expected_categories:
            raise EvaluationIntegrityError(f"Completed validation category metrics changed: {boundary_dir}")

    def _retained_dir(self, boundary_dir: Path) -> Path:
        return self.validation_root / "retained-audio" / boundary_dir.name

    def _stage_retained_boundary(self, candidate: _RetentionCandidate) -> Path:
        if not candidate.source_wavs_exist:
            raise EvaluationIntegrityError(
                f"Completed validation has neither source nor retained audio: {candidate.boundary_dir}"
            )
        retained_parent = self.validation_root / "retained-audio"
        retained_parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(mkdtemp(prefix=f".{candidate.boundary_dir.name}.", suffix=".stage", dir=retained_parent))
        try:
            for item in candidate.audio:
                item_id = int(item["id"])
                _atomic_copy(Path(str(item["source_path"])), staging_dir / f"{item_id:05d}.wav")
            atomic_json(staging_dir / "retention-complete.json", self._retention_manifest(candidate))
            self._validate_retained_boundary(candidate, staging_dir)
            return staging_dir
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _retention_manifest(self, candidate: _RetentionCandidate) -> dict[str, object]:
        identity = {
            "version": 1,
            "completion_snapshot_fingerprint": candidate.record["snapshot_fingerprint"],
            "audio": [
                {"id": item["id"], "name": f"{int(item['id']):05d}.wav", "sha256": item["sha256"]}
                for item in candidate.audio
            ],
        }
        return {**identity, "manifest_fingerprint": fingerprint(identity)}

    def _validate_retained_boundary(self, candidate: _RetentionCandidate, retained_dir: Path) -> None:
        if not retained_dir.is_dir() or retained_dir.is_symlink():
            raise EvaluationIntegrityError(f"Retained validation boundary is unavailable: {retained_dir}")
        expected_names = {*(f"{int(item['id']):05d}.wav" for item in candidate.audio), "retention-complete.json"}
        entries = tuple(retained_dir.iterdir())
        actual_names = {path.name for path in entries}
        if actual_names != expected_names or any(path.is_symlink() or not path.is_file() for path in entries):
            raise EvaluationIntegrityError(f"Retained validation boundary file set changed: {retained_dir}")
        manifest = _read_json(retained_dir / "retention-complete.json", "retention manifest")
        expected_manifest = self._retention_manifest(candidate)
        if manifest != expected_manifest:
            raise EvaluationIntegrityError(f"Retained validation manifest changed: {retained_dir}")
        for item in candidate.audio:
            path = retained_dir / f"{int(item['id']):05d}.wav"
            if sha256_file(path) != item["sha256"]:
                raise EvaluationIntegrityError(f"Retained validation audio content changed: {path}")

    def _raise_published_failure(self, context: _RunContext) -> None:
        path = self.ledger.root / "validation-failed.json"
        record = _read_json(path, "validation failure")
        if record.get("input_fingerprint") != context.input_fingerprint:
            raise EvaluationIntegrityError(f"Validation failure record belongs to different inputs: {path}")
        _raise_error_snapshot(_mapping(record.get("error"), "validation failure error"))


def _checkpoint_fingerprint(checkpoint: object) -> str:
    if isinstance(checkpoint, Mapping):
        value = checkpoint.get("checkpoint_fingerprint")
    elif isinstance(checkpoint, Path) or (
        isinstance(checkpoint, str) and (Path(checkpoint).is_dir() or Path(checkpoint).is_file())
    ):
        path = Path(checkpoint)
        metadata_path = path / "metadata.json" if path.is_dir() else path
        value = _read_json(metadata_path, "checkpoint metadata").get("checkpoint_fingerprint")
    else:
        value = getattr(checkpoint, "checkpoint_fingerprint", checkpoint if isinstance(checkpoint, str) else None)
    if not isinstance(value, str) or not value:
        raise ValueError("Evaluator checkpoint must provide a non-empty checkpoint_fingerprint")
    return value


def _boundary_value(boundary: object) -> dict[str, object]:
    if isinstance(boundary, Mapping):
        values = dict(boundary)
    else:
        values = {
            name: getattr(boundary, name, None)
            for name in ("stage", "epoch", "boundary", "global_step", "stage_progress")
        }
    stage = values.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("Evaluation boundary stage must be a non-empty string")
    for name in ("epoch", "global_step"):
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Evaluation boundary {name} must be a non-negative integer")
    index = values.get("boundary")
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= 8:
        raise ValueError("Evaluation boundary index must be between one and eight")
    progress = values.get("stage_progress")
    if isinstance(progress, bool) or not isinstance(progress, int | float) or not math.isfinite(progress):
        raise ValueError("Evaluation boundary stage_progress must be finite")
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError("Evaluation boundary stage_progress must be between zero and one")
    return {
        "stage": stage,
        "epoch": values["epoch"],
        "boundary": index,
        "global_step": values["global_step"],
        "stage_progress": float(progress),
    }


def _sample_rate(model: Any) -> int:
    value = getattr(model, "sample_rate", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("Unwrapped VoxCPM model must expose a positive integer sample_rate")
    return value


def _audio_array(generated: object) -> np.ndarray:
    value = generated[0] if isinstance(generated, tuple) and generated else generated
    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("VoxCPM generation returned empty or non-finite audio")
    return array


def _atomic_wav(path: Path, generated: object, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".wav", delete=False) as temp:
            temporary_path = Path(temp.name)
        sf.write(str(temporary_path), _audio_array(generated), sample_rate, format="WAV", subtype="PCM_16")
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for row in rows:
                json.dump(dict(row), temporary, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with (
            source.open("rb") as input_stream,
            NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as output_stream,
        ):
            temporary_path = Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _item_snapshot(item: ValidationItem, wav_path: Path, wav_sha256: str) -> dict[str, object]:
    return {
        "row": asdict(item.row),
        "score": asdict(item.score),
        "prompt_id": item.prompt_id,
        "asr_hypothesis": item.asr_hypothesis,
        "wav_path": str(wav_path),
        "wav_sha256": wav_sha256,
    }


def _aggregate_snapshot(metrics: AggregateScore) -> dict[str, object]:
    value = {
        field.name: getattr(metrics, field.name) for field in fields(AggregateScore) if field.name != "category_metrics"
    }
    value.update(
        {
            "num_cer": metrics.num_cer,
            "num_wer": metrics.num_wer,
            "utt_cer": metrics.utt_cer,
            "utt_wer": metrics.utt_wer,
        }
    )
    return value


def _aggregate_from_snapshot(value: Mapping[str, object]) -> AggregateScore:
    return AggregateScore(
        **{field.name: value[field.name] for field in fields(AggregateScore) if field.name != "category_metrics"}
    )


def _training_mode(module: Any) -> bool | None:
    value = getattr(module, "training", None)
    return value if isinstance(value, bool) else None


def _set_training(module: Any, training: bool) -> None:
    method = getattr(module, "train", None)
    if callable(method):
        method(training)


def _restore_training(module: Any, training: bool | None) -> None:
    if training is not None:
        _set_training(module, training)


def _restore_audio_vae(model: Any, had_attribute: bool, previous: Any) -> None:
    if had_attribute:
        setattr(model, "audio_vae", previous)
    elif hasattr(model, "audio_vae"):
        delattr(model, "audio_vae")


def _device_id(runtime: Any) -> int:
    index = getattr(getattr(runtime, "device", None), "index", None)
    return int(index if index is not None else runtime.rank)


def _capture_rng(runtime: Any) -> _RNGState:
    device = getattr(runtime, "device", None)
    cuda_device = torch.device(device) if isinstance(device, torch.device) and device.type == "cuda" else None
    cpu_state = torch.get_rng_state().clone()
    cuda_state = torch.cuda.get_rng_state(cuda_device).clone() if cuda_device is not None else None
    return _RNGState(cpu=cpu_state, cuda_device=cuda_device, cuda=cuda_state)


def _restore_rng(state: _RNGState) -> None:
    torch.set_rng_state(state.cpu)
    if state.cuda_device is not None and state.cuda is not None:
        torch.cuda.set_rng_state(state.cuda, state.cuda_device)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationIntegrityError(f"{label} must be a mapping")
    return value


def _attempt_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvaluationIntegrityError("Validation attempt count must be a non-negative integer")
    return value


def _nonnegative_time(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise EvaluationIntegrityError("Validation elapsed time must be finite and non-negative")
    return float(value)


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationIntegrityError(f"{label} is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise EvaluationIntegrityError(f"{label} is malformed: {path}")
    return value


def _read_item_snapshots(path: Path, expected_count: int) -> list[Mapping[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            values = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationIntegrityError(f"Completed validation items are unreadable: {path}") from error
    if len(values) != expected_count or any(not isinstance(value, Mapping) for value in values):
        raise EvaluationIntegrityError(f"Completed validation items have the wrong cardinality: {path}")
    return values


def _error_snapshot(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _raise_error_snapshot(error: Mapping[str, object]) -> None:
    message = str(error.get("message", "validation failed"))
    if error.get("type") == "IncompleteValidation":
        raise IncompleteValidation(message)
    if error.get("type") == "EvaluationIntegrityError":
        raise EvaluationIntegrityError(message)
    raise RuntimeError(message)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
