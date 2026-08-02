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
from tempfile import NamedTemporaryFile
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
            record = self._verify_completion(context, require_retained=False)
            if self.runtime.rank == 0:
                self._apply_retention()
            self.runtime.barrier()
            self._verify_completion(context, require_retained=True)
            return self._payload_from_artifacts(context, record)

        local_status_path = self.ledger.root / "rank-status" / f"rank-{self.runtime.rank:02d}.json"
        try:
            completed = self.run_rank(model, audio_vae, checkpoint, boundary)
            atomic_json(
                local_status_path,
                {
                    "version": 1,
                    "status": "success",
                    "rank": self.runtime.rank,
                    "input_fingerprint": context.input_fingerprint,
                    "completed_ids": sorted(completed),
                },
            )
        except Exception as error:
            atomic_json(
                local_status_path,
                {
                    "version": 1,
                    "status": "failed",
                    "rank": self.runtime.rank,
                    "input_fingerprint": context.input_fingerprint,
                    "error": _error_snapshot(error),
                },
            )
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
                    if claim is None:
                        claim = self.ledger.claim(item_id, rank=self.runtime.rank, inputs=inputs, attempt_seed=seed)
                    if claim is None:
                        break
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
        completed: list[tuple[int, str, Path, dict[str, object]]] = []
        if not self.validation_root.is_dir():
            return
        for boundary_dir in sorted(self.validation_root.iterdir()):
            if not boundary_dir.is_dir() or boundary_dir.name == "retained-audio":
                continue
            completion_path = boundary_dir / "validation-complete.json"
            if not completion_path.is_file():
                continue
            record = _read_json(completion_path, "retention completion")
            if record.get("status") != "complete" or not isinstance(record.get("audio"), list):
                raise EvaluationIntegrityError(f"Cannot retain malformed completed boundary: {completion_path}")
            boundary = _mapping(record.get("boundary"), "retention boundary")
            global_step = boundary.get("global_step")
            if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
                raise EvaluationIntegrityError(f"Completed boundary has invalid global step: {completion_path}")
            self._retain_boundary_audio(boundary_dir, record)
            completed.append((global_step, str(record.get("published_at", "")), boundary_dir, record))
        if not completed:
            return
        latest = max(completed, key=lambda value: (value[0], value[1], value[2].name))[2]
        for _, _, boundary_dir, _ in completed:
            wavs = boundary_dir / "wavs"
            if boundary_dir != latest and wavs.exists():
                if wavs.is_symlink() or not wavs.is_dir():
                    raise EvaluationIntegrityError(f"Validation WAV retention target is unsafe: {wavs}")
                shutil.rmtree(wavs)

    def _retain_boundary_audio(self, boundary_dir: Path, completion: Mapping[str, object]) -> None:
        for item in completion["audio"]:
            if not isinstance(item, Mapping):
                raise EvaluationIntegrityError(f"Completed boundary audio record is malformed: {boundary_dir}")
            item_id = item.get("id")
            if isinstance(item_id, bool) or not isinstance(item_id, int):
                raise EvaluationIntegrityError(f"Completed boundary audio ID is malformed: {boundary_dir}")
            source = Path(str(item.get("source_path")))
            retained = Path(str(item.get("retained_path")))
            expected_retained = self.validation_root / "retained-audio" / boundary_dir.name / f"{item_id:05d}.wav"
            try:
                source.relative_to(boundary_dir / "wavs")
            except ValueError as error:
                raise EvaluationIntegrityError(
                    f"Completed boundary audio escapes its WAV directory: {source}"
                ) from error
            if retained != expected_retained:
                raise EvaluationIntegrityError(f"Completed boundary retained-audio path changed: {retained}")
            expected_hash = item.get("sha256")
            if retained.is_file() and sha256_file(retained) == expected_hash:
                continue
            if not source.is_file() or not isinstance(expected_hash, str) or sha256_file(source) != expected_hash:
                raise EvaluationIntegrityError(f"Completed boundary source audio content changed: {source}")
            _atomic_copy(source, retained)
            if sha256_file(retained) != expected_hash:
                raise EvaluationIntegrityError(f"Retained boundary audio copy changed: {retained}")

    def _raise_published_failure(self, context: _RunContext) -> None:
        path = self.ledger.root / "validation-failed.json"
        record = _read_json(path, "validation failure")
        if record.get("input_fingerprint") != context.input_fingerprint:
            raise EvaluationIntegrityError(f"Validation failure record belongs to different inputs: {path}")
        error = _mapping(record.get("error"), "validation failure error")
        message = str(error.get("message", "validation failed"))
        if error.get("type") == "IncompleteValidation":
            raise IncompleteValidation(message)
        if error.get("type") == "EvaluationIntegrityError":
            raise EvaluationIntegrityError(message)
        raise RuntimeError(message)


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


def _error_snapshot(error: Exception) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
