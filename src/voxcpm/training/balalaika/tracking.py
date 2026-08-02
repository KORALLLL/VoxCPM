"""Rank-zero W&B tracking and durable validation-boundary payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import importlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from .artifacts import atomic_json, fingerprint
from .metrics import AggregateScore, BenchmarkRow, ItemScore

_BOUNDARY_ITEM_COUNT = 2_000
_AUDIO_EXAMPLE_COUNT = 4
_TABLE_COLUMNS = (
    "id",
    "category",
    "input",
    "normalized_gold",
    "stressed",
    "prompt_id",
    "asr_hypothesis",
    "gold_number_span",
    "hypothesis_number_span",
    "gold_number_text",
    "hypothesis_number_text",
    "num_char_substitutions",
    "num_char_deletions",
    "num_char_insertions",
    "num_char_errors",
    "num_char_reference_units",
    "num_cer",
    "num_word_substitutions",
    "num_word_deletions",
    "num_word_insertions",
    "num_word_errors",
    "num_word_reference_units",
    "num_wer",
    "utt_char_substitutions",
    "utt_char_deletions",
    "utt_char_insertions",
    "utt_char_errors",
    "utt_char_reference_units",
    "utt_cer",
    "utt_word_substitutions",
    "utt_word_deletions",
    "utt_word_insertions",
    "utt_word_errors",
    "utt_word_reference_units",
    "utt_wer",
)


@dataclass(frozen=True)
class ValidationItem:
    """One scored benchmark row enriched with its fixed prompt assignment."""

    row: BenchmarkRow
    score: ItemScore
    prompt_id: str
    asr_hypothesis: str | None = None

    def __post_init__(self) -> None:
        if self.row.id != self.score.row_id or self.row.category != self.score.category:
            raise ValueError("Validation item row and score do not describe the same benchmark item.")
        if not isinstance(self.prompt_id, str) or not self.prompt_id:
            raise ValueError("Validation item prompt_id must be a non-empty string.")
        if self.asr_hypothesis is not None and not isinstance(self.asr_hypothesis, str):
            raise TypeError("Validation item asr_hypothesis must be a string or None.")


@dataclass(frozen=True)
class ValidationPayload:
    """A complete, immutable evaluator handoff for one validation boundary."""

    metrics: AggregateScore
    category_metrics: Mapping[str, AggregateScore]
    items: Sequence[ValidationItem]
    audio_paths: Mapping[int, str | Path]
    artifact_dir: str | Path
    timings: Mapping[str, float] = field(default_factory=dict)
    failure_counts: Mapping[str, int] = field(default_factory=dict)
    stage_progress: float = 0.0
    input_fingerprint: str = ""

    def __post_init__(self) -> None:
        materialized_items = tuple(self.items)
        if len(materialized_items) != _BOUNDARY_ITEM_COUNT:
            raise ValueError(f"Validation payload must contain exactly {_BOUNDARY_ITEM_COUNT:,} item rows.")
        if self.metrics.item_count != _BOUNDARY_ITEM_COUNT:
            raise ValueError(f"Validation metrics must report exactly {_BOUNDARY_ITEM_COUNT:,} item rows.")
        if len({item.row.id for item in materialized_items}) != _BOUNDARY_ITEM_COUNT:
            raise ValueError("Validation payload item IDs must be unique.")
        if any(not isinstance(item, ValidationItem) for item in materialized_items):
            raise TypeError("Validation payload items must be ValidationItem instances.")

        normalized_audio = {int(item_id): Path(path) for item_id, path in self.audio_paths.items()}
        if len(normalized_audio) != _AUDIO_EXAMPLE_COUNT:
            raise ValueError(
                f"Validation payload must contain exactly four audio examples, not {len(normalized_audio)}."
            )
        item_ids = {item.row.id for item in materialized_items}
        if set(normalized_audio) - item_ids:
            raise ValueError("Validation audio IDs must identify payload items.")
        if any(not path.is_file() for path in normalized_audio.values()):
            raise FileNotFoundError("Validation audio examples must be local WAV files before logging.")

        normalized_categories = {str(key): value for key, value in self.category_metrics.items()}
        if set(normalized_categories) != {item.row.category for item in materialized_items}:
            raise ValueError("Validation category metrics must cover exactly the scored item categories.")
        if any(value.item_count <= 0 for value in normalized_categories.values()):
            raise ValueError("Validation category metrics must have positive item counts.")
        normalized_timings = _normalize_nonnegative_float_mapping(self.timings, "timings")
        normalized_failures = _normalize_nonnegative_int_mapping(self.failure_counts, "failure_counts")
        if not isinstance(self.stage_progress, (int, float)) or not math.isfinite(self.stage_progress):
            raise ValueError("Validation stage_progress must be finite.")
        if not 0.0 <= float(self.stage_progress) <= 1.0:
            raise ValueError("Validation stage_progress must be between zero and one.")
        if not isinstance(self.input_fingerprint, str):
            raise TypeError("Validation input_fingerprint must be a string.")

        object.__setattr__(self, "items", materialized_items)
        object.__setattr__(self, "audio_paths", MappingProxyType(normalized_audio))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        object.__setattr__(self, "category_metrics", MappingProxyType(normalized_categories))
        object.__setattr__(self, "timings", MappingProxyType(normalized_timings))
        object.__setattr__(self, "failure_counts", MappingProxyType(normalized_failures))
        object.__setattr__(self, "stage_progress", float(self.stage_progress))


class NullRunManager:
    """Strict no-op tracking surface used by every non-main distributed rank."""

    def log_train(self, metrics: Mapping[str, object], global_step: int) -> None:
        return None

    def log_validation(
        self,
        result: ValidationPayload,
        audio_paths: Mapping[int, str | Path] | Sequence[str | Path] | None = None,
        global_step: int = 0,
    ) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbRunManager:
    """One rank-zero resumable W&B run with durable local boundary manifests."""

    def __init__(
        self,
        *,
        job_type: str,
        run_id: str,
        settings: "_RunSettings",
        wandb_module: Any,
        run: Any,
    ) -> None:
        self.job_type = job_type
        self.run_id = run_id
        self._settings = settings
        self._wandb = wandb_module
        self._run = run
        self._finished = False

    @classmethod
    def start(
        cls,
        job_type: str,
        run_state_path: str | Path,
        config: object,
        *,
        wandb_module: Any | None = None,
    ) -> "WandbRunManager":
        """Persist an identity before W&B initialization, then resume that job-type run."""
        if job_type not in {"memorization", "stage1", "stage2"}:
            raise ValueError("W&B job_type must be memorization, stage1, or stage2.")
        settings = _RunSettings.from_config(config)
        run_id = _load_or_create_run_id(Path(run_state_path), job_type, settings.config_fingerprint)
        module = _import_wandb() if wandb_module is None else wandb_module
        run = module.init(
            project=settings.project,
            entity=settings.entity,
            id=run_id,
            resume="allow",
            group=settings.group,
            job_type=job_type,
            mode=settings.mode,
            dir=str(settings.directory),
            config=settings.wandb_config,
        )
        if run is None:
            run = getattr(module, "run", None)
        if run is None:
            raise RuntimeError("wandb.init did not return a run.")
        return cls(job_type=job_type, run_id=run_id, settings=settings, wandb_module=module, run=run)

    def log_train(self, metrics: Mapping[str, object], global_step: int) -> None:
        """Log rank-zero training scalars without mutating the caller's mapping."""
        payload = {str(key): value for key, value in metrics.items()}
        payload["train/global_step"] = int(global_step)
        self._run.log(payload, step=int(global_step))

    def log_validation(
        self,
        result: ValidationPayload,
        audio_paths: Mapping[int, str | Path] | Sequence[str | Path] | None = None,
        global_step: int = 0,
    ) -> None:
        """Log a new immutable W&B table and complete its local manifest only after upload."""
        if not isinstance(result, ValidationPayload):
            raise TypeError("log_validation requires a ValidationPayload.")
        selected_audio = _selected_audio_paths(result, audio_paths)
        boundary_path = self._boundary_path(result.artifact_dir, global_step)
        if _is_completed_boundary(
            boundary_path,
            run_id=self.run_id,
            config_fingerprint=self._settings.config_fingerprint,
            input_fingerprint=result.input_fingerprint or _payload_input_fingerprint(result),
        ):
            return
        pending = self._boundary_record(result, selected_audio, global_step, status="pending")
        atomic_json(boundary_path, pending)

        payload = self._validation_log_payload(result, selected_audio, global_step)
        self._run.log(payload, step=int(global_step))
        _fsync_directory(_run_directory(self._run))
        atomic_json(boundary_path, self._boundary_record(result, selected_audio, global_step, status="complete"))

    def finish(self) -> None:
        """Finish this rank-zero run once, after all successfully logged boundaries."""
        if self._finished:
            return
        self._run.finish()
        self._finished = True

    def _validation_log_payload(
        self,
        result: ValidationPayload,
        audio_paths: Mapping[int, Path],
        global_step: int,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "val/num_cer": result.metrics.num_cer,
            "val/num_wer": result.metrics.num_wer,
            "val/utt_cer": result.metrics.utt_cer,
            "val/utt_wer": result.metrics.utt_wer,
            "val/item_count": result.metrics.item_count,
            "val/items": self._wandb.Table(columns=list(_TABLE_COLUMNS), data=_table_data(result.items)),
            "val/examples": _audio_examples(self._wandb, result, audio_paths),
            "stage/progress": result.stage_progress,
            "global_step": int(global_step),
            "train/global_step": int(global_step),
            "config/fingerprint": self._settings.config_fingerprint,
            "input/fingerprint": result.input_fingerprint or _payload_input_fingerprint(result),
        }
        for category, metrics in result.category_metrics.items():
            prefix = f"val/category/{category}"
            payload.update(_aggregate_metric_payload(prefix, metrics))
        for name, value in result.timings.items():
            payload[f"val/timing/{name}"] = value
        for name, value in result.failure_counts.items():
            payload[f"val/failures/{name}"] = value
        return payload

    def _boundary_path(self, artifact_dir: Path, global_step: int) -> Path:
        return artifact_dir / "wandb-boundaries" / f"{self.job_type}-step-{int(global_step):09d}.json"

    def _boundary_record(
        self,
        result: ValidationPayload,
        audio_paths: Mapping[int, Path],
        global_step: int,
        *,
        status: str,
    ) -> dict[str, object]:
        return {
            "version": 1,
            "status": status,
            "job_type": self.job_type,
            "run_id": self.run_id,
            "mode": self._settings.mode,
            "global_step": int(global_step),
            "stage_progress": result.stage_progress,
            "config_fingerprint": self._settings.config_fingerprint,
            "input_fingerprint": result.input_fingerprint or _payload_input_fingerprint(result),
            "metrics": _aggregate_metric_payload("", result.metrics),
            "category_metrics": {
                category: _aggregate_metric_payload("", metrics)
                for category, metrics in result.category_metrics.items()
            },
            "timings": dict(result.timings),
            "failure_counts": dict(result.failure_counts),
            "item_ids": [item.row.id for item in result.items],
            "audio": [{"id": item_id, "path": str(path)} for item_id, path in audio_paths.items()],
        }


def create_run_manager(
    *,
    is_main_process: bool,
    config: object,
    job_type: str | None = None,
    run_state_path: str | Path | None = None,
    wandb_module: Any | None = None,
) -> WandbRunManager | NullRunManager:
    """Return a no-op on workers, importing W&B only for the main process."""
    if not is_main_process:
        return NullRunManager()
    if job_type is None or run_state_path is None:
        raise ValueError("Main-process W&B tracking requires job_type and run_state_path.")
    return WandbRunManager.start(job_type, run_state_path, config, wandb_module=wandb_module)


@dataclass(frozen=True)
class _RunSettings:
    project: str
    entity: str | None
    mode: str
    group: str
    directory: Path
    wandb_config: Mapping[str, object]
    config_fingerprint: str

    @classmethod
    def from_config(cls, config: object) -> "_RunSettings":
        values = _config_mapping(config)
        wandb_values = values.get("wandb")
        if isinstance(wandb_values, Mapping):
            values = {**_config_mapping(wandb_values), **values}
        project = _required_string(values, "project", default="voxcpm-balalaika")
        entity = values.get("entity")
        if entity is not None and not isinstance(entity, str):
            raise TypeError("W&B entity must be a string or None.")
        mode = _required_string(values, "mode", default="disabled")
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("W&B mode must be online, offline, or disabled.")
        config_fingerprint = values.get("config_fingerprint")
        if config_fingerprint is None:
            config_fingerprint = fingerprint(_fingerprintable_config(config))
        if not isinstance(config_fingerprint, str) or not config_fingerprint:
            raise ValueError("W&B config_fingerprint must be a non-empty string.")
        group = _required_string(values, "group", default=f"balalaika-{config_fingerprint[:12]}")
        directory_value = values.get("dir")
        if directory_value is None:
            output_dir = values.get("output_dir")
            directory_value = Path(output_dir) / "wandb" if output_dir is not None else Path("wandb")
        directory = Path(directory_value)
        explicit_wandb_config = values.get("config")
        if explicit_wandb_config is not None and not isinstance(explicit_wandb_config, Mapping):
            raise TypeError("W&B config must be a mapping.")
        serialized_config = dict(explicit_wandb_config or _fingerprintable_config(config))
        serialized_config.setdefault("config_fingerprint", config_fingerprint)
        return cls(
            project=project,
            entity=entity,
            mode=mode,
            group=group,
            directory=directory,
            wandb_config=MappingProxyType(serialized_config),
            config_fingerprint=config_fingerprint,
        )


def _load_or_create_run_id(path: Path, job_type: str, config_fingerprint: str) -> str:
    if path.is_file():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"W&B run state is unreadable: {path}") from error
        if not isinstance(state, dict) or state.get("job_type") != job_type:
            raise RuntimeError(f"W&B run state does not belong to {job_type}: {path}")
        run_id = state.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(f"W&B run state has no valid run_id: {path}")
        return run_id
    run_id = uuid4().hex
    atomic_json(
        path,
        {"version": 1, "job_type": job_type, "run_id": run_id, "config_fingerprint": config_fingerprint},
    )
    return run_id


def _import_wandb() -> Any:
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise RuntimeError("Install the balalaika extra to enable rank-zero W&B tracking.") from error


def _config_mapping(config: object) -> dict[str, object]:
    if isinstance(config, Mapping):
        return {str(key): value for key, value in config.items()}
    if hasattr(config, "model_dump"):
        dumped = config.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return {str(key): value for key, value in dumped.items()}
    return {key: value for key, value in vars(config).items() if not key.startswith("_") and not callable(value)}


def _fingerprintable_config(config: object) -> dict[str, object]:
    value = _config_mapping(config)
    return {key: _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_value(value.model_dump(mode="json"))
    return value


def _required_string(values: Mapping[str, object], name: str, *, default: str) -> str:
    value = values.get(name, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"W&B {name} must be a non-empty string.")
    return value


def _selected_audio_paths(
    result: ValidationPayload,
    supplied: Mapping[int, str | Path] | Sequence[str | Path] | None,
) -> Mapping[int, Path]:
    if supplied is None:
        selected = dict(result.audio_paths)
    elif isinstance(supplied, Mapping):
        selected = {int(item_id): Path(path) for item_id, path in supplied.items()}
    else:
        if len(supplied) != _AUDIO_EXAMPLE_COUNT:
            raise ValueError("Validation logging requires exactly four audio examples.")
        selected = {item_id: Path(path) for item_id, path in zip(result.audio_paths, supplied, strict=True)}
    if list(selected) != list(result.audio_paths):
        raise ValueError("Validation logging audio IDs must retain the payload's fixed order.")
    if any(path != result.audio_paths[item_id] for item_id, path in selected.items()):
        raise ValueError("Validation logging audio paths must match the durable payload.")
    return MappingProxyType(selected)


def _table_data(items: Sequence[ValidationItem]) -> list[list[object]]:
    return [[_table_value(item, column) for column in _TABLE_COLUMNS] for item in items]


def _table_value(item: ValidationItem, column: str) -> object:
    if column == "id":
        return item.row.id
    if column == "input":
        return item.row.text
    if column == "stressed":
        return item.row.stressed
    if column == "prompt_id":
        return item.prompt_id
    if column == "asr_hypothesis":
        return item.asr_hypothesis if item.asr_hypothesis is not None else item.score.normalized_hypothesis
    if column in {"gold_number_span", "hypothesis_number_span"}:
        return list(getattr(item.score, column))
    if column in {"category", "normalized_gold"}:
        return getattr(item.row, column)
    return getattr(item.score, column)


def _audio_examples(wandb_module: Any, result: ValidationPayload, audio_paths: Mapping[int, Path]) -> list[object]:
    by_id = {item.row.id: item for item in result.items}
    return [
        wandb_module.Audio(
            str(path),
            caption=(
                f"id={item_id} | category={by_id[item_id].row.category} | "
                f"prompt_id={by_id[item_id].prompt_id} | stressed={by_id[item_id].row.stressed}"
            ),
        )
        for item_id, path in audio_paths.items()
    ]


def _aggregate_metric_payload(prefix: str, metrics: AggregateScore) -> dict[str, object]:
    separator = "/" if prefix else ""
    payload: dict[str, object] = {
        f"{prefix}{separator}count": metrics.item_count,
        f"{prefix}{separator}num_cer": metrics.num_cer,
        f"{prefix}{separator}num_wer": metrics.num_wer,
        f"{prefix}{separator}utt_cer": metrics.utt_cer,
        f"{prefix}{separator}utt_wer": metrics.utt_wer,
    }
    for scope in ("num_char", "num_word", "utt_char", "utt_word"):
        for suffix in ("substitutions", "deletions", "insertions", "errors", "reference_units"):
            payload[f"{prefix}{separator}{scope}_{suffix}"] = getattr(metrics, f"{scope}_{suffix}")
    return payload


def _payload_input_fingerprint(result: ValidationPayload) -> str:
    return fingerprint(
        {
            "item_ids": [item.row.id for item in result.items],
            "prompt_ids": [item.prompt_id for item in result.items],
            "audio_ids": list(result.audio_paths),
        }
    )


def _run_directory(run: Any) -> Path:
    directory = getattr(run, "dir", None)
    if not isinstance(directory, str | Path):
        raise RuntimeError("W&B run has no durable local directory.")
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise RuntimeError(f"W&B run directory is unavailable: {directory_path}")
    return directory_path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_completed_boundary(
    path: Path,
    *,
    run_id: str,
    config_fingerprint: str,
    input_fingerprint: str,
) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Validation boundary manifest is unreadable: {path}") from error
    if not isinstance(record, dict):
        raise RuntimeError(f"Validation boundary manifest is malformed: {path}")
    if record.get("status") != "complete":
        return False
    expected = {
        "run_id": run_id,
        "config_fingerprint": config_fingerprint,
        "input_fingerprint": input_fingerprint,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Validation boundary is already complete for different inputs: {path}")
    return True


def _normalize_nonnegative_float_mapping(value: Mapping[str, float], label: str) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item < 0
        ):
            raise ValueError(f"Validation {label} must contain non-negative finite numeric values.")
        normalized[key] = float(item)
    return normalized


def _normalize_nonnegative_int_mapping(value: Mapping[str, int], label: str) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"Validation {label} must contain non-negative integer values.")
        normalized[key] = item
    return normalized
