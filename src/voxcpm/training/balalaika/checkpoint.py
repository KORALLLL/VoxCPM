"""Atomic LoRA-only checkpoints backed by Accelerate's public state hooks."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .artifacts import atomic_json, fingerprint, sha256_file
from .schedule import TrainingProgress


class CheckpointError(RuntimeError):
    """A checkpoint is incomplete, corrupt, or unsafe to load."""


class CheckpointMismatch(CheckpointError):
    """A checkpoint does not match the immutable requested training identity."""


class CheckpointRestoreError(CheckpointError):
    """A restore mutated process state and requires a process restart."""


class CheckpointManager:
    """Publish and restore immutable boundary checkpoints beneath one root."""

    _SCHEMA_VERSION = 3
    _ADAPTER_FILE = "adapter_model.safetensors"
    _METADATA_FILE = "metadata.json"
    _MANIFEST_FILE = "manifest.json"
    _REQUIRED_SUPPLIED_METADATA = frozenset(
        {
            "stage_epochs",
            "world_size",
            "microbatch",
            "accumulation",
            "global_batch_size",
            "optimizer_steps_per_epoch",
            "optimizer_count",
            "scheduler_count",
            "checkpointable_count",
            "stage_start_global_step",
            "base_revision",
            "evaluator_revision",
            "data_fingerprint",
            "selection_fingerprint",
            "lora_fingerprint",
            "optimization_fingerprint",
            "wandb_run_id",
            "wandb_group",
        }
    )
    _PROGRESS_FIELDS = frozenset(
        {"stage", "epoch", "boundary", "microstep", "optimizer_step", "global_step", "sampler_seed", "sampler_epoch"}
    )
    _SAME_STAGE_EXPECTED_FIELDS = _REQUIRED_SUPPLIED_METADATA | frozenset({"stage", "sampler_seed"})
    _STAGE2_EXPECTED_FIELDS = frozenset(
        {
            "source_stage",
            "source_stage_epochs",
            "source_epoch",
            "source_boundary",
            "base_revision",
            "data_fingerprint",
            "selection_fingerprint",
            "lora_fingerprint",
        }
    )
    _SUPPORTED_DISTRIBUTED_TYPES = frozenset({"NO", "MULTI_CPU", "MULTI_GPU"})
    _CHECKPOINT_KINDS = frozenset({"boundary", "recovery"})

    def __init__(self, root: Path):
        self.root = Path(root)
        self._registrations: set[tuple[int, int]] = set()
        self._poisoned_reason: str | None = None

    def save_same_stage(
        self,
        accelerator: Any,
        model: Any,
        progress: TrainingProgress,
        metadata: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> Path:
        """Save adapter plus same-stage Accelerate state, then publish atomically."""
        return self._save_checkpoint(
            accelerator,
            model,
            progress,
            metadata,
            checkpoint_kind="boundary",
            name=name,
        )

    def save_recovery(
        self,
        accelerator: Any,
        model: Any,
        progress: TrainingProgress,
        metadata: Mapping[str, Any],
        *,
        name: str | None = None,
    ) -> Path:
        """Save a synchronized same-stage recovery after a complete accumulation group."""
        return self._save_checkpoint(
            accelerator,
            model,
            progress,
            metadata,
            checkpoint_kind="recovery",
            name=name,
        )

    def _save_checkpoint(
        self,
        accelerator: Any,
        model: Any,
        progress: TrainingProgress,
        metadata: Mapping[str, Any],
        *,
        checkpoint_kind: str,
        name: str | None,
    ) -> Path:
        self._ensure_usable()
        self._ensure_supported_accelerator(accelerator)
        complete_metadata = self._build_metadata(progress, metadata, checkpoint_kind=checkpoint_kind)
        self._validate_accelerator_world_size(accelerator, complete_metadata["world_size"])
        default_name = (
            self._checkpoint_name(progress) if checkpoint_kind == "boundary" else self._recovery_name(progress)
        )
        checkpoint_name = name or default_name
        self._validate_checkpoint_name(checkpoint_name)
        destination = self.root / checkpoint_name
        temporary = self.root / f".{checkpoint_name}.tmp"
        is_main_process = bool(accelerator.is_main_process)

        self.root.mkdir(parents=True, exist_ok=True)
        if is_main_process:
            if destination.exists():
                raise FileExistsError(f"immutable checkpoint already exists: {destination}")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=False)
        accelerator.wait_for_everyone()

        self._register_progress(accelerator, progress)
        hook = accelerator.register_save_state_pre_hook(self._save_adapter_hook(accelerator, model))
        try:
            accelerator.save_state(str(temporary))
        finally:
            hook.remove()
        accelerator.wait_for_everyone()

        if is_main_process:
            try:
                complete_metadata = self._finalize_metadata(temporary, complete_metadata)
                atomic_json(temporary / self._METADATA_FILE, complete_metadata)
                self._write_manifest(temporary)
                self._fsync_tree(temporary)
                os.rename(temporary, destination)
                self._fsync_directory(self.root)
                atomic_json(
                    self.root / "latest.json",
                    {"checkpoint": checkpoint_name, "fingerprint": complete_metadata["checkpoint_fingerprint"]},
                )
            except BaseException:
                # A hidden sibling temporary directory is intentionally not a checkpoint.
                raise
        accelerator.wait_for_everyone()
        if not destination.is_dir():
            raise CheckpointError(f"checkpoint publication did not complete: {destination}")
        return destination

    def resume_same_stage(
        self,
        accelerator: Any,
        model: Any,
        progress: TrainingProgress,
        checkpoint: Path,
        *,
        expected: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strictly verify and restore adapter, optimizer, scheduler, RNG, and progress."""
        self._ensure_usable()
        self._ensure_supported_accelerator(accelerator)
        self._require_expected_identity(expected, self._SAME_STAGE_EXPECTED_FIELDS, "expected identity")
        checkpoint = Path(checkpoint)
        metadata = self.verify(checkpoint, expected)
        self._validate_accelerator_world_size(accelerator, metadata["world_size"])
        if metadata["stage"] != progress.stage:
            raise CheckpointMismatch(
                f"stage mismatch: checkpoint has {metadata['stage']!r}, progress expects {progress.stage!r}"
            )

        target = accelerator.unwrap_model(model)
        self._validate_adapter_keys(checkpoint / self._ADAPTER_FILE, target.state_dict())
        self._preflight_progress(checkpoint, metadata)
        self._register_progress(accelerator, progress)
        hook = accelerator.register_load_state_pre_hook(self._load_adapter_hook(accelerator, model, checkpoint))
        try:
            try:
                accelerator.load_state(str(checkpoint))
            except Exception as error:
                self._poisoned_reason = f"Accelerate restore failed after model mutation: {error}"
                raise CheckpointRestoreError(
                    f"checkpoint restore failed after mutation; process restart required: {error}"
                ) from error
        finally:
            hook.remove()
        try:
            self._assert_progress_matches_metadata(progress, metadata)
        except CheckpointRestoreError as error:
            self._poisoned_reason = str(error)
            raise
        return metadata

    def load_stage_adapter(
        self,
        model: Any,
        checkpoint: Path,
        *,
        expected: Mapping[str, Any],
        progress: TrainingProgress | None = None,
        sampler_seed: int | None = None,
    ) -> dict[str, Any]:
        """Load a final stage-1 adapter without touching any Accelerate state."""
        self._ensure_usable()
        self._require_expected_identity(expected, self._STAGE2_EXPECTED_FIELDS, "expected source identity")
        if (
            not self._is_stage_one(expected["source_stage"])
            or expected["source_epoch"] != expected["source_stage_epochs"] - 1
            or expected["source_boundary"] != 8
        ):
            raise CheckpointMismatch("expected source identity does not describe a final stage1 checkpoint")
        checkpoint = Path(checkpoint)
        compatible_expected = {
            "stage": expected["source_stage"],
            "stage_epochs": expected["source_stage_epochs"],
            **{
                key: expected[key]
                for key in ("base_revision", "data_fingerprint", "selection_fingerprint", "lora_fingerprint")
            },
        }
        metadata = self.verify(checkpoint, compatible_expected)
        if metadata["checkpoint_kind"] != "boundary":
            raise CheckpointMismatch("stage transition requires a final stage1 boundary checkpoint, not recovery")
        if (
            not self._is_stage_one(metadata["stage"])
            or metadata["boundary"] != 8
            or metadata["epoch"] != metadata["stage_epochs"] - 1
            or metadata["epoch"] != expected["source_epoch"]
            or metadata["boundary"] != expected["source_boundary"]
        ):
            raise CheckpointMismatch("stage transition requires the final stage1 boundary checkpoint")
        self._validate_adapter_keys(checkpoint / self._ADAPTER_FILE, model.state_dict())
        self._load_adapter(model, checkpoint / self._ADAPTER_FILE)
        if progress is not None:
            progress.global_step = metadata["global_step"]
            progress.reset_for_stage("stage2", sampler_seed=sampler_seed)
        return metadata

    def verify(self, checkpoint: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
        """Verify structure, hashes, metadata fingerprint, adapter keys, and expected identity."""
        checkpoint = Path(checkpoint)
        if not checkpoint.is_dir() or checkpoint.name.startswith("."):
            raise CheckpointError(f"checkpoint is missing or unpublished: {checkpoint}")
        manifest_path = checkpoint / self._MANIFEST_FILE
        metadata_path = checkpoint / self._METADATA_FILE
        adapter_path = checkpoint / self._ADAPTER_FILE
        for path in (manifest_path, metadata_path, adapter_path):
            if not path.is_file():
                raise CheckpointError(f"checkpoint piece is missing: {path.name}")

        manifest = self._read_json(manifest_path, "manifest")
        if manifest.get("schema_version") != self._SCHEMA_VERSION or not isinstance(manifest.get("files"), dict):
            raise CheckpointError("checkpoint manifest has an unsupported schema or files table")
        recorded_files = manifest["files"]
        actual_files = {
            str(path.relative_to(checkpoint))
            for path in checkpoint.rglob("*")
            if path.is_file() and path.name != self._MANIFEST_FILE
        }
        if set(recorded_files) != actual_files:
            missing = sorted(set(recorded_files) - actual_files)
            extra = sorted(actual_files - set(recorded_files))
            raise CheckpointError(f"checkpoint file set mismatch; missing={missing}, extra={extra}")
        for relative_path, recorded_hash in sorted(recorded_files.items()):
            actual_hash = sha256_file(checkpoint / relative_path)
            if actual_hash != recorded_hash:
                raise CheckpointError(f"checkpoint piece {relative_path} checksum mismatch")
        metadata = self._read_json(metadata_path, "metadata")
        self._validate_metadata(metadata)
        fingerprint_payload = {key: value for key, value in metadata.items() if key != "checkpoint_fingerprint"}
        if metadata["checkpoint_fingerprint"] != fingerprint(fingerprint_payload):
            raise CheckpointError("checkpoint metadata fingerprint mismatch")
        self._require_exact_state_files(checkpoint, metadata)
        self._verify_expected(metadata, expected)
        self._validate_adapter_keys(adapter_path)
        return metadata

    def latest(self) -> Path | None:
        pointer_path = self.root / "latest.json"
        if not pointer_path.is_file():
            return None
        pointer = self._read_json(pointer_path, "latest pointer")
        checkpoint_name = pointer.get("checkpoint")
        if not isinstance(checkpoint_name, str):
            raise CheckpointError("latest pointer has no checkpoint name")
        self._validate_checkpoint_name(checkpoint_name)
        checkpoint = self.root / checkpoint_name
        metadata = self.verify(checkpoint, {})
        if pointer.get("fingerprint") != metadata["checkpoint_fingerprint"]:
            raise CheckpointError("latest pointer fingerprint mismatch")
        return checkpoint

    def _build_metadata(
        self,
        progress: TrainingProgress,
        supplied: Mapping[str, Any],
        *,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        supplied = dict(supplied)
        supplied.pop("rng_state", None)
        conflicting = (self._PROGRESS_FIELDS | {"checkpoint_kind", "accumulation_microstep"}).intersection(supplied)
        if conflicting:
            raise CheckpointError(f"progress metadata is manager-owned: {sorted(conflicting)}")
        missing = sorted(self._REQUIRED_SUPPLIED_METADATA - set(supplied))
        if missing:
            raise CheckpointError(f"checkpoint metadata is missing required fields: {missing}")
        if checkpoint_kind not in self._CHECKPOINT_KINDS:
            raise CheckpointError(f"unsupported checkpoint kind: {checkpoint_kind!r}")
        value: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            "checkpoint_kind": checkpoint_kind,
            "accumulation_microstep": progress.microstep % supplied["accumulation"],
            **supplied,
            **{key: item for key, item in progress.state_dict().items() if key != "schema_version"},
        }
        self._validate_metadata(value, require_rng=False, require_fingerprint=False)
        return value

    def _finalize_metadata(self, checkpoint: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(metadata)
        self._require_exact_state_files(checkpoint, value, require_rng_metadata=False)
        rng_names = self._expected_indexed_files("random_states", ".pkl", value["world_size"], rank_style=True)
        value["rng_state"] = {
            "format": "accelerate-1.x",
            "files": {name: sha256_file(checkpoint / name) for name in sorted(rng_names)},
        }
        value["checkpoint_fingerprint"] = fingerprint(value)
        self._validate_metadata(value)
        return value

    def _validate_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        require_rng: bool = True,
        require_fingerprint: bool = True,
    ) -> None:
        required = (
            self._REQUIRED_SUPPLIED_METADATA
            | self._PROGRESS_FIELDS
            | {
                "schema_version",
                "checkpoint_kind",
                "accumulation_microstep",
            }
        )
        if require_rng:
            required = required | {"rng_state"}
        if require_fingerprint:
            required = required | {"checkpoint_fingerprint"}
        missing = sorted(required - set(metadata))
        if missing:
            raise CheckpointError(f"checkpoint metadata is missing required fields: {missing}")
        if metadata["schema_version"] != self._SCHEMA_VERSION:
            raise CheckpointError(f"unsupported checkpoint schema_version: {metadata['schema_version']!r}")
        if metadata["checkpoint_kind"] not in self._CHECKPOINT_KINDS:
            raise CheckpointError(f"unsupported checkpoint kind: {metadata['checkpoint_kind']!r}")
        accumulation_microstep = metadata["accumulation_microstep"]
        if (
            isinstance(accumulation_microstep, bool)
            or not isinstance(accumulation_microstep, int)
            or accumulation_microstep < 0
        ):
            raise CheckpointError("checkpoint metadata accumulation_microstep must be a non-negative integer")
        for name in (
            "stage_epochs",
            "world_size",
            "microbatch",
            "accumulation",
            "global_batch_size",
            "optimizer_steps_per_epoch",
            "optimizer_count",
            "scheduler_count",
            "checkpointable_count",
        ):
            value = metadata[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CheckpointError(f"checkpoint metadata {name} must be a positive integer")
        if metadata["global_batch_size"] != metadata["world_size"] * metadata["microbatch"] * metadata["accumulation"]:
            raise CheckpointError("checkpoint metadata global_batch_size is inconsistent")
        if metadata["checkpointable_count"] != 1:
            raise CheckpointError("checkpoint metadata checkpointable_count must be exactly one for TrainingProgress")
        progress = TrainingProgress(**{field: metadata[field] for field in self._PROGRESS_FIELDS})
        self._validate_progress_geometry(progress, metadata)
        for name in (
            "base_revision",
            "evaluator_revision",
            "data_fingerprint",
            "selection_fingerprint",
            "lora_fingerprint",
            "optimization_fingerprint",
            "wandb_run_id",
        ):
            if not isinstance(metadata[name], str) or not metadata[name]:
                raise CheckpointError(f"checkpoint metadata {name} must be a non-empty string")
        if metadata["wandb_group"] is not None and not isinstance(metadata["wandb_group"], str):
            raise CheckpointError("checkpoint metadata wandb_group must be a string or null")
        if require_rng:
            rng_state = metadata["rng_state"]
            if (
                not isinstance(rng_state, Mapping)
                or rng_state.get("format") != "accelerate-1.x"
                or not isinstance(rng_state.get("files"), Mapping)
            ):
                raise CheckpointError("checkpoint metadata rng_state must bind Accelerate RNG files")
        if require_fingerprint and not isinstance(metadata["checkpoint_fingerprint"], str):
            raise CheckpointError("checkpoint metadata checkpoint_fingerprint must be a string")

    def _save_adapter_hook(self, accelerator: Any, model: Any):
        def hook(models: list[Any], weights: list[dict[str, torch.Tensor]], output_dir: str | Path) -> None:
            target_state = accelerator.unwrap_model(model).state_dict()
            source_state = weights[0] if weights else target_state
            adapter = self._normalized_adapter_tensors(source_state, target_state)
            if accelerator.is_main_process:
                save_file(adapter, Path(output_dir) / self._ADAPTER_FILE)
            weights.clear()

        return hook

    def _load_adapter_hook(self, accelerator: Any, model: Any, checkpoint: Path):
        def hook(models: list[Any], input_dir: str | Path) -> None:
            self._load_adapter(accelerator.unwrap_model(model), Path(input_dir) / self._ADAPTER_FILE)
            models.clear()

        return hook

    def _load_adapter(self, model: Any, adapter_path: Path) -> None:
        adapter = load_file(adapter_path, device="cpu")
        self._validate_adapter_keys(adapter_path, model.state_dict(), adapter=adapter)
        result = model.load_state_dict(adapter, strict=False)
        unexpected = getattr(
            result, "unexpected_keys", result[1] if isinstance(result, tuple) and len(result) > 1 else ()
        )
        if unexpected:
            raise CheckpointMismatch(f"adapter contains unexpected model keys: {sorted(unexpected)}")

    @staticmethod
    def _adapter_tensors(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        adapter: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if "lora_" not in key:
                continue
            if not isinstance(value, torch.Tensor):
                raise CheckpointError(f"LoRA state {key!r} is not a tensor")
            adapter[key] = value.detach().cpu().contiguous()
        if not adapter:
            raise CheckpointError("model state contains no LoRA tensors")
        return adapter

    @classmethod
    def _normalized_adapter_tensors(
        cls,
        source_state: Mapping[str, Any],
        target_state: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        target = cls._adapter_tensors(target_state)
        normalized: dict[str, torch.Tensor] = {}
        extra: list[str] = []
        for source_key, tensor in cls._adapter_tensors(source_state).items():
            candidate = source_key
            while candidate not in target and candidate.startswith("module."):
                candidate = candidate.removeprefix("module.")
            if candidate not in target:
                extra.append(source_key)
                continue
            if candidate in normalized:
                raise CheckpointMismatch(f"multiple wrapped LoRA keys normalize to {candidate!r}")
            normalized[candidate] = tensor
        missing = sorted(set(target) - set(normalized))
        if missing or extra:
            raise CheckpointMismatch(f"model LoRA key mismatch; missing={missing}, extra={sorted(extra)}")
        return normalized

    def _validate_adapter_keys(
        self,
        adapter_path: Path,
        target_state: Mapping[str, Any] | None = None,
        *,
        adapter: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        try:
            adapter = dict(adapter) if adapter is not None else load_file(adapter_path, device="cpu")
        except Exception as error:
            raise CheckpointError(f"cannot read LoRA adapter: {error}") from error
        keys = set(adapter)
        if not keys:
            raise CheckpointError("LoRA adapter is empty")
        non_lora = sorted(key for key in keys if "lora_" not in key)
        if non_lora:
            raise CheckpointError(f"adapter contains non-LoRA keys: {non_lora}")
        if target_state is not None:
            target_keys = set(self._adapter_tensors(target_state))
            missing = sorted(target_keys - keys)
            extra = sorted(keys - target_keys)
            if missing or extra:
                raise CheckpointMismatch(f"adapter LoRA key mismatch; missing={missing}, extra={extra}")

    def _register_progress(self, accelerator: Any, progress: TrainingProgress) -> None:
        identity = (id(accelerator), id(progress))
        if identity not in self._registrations:
            accelerator.register_for_checkpointing(progress)
            self._registrations.add(identity)

    def _write_manifest(self, checkpoint: Path) -> None:
        checkpoint = Path(checkpoint)
        files = {
            str(path.relative_to(checkpoint)): sha256_file(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file() and path.name != self._MANIFEST_FILE
        }
        atomic_json(checkpoint / self._MANIFEST_FILE, {"schema_version": self._SCHEMA_VERSION, "files": files})

    def _require_exact_state_files(
        self,
        checkpoint: Path,
        metadata: Mapping[str, Any],
        *,
        require_rng_metadata: bool = True,
    ) -> None:
        filenames = {path.name for path in Path(checkpoint).iterdir() if path.is_file()}
        specifications = {
            "optimizer": (
                re.compile(r"^optimizer(?:_\d+)?\.bin$"),
                self._expected_indexed_files("optimizer", ".bin", metadata["optimizer_count"]),
            ),
            "scheduler": (
                re.compile(r"^scheduler(?:_\d+)?\.bin$"),
                self._expected_indexed_files("scheduler", ".bin", metadata["scheduler_count"]),
            ),
            "registered progress": (
                re.compile(r"^custom_checkpoint_\d+\.pkl$"),
                self._expected_indexed_files(
                    "custom_checkpoint", ".pkl", metadata["checkpointable_count"], rank_style=True
                ),
            ),
            "RNG": (
                re.compile(r"^random_states_\d+\.pkl$"),
                self._expected_indexed_files("random_states", ".pkl", metadata["world_size"], rank_style=True),
            ),
        }
        for category, (pattern, expected) in specifications.items():
            actual = {name for name in filenames if pattern.fullmatch(name)}
            if actual != expected:
                raise CheckpointError(
                    f"checkpoint {category} file set mismatch; missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
        if require_rng_metadata:
            rng_files = metadata["rng_state"]["files"]
            expected_rng = specifications["RNG"][1]
            if set(rng_files) != expected_rng:
                raise CheckpointError(
                    f"checkpoint RNG metadata file set mismatch; missing={sorted(expected_rng - set(rng_files))}, "
                    f"extra={sorted(set(rng_files) - expected_rng)}"
                )
            for name, expected_hash in rng_files.items():
                if sha256_file(Path(checkpoint) / name) != expected_hash:
                    raise CheckpointError(f"checkpoint RNG metadata checksum mismatch for {name}")

    @staticmethod
    def _expected_indexed_files(stem: str, suffix: str, count: int, *, rank_style: bool = False) -> set[str]:
        if rank_style:
            return {f"{stem}_{index}{suffix}" for index in range(count)}
        return {f"{stem}{'' if index == 0 else f'_{index}'}{suffix}" for index in range(count)}

    @staticmethod
    def _validate_progress_geometry(progress: TrainingProgress, metadata: Mapping[str, Any]) -> None:
        steps_per_epoch = metadata["optimizer_steps_per_epoch"]
        if steps_per_epoch < 8:
            raise CheckpointError("checkpoint metadata optimizer_steps_per_epoch must be at least eight")
        if progress.epoch >= metadata["stage_epochs"]:
            raise CheckpointError("checkpoint metadata epoch is outside stage_epochs")
        epoch_start = progress.epoch * steps_per_epoch
        epoch_step = progress.optimizer_step - epoch_start
        if not 1 <= epoch_step <= steps_per_epoch:
            raise CheckpointError("checkpoint metadata optimizer_step is outside the active epoch")
        boundaries = tuple((fraction * steps_per_epoch + 7) // 8 for fraction in range(1, 9))
        reached = sum(step <= epoch_step for step in boundaries)
        if metadata["checkpoint_kind"] == "boundary":
            if progress.boundary != reached or progress.optimizer_step != epoch_start + boundaries[reached - 1]:
                raise CheckpointError(
                    "checkpoint metadata boundary does not match the completed optimizer_step boundary cursor"
                )
        elif progress.boundary != reached:
            raise CheckpointError("recovery checkpoint boundary cursor does not match the completed optimizer_step")
        if metadata["accumulation_microstep"] != 0:
            raise CheckpointError("checkpoint requires a complete accumulation group with accumulation_microstep=0")
        expected_microstep = progress.optimizer_step * metadata["accumulation"]
        if progress.microstep != expected_microstep:
            raise CheckpointError(
                f"checkpoint metadata microstep must equal optimizer_step * accumulation ({expected_microstep})"
            )
        if progress.sampler_epoch != progress.epoch:
            raise CheckpointError("checkpoint metadata sampler_epoch must equal epoch")
        stage_start = metadata["stage_start_global_step"]
        if not isinstance(stage_start, int) or isinstance(stage_start, bool) or stage_start < 0:
            raise CheckpointError("checkpoint metadata stage_start_global_step must be a non-negative integer")
        if progress.global_step != stage_start + progress.optimizer_step:
            raise CheckpointError("checkpoint metadata global_step is inconsistent with stage-relative optimizer_step")
        if CheckpointManager._is_stage_one(progress.stage) and stage_start != 0:
            raise CheckpointError("stage1 checkpoint metadata stage_start_global_step must be zero")

    def _preflight_progress(self, checkpoint: Path, metadata: Mapping[str, Any]) -> None:
        progress_path = checkpoint / "custom_checkpoint_0.pkl"
        try:
            state = torch.load(progress_path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise CheckpointError(f"cannot preflight registered progress state: {error}") from error
        if not isinstance(state, Mapping):
            raise CheckpointError("registered progress state must be a mapping")
        expected = {
            "schema_version": TrainingProgress._SCHEMA_VERSION,
            **{field: metadata[field] for field in self._PROGRESS_FIELDS},
        }
        if dict(state) != expected:
            raise CheckpointError("registered progress state does not match checkpoint metadata")
        probe = TrainingProgress(stage=metadata["stage"])
        try:
            probe.load_state_dict(state)
        except (TypeError, ValueError) as error:
            raise CheckpointError(f"registered progress state is invalid: {error}") from error

    @staticmethod
    def _assert_progress_matches_metadata(progress: TrainingProgress, metadata: Mapping[str, Any]) -> None:
        expected = {
            "schema_version": TrainingProgress._SCHEMA_VERSION,
            **{field: metadata[field] for field in CheckpointManager._PROGRESS_FIELDS},
        }
        if progress.state_dict() != expected:
            raise CheckpointRestoreError(
                "Accelerate returned without restoring exact progress; process restart required"
            )

    @staticmethod
    def _require_expected_identity(expected: Mapping[str, Any], required: frozenset[str], description: str) -> None:
        missing = sorted(required - set(expected))
        if missing:
            raise CheckpointMismatch(f"{description} is incomplete; missing={missing}")

    def _ensure_usable(self) -> None:
        if self._poisoned_reason is not None:
            raise CheckpointRestoreError(
                f"checkpoint manager is unusable after a partial restore; process restart required: "
                f"{self._poisoned_reason}"
            )

    @classmethod
    def _ensure_supported_accelerator(cls, accelerator: Any) -> None:
        distributed_type = getattr(accelerator, "distributed_type", None)
        value = getattr(distributed_type, "value", distributed_type)
        normalized = str(value).upper()
        if normalized not in cls._SUPPORTED_DISTRIBUTED_TYPES:
            raise CheckpointError(
                f"unsupported Accelerate distributed type {normalized}; "
                "LoRA-only checkpoints support only NO, MULTI_CPU, and MULTI_GPU"
            )

    @staticmethod
    def _validate_accelerator_world_size(accelerator: Any, expected_world_size: int) -> None:
        actual = getattr(accelerator, "num_processes", getattr(accelerator, "world_size", None))
        if actual is not None and actual != expected_world_size:
            raise CheckpointMismatch(f"Accelerate world_size mismatch: expected {expected_world_size}, found {actual}")

    @staticmethod
    def _verify_expected(actual: Mapping[str, Any], expected: Mapping[str, Any], prefix: str = "") -> None:
        for key, expected_value in expected.items():
            qualified = f"{prefix}.{key}" if prefix else str(key)
            if key not in actual:
                raise CheckpointMismatch(f"{qualified} is missing from checkpoint metadata")
            actual_value = actual[key]
            if isinstance(expected_value, Mapping):
                if not isinstance(actual_value, Mapping):
                    raise CheckpointMismatch(f"{qualified} mismatch: expected mapping, found {actual_value!r}")
                CheckpointManager._verify_expected(actual_value, expected_value, qualified)
            elif actual_value != expected_value:
                raise CheckpointMismatch(f"{qualified} mismatch: expected {expected_value!r}, found {actual_value!r}")

    @staticmethod
    def _read_json(path: Path, description: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(f"cannot read checkpoint {description}: {error}") from error
        if not isinstance(value, dict):
            raise CheckpointError(f"checkpoint {description} must be a JSON object")
        return value

    @staticmethod
    def _checkpoint_name(progress: TrainingProgress) -> str:
        return f"{progress.stage}-epoch-{progress.epoch + 1:04d}-boundary-{progress.boundary:02d}"

    @staticmethod
    def _recovery_name(progress: TrainingProgress) -> str:
        return f"{progress.stage}-epoch-{progress.epoch + 1:04d}-recovery-step-{progress.optimizer_step:010d}"

    @staticmethod
    def _validate_checkpoint_name(name: str) -> None:
        if not name or name in {".", ".."} or Path(name).name != name or name.startswith("."):
            raise CheckpointError(f"invalid checkpoint name: {name!r}")

    @staticmethod
    def _is_stage_one(stage: Any) -> bool:
        return str(stage).lower().replace("_", "") in {"1", "stage1"}

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _fsync_tree(cls, checkpoint: Path) -> None:
        for path in sorted(checkpoint.rglob("*")):
            if path.is_file():
                with path.open("rb") as file:
                    os.fsync(file.fileno())
        cls._fsync_directory(checkpoint)
