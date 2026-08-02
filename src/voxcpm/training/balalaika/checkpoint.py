"""Atomic LoRA-only checkpoints backed by Accelerate's public state hooks."""

from __future__ import annotations

import json
import os
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


class CheckpointManager:
    """Publish and restore immutable boundary checkpoints beneath one root."""

    _SCHEMA_VERSION = 1
    _ADAPTER_FILE = "adapter_model.safetensors"
    _METADATA_FILE = "metadata.json"
    _MANIFEST_FILE = "manifest.json"
    _REQUIRED_METADATA = frozenset(
        {
            "stage_epochs",
            "world_size",
            "microbatch",
            "accumulation",
            "global_batch_size",
            "rng_state",
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
    _FRESH_STAGE_FIELDS = frozenset(
        {
            "stage",
            "stage_epochs",
            "world_size",
            "microbatch",
            "accumulation",
            "global_batch_size",
            "rng_state",
            "data_fingerprint",
            "selection_fingerprint",
            "optimization_fingerprint",
            "wandb_run_id",
            "wandb_group",
            *(_PROGRESS_FIELDS - {"stage"}),
        }
    )

    def __init__(self, root: Path):
        self.root = Path(root)
        self._registrations: set[tuple[int, int]] = set()

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
        complete_metadata = self._build_metadata(progress, metadata)
        checkpoint_name = name or self._checkpoint_name(progress)
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
                atomic_json(temporary / self._METADATA_FILE, complete_metadata)
                self._require_state_categories(temporary)
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
        checkpoint = Path(checkpoint)
        metadata = self.verify(checkpoint, expected)
        if metadata["stage"] != progress.stage:
            raise CheckpointMismatch(
                f"stage mismatch: checkpoint has {metadata['stage']!r}, progress expects {progress.stage!r}"
            )

        self._register_progress(accelerator, progress)
        hook = accelerator.register_load_state_pre_hook(self._load_adapter_hook(accelerator, model, checkpoint))
        try:
            accelerator.load_state(str(checkpoint))
        finally:
            hook.remove()
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
        checkpoint = Path(checkpoint)
        compatible_expected = {key: value for key, value in expected.items() if key not in self._FRESH_STAGE_FIELDS}
        metadata = self.verify(checkpoint, compatible_expected)
        if (
            not self._is_stage_one(metadata["stage"])
            or metadata["boundary"] != 8
            or metadata["epoch"] != metadata["stage_epochs"] - 1
        ):
            raise CheckpointMismatch("stage transition requires the final stage1 boundary checkpoint")
        self._load_adapter(model, checkpoint / self._ADAPTER_FILE)
        if progress is not None:
            target_stage = expected.get("stage", "stage2")
            if not isinstance(target_stage, str):
                target_stage = str(target_stage)
            progress.reset_for_stage(target_stage, sampler_seed=sampler_seed)
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
        self._require_state_categories(checkpoint)

        metadata = self._read_json(metadata_path, "metadata")
        self._validate_metadata(metadata)
        fingerprint_payload = {key: value for key, value in metadata.items() if key != "checkpoint_fingerprint"}
        if metadata["checkpoint_fingerprint"] != fingerprint(fingerprint_payload):
            raise CheckpointError("checkpoint metadata fingerprint mismatch")
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

    def _build_metadata(self, progress: TrainingProgress, supplied: Mapping[str, Any]) -> dict[str, Any]:
        supplied = dict(supplied)
        conflicting = self._PROGRESS_FIELDS.intersection(supplied)
        if conflicting:
            raise CheckpointError(f"progress metadata is manager-owned: {sorted(conflicting)}")
        missing = sorted(self._REQUIRED_METADATA - set(supplied))
        if missing:
            raise CheckpointError(f"checkpoint metadata is missing required fields: {missing}")
        value: dict[str, Any] = {
            "schema_version": self._SCHEMA_VERSION,
            **supplied,
            **{key: item for key, item in progress.state_dict().items() if key != "schema_version"},
        }
        self._validate_metadata({**value, "checkpoint_fingerprint": "pending"}, check_fingerprint=False)
        value["checkpoint_fingerprint"] = fingerprint(value)
        return value

    def _validate_metadata(self, metadata: Mapping[str, Any], *, check_fingerprint: bool = True) -> None:
        required = self._REQUIRED_METADATA | self._PROGRESS_FIELDS | {"schema_version", "checkpoint_fingerprint"}
        missing = sorted(required - set(metadata))
        if missing:
            raise CheckpointError(f"checkpoint metadata is missing required fields: {missing}")
        if metadata["schema_version"] != self._SCHEMA_VERSION:
            raise CheckpointError(f"unsupported checkpoint schema_version: {metadata['schema_version']!r}")
        for name in ("stage_epochs", "world_size", "microbatch", "accumulation", "global_batch_size"):
            value = metadata[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise CheckpointError(f"checkpoint metadata {name} must be a positive integer")
        if metadata["global_batch_size"] != metadata["world_size"] * metadata["microbatch"] * metadata["accumulation"]:
            raise CheckpointError("checkpoint metadata global_batch_size is inconsistent")
        progress = TrainingProgress(**{field: metadata[field] for field in self._PROGRESS_FIELDS})
        if progress.epoch >= metadata["stage_epochs"]:
            raise CheckpointError("checkpoint metadata epoch is outside stage_epochs")
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
        if not isinstance(metadata["rng_state"], Mapping) or not metadata["rng_state"]:
            raise CheckpointError("checkpoint metadata rng_state must be a non-empty mapping")
        if check_fingerprint and not isinstance(metadata["checkpoint_fingerprint"], str):
            raise CheckpointError("checkpoint metadata checkpoint_fingerprint must be a string")

    def _save_adapter_hook(self, accelerator: Any, model: Any):
        def hook(models: list[Any], weights: list[dict[str, torch.Tensor]], output_dir: str | Path) -> None:
            source = weights[0] if weights else accelerator.unwrap_model(model).state_dict()
            adapter = self._adapter_tensors(source)
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
        self._validate_adapter_keys(adapter_path)
        result = model.load_state_dict(load_file(adapter_path, device="cpu"), strict=False)
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

    def _validate_adapter_keys(self, adapter_path: Path) -> None:
        try:
            keys = set(load_file(adapter_path, device="cpu"))
        except Exception as error:
            raise CheckpointError(f"cannot read LoRA adapter: {error}") from error
        if not keys:
            raise CheckpointError("LoRA adapter is empty")
        non_lora = sorted(key for key in keys if "lora_" not in key)
        if non_lora:
            raise CheckpointError(f"adapter contains non-LoRA keys: {non_lora}")

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

    @staticmethod
    def _require_state_categories(checkpoint: Path) -> None:
        filenames = {path.name for path in Path(checkpoint).iterdir() if path.is_file()}
        categories = {
            "optimizer": any(name.startswith("optimizer") for name in filenames),
            "scheduler": any(name.startswith("scheduler") for name in filenames),
            "RNG": any(name.startswith("random_states") for name in filenames),
            "registered progress": any(name.startswith("custom_checkpoint") for name in filenames),
        }
        missing = [name for name, present in categories.items() if not present]
        if missing:
            raise CheckpointError(f"checkpoint is missing Accelerate state: {', '.join(missing)}")

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
