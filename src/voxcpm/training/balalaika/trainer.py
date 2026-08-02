"""Two-stage, resumable LoRA training for the indexed Balalaika corpus."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import signal
from typing import Any, Literal, Protocol

import torch
from torch.optim import AdamW

from voxcpm.training.data import BatchProcessor

from .artifacts import atomic_json, sha256_file
from .checkpoint import CheckpointCollectiveError, CheckpointManager
from .dataset import IndexedBalalaikaDataset, build_unsharded_dataloader
from .schedule import EpochGeometry, TrainingProgress


class ApprovalRequired(RuntimeError):
    """Stage 1 cannot start without a matching manual memorization approval."""


class TrainingRestartRequired(RuntimeError):
    """Distributed state may have diverged and must be restored in a new process."""


class TrainerConfigurationError(ValueError):
    """Trainer inputs cannot produce an identity-safe stage run."""


@dataclass(frozen=True)
class EvaluationBoundary:
    """Evaluator-facing identity for one exact eighth-epoch boundary."""

    stage: str
    epoch: int
    boundary: int
    global_step: int
    stage_progress: float


@dataclass(frozen=True)
class VerifiedAdapterCheckpoint:
    """A stage-1 adapter load paired with the complete caller-expected identity."""

    manager: CheckpointManager
    path: Path
    expected: Mapping[str, Any]


class EvaluatorFactory(Protocol):
    def __call__(self, boundary: EvaluationBoundary, checkpoint: Path) -> Any: ...


_IDENTITY_FIELDS = frozenset(
    {
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


def build_model(
    config: Any,
    stage: Literal[1, 2],
    adapter_checkpoint: VerifiedAdapterCheckpoint | None = None,
    *,
    model_cls: type[Any] | None = None,
    lora_config_cls: type[Any] | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module, Callable[[str], Sequence[int]]]:
    """Load the pinned local VoxCPM2 revision and expose only LoRA for training.

    A direct path is deliberately not accepted for ``adapter_checkpoint``:
    stage transitions must pass a :class:`VerifiedAdapterCheckpoint`, which
    routes the load through Task 9's strict identity and tensor-geometry checks.
    The trainer performs its own transition load so it can also restore global
    progress before creating the fresh stage-2 optimizer and scheduler.
    """
    _stage_name(stage)
    pin, model_dir = _verified_model_pin(config)
    if model_cls is None:
        from voxcpm.model import VoxCPM2Model

        model_cls = VoxCPM2Model

    lora = getattr(config, "lora", None)
    if lora is None:
        raise TrainerConfigurationError("configuration must define LoRA settings")
    if lora_config_cls is None:
        from voxcpm.model.voxcpm2 import LoRAConfig as lora_config_cls

    lora_config = lora_config_cls(
        enable_lm=bool(getattr(lora, "enable_lm", True)),
        enable_dit=bool(getattr(lora, "enable_dit", True)),
        enable_proj=bool(getattr(lora, "enable_proj", False)),
        r=getattr(lora, "r", 32),
        alpha=getattr(lora, "alpha", 32),
        dropout=getattr(lora, "dropout", 0.0),
    )
    model = model_cls.from_local(
        str(model_dir),
        optimize=False,
        training=True,
        lora_config=lora_config,
    )
    if not isinstance(model, torch.nn.Module):
        raise TypeError("VoxCPM2 loader must return a torch.nn.Module")

    if adapter_checkpoint is not None:
        if stage != 2 or not isinstance(adapter_checkpoint, VerifiedAdapterCheckpoint):
            raise TypeError("adapter_checkpoint must be a verified final-stage1 adapter for stage 2")
        adapter_checkpoint.manager.load_stage_adapter(
            model,
            adapter_checkpoint.path,
            expected=adapter_checkpoint.expected,
        )

    audio_vae = getattr(model, "audio_vae", None)
    tokenizer = getattr(model, "text_tokenizer", None)
    if not isinstance(audio_vae, torch.nn.Module):
        raise TrainerConfigurationError("loaded VoxCPM2 model does not expose its AudioVAE")
    if not callable(tokenizer):
        raise TrainerConfigurationError("loaded VoxCPM2 model does not expose a callable text tokenizer")

    trainable = 0
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_" in name)
        trainable += int(parameter.requires_grad)
    if trainable == 0:
        raise TrainerConfigurationError("loaded VoxCPM2 model contains no LoRA parameters")
    audio_vae.requires_grad_(False)
    delattr(model, "audio_vae")
    model.train()
    # Retain the revision in a lightweight runtime attribute for diagnostics;
    # immutable checkpoint identity still comes from the caller-supplied map.
    setattr(model, "balalaika_base_revision", pin["revision"])
    return model, audio_vae, tokenizer


class BalalaikaTrainer:
    """Run stage 1 or stage 2 through one dependency-injected training path."""

    def __init__(
        self,
        config: Any,
        runtime: Any,
        *,
        checkpoint_manager: CheckpointManager,
        evaluator_factory: EvaluatorFactory,
        identity: Mapping[str, Any],
        approval_verifier: Callable[[Path, Mapping[str, Any]], Any] | None = None,
        approval_path: Path | None = None,
        approval_expected: Mapping[str, Any] | None = None,
        model_builder: Callable[..., tuple[torch.nn.Module, torch.nn.Module, Callable[..., Any]]] = build_model,
        dataset_factory: Callable[[Any, int, Callable[..., Any]], Any] | None = None,
        loader_factory: Callable[..., Any] | None = None,
        batch_processor_factory: Callable[[torch.nn.Module, torch.nn.Module, Any], Callable[..., Any]] | None = None,
        optimizer_factory: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler_factory: Callable[..., Any] | None = None,
        microbatch_selector: Callable[[Any, torch.nn.Module, torch.optim.Optimizer, Any], int] | None = None,
        boundary_marker: Callable[[EvaluationBoundary, Path], Path] | None = None,
        accumulation: int = 1,
        workers: int = 0,
        sampler_seed: int = 0,
        warmup_fraction: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        loss_weights: Mapping[str, float] | None = None,
        resume_checkpoint: Path | None = None,
        stage1_checkpoint: Path | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.checkpoint_manager = checkpoint_manager
        self.evaluator_factory = evaluator_factory
        self.identity = _validated_identity(identity)
        self.approval_verifier = approval_verifier
        self.approval_path = Path(approval_path) if approval_path is not None else None
        self.approval_expected = dict(approval_expected) if approval_expected is not None else None
        self.model_builder = model_builder
        self.dataset_factory = dataset_factory or _default_dataset_factory
        self.loader_factory = loader_factory or _default_loader_factory
        self.batch_processor_factory = batch_processor_factory or _default_batch_processor_factory
        self.optimizer_factory = optimizer_factory or _default_optimizer_factory
        self.scheduler_factory = scheduler_factory or _default_scheduler_factory
        self.microbatch_selector = microbatch_selector
        self.boundary_marker = boundary_marker
        self.accumulation = _positive_integer(accumulation, "accumulation")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
            raise ValueError("workers must be a non-negative integer")
        if isinstance(sampler_seed, bool) or not isinstance(sampler_seed, int):
            raise ValueError("sampler_seed must be an integer")
        if not 0.0 <= warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be between zero and one")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        self.workers = workers
        self.sampler_seed = sampler_seed
        self.warmup_fraction = float(warmup_fraction)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.loss_weights = dict(loss_weights or {"loss/diff": 1.0, "loss/stop": 1.0})
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        self.stage1_checkpoint = Path(stage1_checkpoint) if stage1_checkpoint is not None else None
        self.install_signal_handlers = install_signal_handlers
        self.last_durable_checkpoint: Path | None = None
        self.model: torch.nn.Module | None = None
        self.audio_vae: torch.nn.Module | None = None
        self.tokenizer: Callable[..., Any] | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None
        self.progress: TrainingProgress | None = None
        self._stop_signal: int | None = None
        self._safe_for_recovery = True
        self._accumulation_cursor = 0
        self._coordinated_recovery = True
        self._pending_boundary_checkpoint: Path | None = None
        self._completed_boundary_proof: dict[str, Any] | None = None

    def request_stop(self, signum: int = signal.SIGTERM) -> None:
        """Request cooperative stop after the current complete accumulation group."""
        self._stop_signal = int(signum)

    def run_stage(self, stage: Literal[1, 2]) -> Path:
        """Train one complete curriculum stage or stop at a durable recovery."""
        stage_name = _stage_name(stage)
        stage_config = getattr(self.config, stage_name, None)
        if stage_config is None:
            raise TrainerConfigurationError(f"configuration does not define {stage_name}")
        self._validate_curriculum()
        if stage == 1:
            self._verify_stage1_approval()

        model, audio_vae, tokenizer = self.model_builder(self.config, stage, adapter_checkpoint=None)
        self.model, self.audio_vae, self.tokenizer = model, audio_vae, tokenizer
        progress = TrainingProgress(stage=stage_name, sampler_seed=self.sampler_seed)
        self.progress = progress

        if stage == 2 and self.resume_checkpoint is None:
            if self.stage1_checkpoint is None:
                raise TrainerConfigurationError("stage 2 requires the final stage1 adapter checkpoint")
            metadata = self.checkpoint_manager.load_stage_adapter(
                model,
                self.stage1_checkpoint,
                expected=self._stage2_expected(),
                progress=progress,
                sampler_seed=self.sampler_seed,
            )
            # Enforce the evolved transition contract even for injected managers.
            progress.global_step = _nonnegative_integer(metadata.get("global_step"), "stage1 global_step")
            progress.reset_for_stage("stage2", sampler_seed=self.sampler_seed)

        trainable = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        if not trainable:
            raise TrainerConfigurationError("optimizer requires at least one trainable LoRA parameter")
        if any("lora_" not in name for name, parameter in model.named_parameters() if parameter.requires_grad):
            raise TrainerConfigurationError("only LoRA parameters may be trainable")
        optimizer = self.optimizer_factory(
            trainable,
            lr=float(getattr(stage_config, "learning_rate")),
            weight_decay=self.weight_decay,
        )
        self.optimizer = optimizer

        microbatch = _positive_integer(getattr(stage_config, "batch_size"), f"{stage_name}.batch_size")
        if self.microbatch_selector is not None:
            selected = self.microbatch_selector(self.runtime, model, optimizer, stage_config)
            microbatch = _positive_integer(selected, "selected microbatch")

        dataset = self.dataset_factory(self.config, stage, tokenizer)
        geometry = EpochGeometry.from_counts(len(dataset), self.runtime.world_size, microbatch, self.accumulation)
        stage_start_global_step = progress.global_step
        if stage == 2 and self.resume_checkpoint is not None:
            verified_resume = self.checkpoint_manager.verify(
                self.resume_checkpoint,
                self._resume_preflight_expected(
                    stage_name=stage_name,
                    stage_config=stage_config,
                    geometry=geometry,
                    microbatch=microbatch,
                ),
            )
            if verified_resume.get("checkpoint_kind") not in {"boundary", "recovery"}:
                raise TrainerConfigurationError("stage2 resume checkpoint has no supported checkpoint_kind")
            stage_start_global_step = _nonnegative_integer(
                verified_resume.get("stage_start_global_step"),
                "stage2 stage_start_global_step",
            )
            progress.global_step = stage_start_global_step
        loader = self.loader_factory(
            dataset,
            microbatch=microbatch,
            workers=self.workers,
            seed=self.sampler_seed,
            world_size=self.runtime.world_size,
            accumulation=self.accumulation,
            dropped_samples=geometry.dropped_samples,
        )
        total_steps = geometry.optimizer_steps_per_epoch * _positive_integer(
            getattr(stage_config, "epochs"), f"{stage_name}.epochs"
        )
        scheduler = self.scheduler_factory(
            optimizer,
            warmup_steps=int(total_steps * self.warmup_fraction),
            total_steps=total_steps,
        )
        self.scheduler = scheduler
        processor = self.batch_processor_factory(model, audio_vae, self.runtime)

        prepared = self.runtime.prepare(model, optimizer, loader, scheduler)
        if not isinstance(prepared, tuple) or len(prepared) != 4:
            raise TypeError("runtime.prepare(model, optimizer, loader, scheduler) must return four objects")
        model, optimizer, loader, scheduler = prepared
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        checkpoint_metadata = self._checkpoint_metadata(
            stage_config=stage_config,
            geometry=geometry,
            microbatch=microbatch,
            stage_start_global_step=stage_start_global_step,
        )

        if self.resume_checkpoint is not None:
            restored = self.checkpoint_manager.resume_same_stage(
                _accelerator(self.runtime),
                model,
                progress,
                self.resume_checkpoint,
                expected={
                    **checkpoint_metadata,
                    "stage": stage_name,
                    "sampler_seed": self.sampler_seed,
                },
            )
            checkpoint_kind = restored.get("checkpoint_kind")
            if checkpoint_kind not in {"boundary", "recovery"}:
                raise TrainerConfigurationError("resume checkpoint has no supported checkpoint_kind")
            self.last_durable_checkpoint = self.resume_checkpoint
            at_validation_boundary = self._is_exact_validation_boundary(progress, geometry)
            if checkpoint_kind == "boundary" and not at_validation_boundary:
                raise TrainerConfigurationError("boundary checkpoint progress is not at an exact validation boundary")
            if at_validation_boundary:
                boundary = self._evaluation_boundary(progress, total_steps)
                proof_complete = checkpoint_kind == "recovery" and self._recovery_boundary_proof_complete(
                    self.resume_checkpoint,
                    restored,
                    progress,
                    boundary,
                    checkpoint_metadata,
                )
                if not proof_complete:
                    self._finish_boundary(model, audio_vae, self.resume_checkpoint, boundary)

        with self._signal_handlers():
            return self._train(
                model=model,
                audio_vae=audio_vae,
                optimizer=optimizer,
                scheduler=scheduler,
                loader=loader,
                processor=processor,
                progress=progress,
                geometry=geometry,
                total_steps=total_steps,
                stage_epochs=int(getattr(stage_config, "epochs")),
                checkpoint_metadata=checkpoint_metadata,
            )

    def _train(
        self,
        *,
        model: torch.nn.Module,
        audio_vae: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loader: Any,
        processor: Callable[[Any], Mapping[str, torch.Tensor]],
        progress: TrainingProgress,
        geometry: EpochGeometry,
        total_steps: int,
        stage_epochs: int,
        checkpoint_metadata: Mapping[str, Any],
    ) -> Path:
        try:
            while progress.epoch < stage_epochs:
                epoch_end_step = (progress.epoch + 1) * geometry.optimizer_steps_per_epoch
                if progress.optimizer_step >= epoch_end_step:
                    if progress.boundary != 8:
                        raise TrainerConfigurationError("completed epoch is missing a validation boundary")
                    if progress.epoch + 1 == stage_epochs:
                        if self.last_durable_checkpoint is None:
                            raise TrainerConfigurationError("completed stage has no durable final checkpoint")
                        return self.last_durable_checkpoint
                    progress.start_next_epoch()
                    continue

                epoch_iterator = self._epoch_iterator(loader, progress, geometry)
                while progress.optimizer_step < epoch_end_step:
                    self._coordinated_recovery = True
                    batch = self._next_coordinated_batch(epoch_iterator)
                    with self.runtime.accumulate(model):
                        outputs_error: BaseException | None = None
                        total_loss: torch.Tensor | None = None
                        try:
                            processed = processor(batch)
                            outputs = _forward(model, processed, progress.optimizer_step / total_steps)
                            total_loss = _weighted_loss(outputs, self.loss_weights)
                        except BaseException as error:
                            outputs_error = error
                        self._coordinate_local_phase(outputs_error is None, "forward")
                        if outputs_error is not None:
                            raise outputs_error
                        assert total_loss is not None

                        self._safe_for_recovery = False
                        self._coordinated_recovery = False
                        self.runtime.backward(total_loss)
                        self._accumulation_cursor += 1
                        if self._accumulation_cursor > geometry.accumulation:
                            raise TrainingRestartRequired(
                                "runtime accumulation cursor exceeded configured accumulation"
                            )
                        if not self.runtime.sync_gradients:
                            continue
                        if self._accumulation_cursor != geometry.accumulation:
                            raise TrainingRestartRequired(
                                "runtime synchronized before the configured accumulation group completed"
                            )

                        step_error: BaseException | None = None
                        skipped = False
                        try:
                            self.runtime.clip_grad_norm_(
                                (parameter for parameter in model.parameters() if parameter.requires_grad),
                                self.max_grad_norm,
                            )
                            optimizer.step()
                            skipped = _optimizer_step_was_skipped(self.runtime, optimizer)
                            if not skipped:
                                scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                        except BaseException as error:
                            step_error = error
                        self._coordinate_local_phase(step_error is None, "optimizer")
                        if step_error is not None:
                            raise step_error
                        self._accumulation_cursor = 0
                        self._coordinated_recovery = True

                        if skipped:
                            self._safe_for_recovery = False
                            # Observe and latch a remote request, but a skipped
                            # optimizer attempt is not a durable update and may
                            # never be checkpointed as if it were one.
                            self._synchronized_stop_requested()
                            continue

                        due = progress.complete_optimizer_step(geometry)
                        self._safe_for_recovery = True
                        for boundary_index in due:
                            checkpoint = self.checkpoint_manager.save_same_stage(
                                _accelerator(self.runtime),
                                model,
                                progress,
                                checkpoint_metadata,
                            )
                            self.last_durable_checkpoint = checkpoint
                            boundary = self._evaluation_boundary(progress, total_steps, boundary_index)
                            self._finish_boundary(model, audio_vae, checkpoint, boundary)

                        if self._synchronized_stop_requested():
                            return self._save_recovery(model, progress, checkpoint_metadata, reason="signal")
        except BaseException as error:
            if isinstance(error, TrainingRestartRequired):
                raise
            if isinstance(error, CheckpointCollectiveError):
                raise self._restart_required(error) from error
            if self._pending_boundary_checkpoint is not None:
                # The boundary checkpoint precedes evaluation by contract and is
                # the exact state from which the durable evaluator resumes.
                self.last_durable_checkpoint = self._pending_boundary_checkpoint
                raise
            if (
                self._safe_for_recovery
                and self._accumulation_cursor == 0
                and self._coordinated_recovery
                and progress.optimizer_step > 0
            ):
                try:
                    self._save_recovery(model, progress, checkpoint_metadata, reason=type(error).__name__)
                except BaseException as save_error:
                    raise self._restart_required(error, save_error) from error
                raise
            raise self._restart_required(error) from error
        raise TrainerConfigurationError("training stage exited without a durable checkpoint")

    def _coordinate_local_phase(self, succeeded: bool, description: str) -> None:
        status = torch.tensor([1 if succeeded else 0], dtype=torch.int8, device=self.runtime.device)
        self._coordinated_recovery = False
        try:
            gathered = self.runtime.gather(status).reshape(-1)
        except BaseException as error:
            raise self._restart_required(error) from error
        self._coordinated_recovery = True
        if not bool((gathered == 1).all().item()) and succeeded:
            raise RuntimeError(f"peer rank failed during {description} before the next distributed collective")

    def _next_coordinated_batch(self, iterator: Iterable[Any]) -> Any:
        batch: Any = None
        fetch_error: BaseException | None = None
        try:
            batch = next(iterator)
        except BaseException as error:
            fetch_error = error
        self._coordinate_local_phase(fetch_error is None, "data loading")
        if fetch_error is not None:
            raise fetch_error
        return batch

    def _synchronized_stop_requested(self) -> bool:
        requested = torch.tensor(
            [1 if self._stop_signal is not None else 0],
            dtype=torch.int8,
            device=self.runtime.device,
        )
        self._coordinated_recovery = False
        try:
            gathered = self.runtime.gather(requested).reshape(-1)
        except BaseException as error:
            raise self._restart_required(error) from error
        self._coordinated_recovery = True
        stopped = bool((gathered != 0).any().item())
        if stopped and self._stop_signal is None:
            # A non-signalled rank must retain the distributed decision across
            # AMP-overflow retries just like the rank that caught the signal.
            self._stop_signal = 0
        return stopped

    def _save_recovery(
        self,
        model: torch.nn.Module,
        progress: TrainingProgress,
        metadata: Mapping[str, Any],
        *,
        reason: str,
    ) -> Path:
        if self._accumulation_cursor != 0 or not self._safe_for_recovery:
            raise TrainingRestartRequired("cannot save recovery inside an incomplete accumulation group")
        self.runtime.barrier()
        recovery_metadata: dict[str, Any] = {
            **metadata,
            "recovery_reason": reason,
            "recovery_signal": self._stop_signal,
        }
        if self._proof_matches_progress(self._completed_boundary_proof, progress):
            recovery_metadata["completed_boundary_proof"] = self._completed_boundary_proof
        path = self.checkpoint_manager.save_recovery(
            _accelerator(self.runtime),
            model,
            progress,
            recovery_metadata,
        )
        self.last_durable_checkpoint = path
        return path

    def _finish_boundary(
        self,
        model: torch.nn.Module,
        audio_vae: torch.nn.Module,
        checkpoint: Path,
        boundary: EvaluationBoundary,
    ) -> Path:
        self._pending_boundary_checkpoint = checkpoint
        if self.boundary_marker is None:
            existing = self._completed_marker_collectively(checkpoint, boundary)
            if existing is not None:
                self._completed_boundary_proof = self._boundary_proof(checkpoint, boundary)
                self._pending_boundary_checkpoint = None
                return existing
        evaluator = self.evaluator_factory(boundary, checkpoint)
        evaluator.run(model, audio_vae, checkpoint, boundary)
        marker = self._write_boundary_marker_collectively(checkpoint, boundary)
        self._completed_boundary_proof = (
            self._boundary_proof(checkpoint, boundary) if self.boundary_marker is None else None
        )
        self._pending_boundary_checkpoint = None
        return marker

    def _recovery_boundary_proof_complete(
        self,
        recovery_checkpoint: Path,
        recovery_metadata: Mapping[str, Any],
        progress: TrainingProgress,
        boundary: EvaluationBoundary,
        checkpoint_metadata: Mapping[str, Any],
    ) -> bool:
        proof = recovery_metadata.get("completed_boundary_proof")
        complete = False
        if self.runtime.rank == 0:
            complete = self._validate_boundary_proof(
                recovery_checkpoint,
                proof,
                progress,
                boundary,
                checkpoint_metadata,
            )
        outcome = torch.tensor([1 if complete else 0], dtype=torch.int8, device=self.runtime.device)
        try:
            gathered = self.runtime.gather(outcome).reshape(-1)
        except BaseException as error:
            self._coordinated_recovery = False
            raise self._restart_required(error) from error
        complete = bool(gathered[0].item())
        if complete:
            assert isinstance(proof, Mapping)
            self._completed_boundary_proof = dict(proof)
        return complete

    def _validate_boundary_proof(
        self,
        recovery_checkpoint: Path,
        proof: Any,
        progress: TrainingProgress,
        boundary: EvaluationBoundary,
        checkpoint_metadata: Mapping[str, Any],
    ) -> bool:
        if not isinstance(proof, Mapping) or set(proof) != {
            "version",
            "checkpoint_name",
            "checkpoint_fingerprint",
            "boundary",
        }:
            return False
        checkpoint_name = proof.get("checkpoint_name")
        checkpoint_fingerprint = proof.get("checkpoint_fingerprint")
        if (
            proof.get("version") != 1
            or not isinstance(checkpoint_name, str)
            or not checkpoint_name
            or not isinstance(checkpoint_fingerprint, str)
            or not checkpoint_fingerprint
            or Path(checkpoint_name).name != checkpoint_name
            or checkpoint_name.startswith(".")
            or proof.get("boundary") != asdict(boundary)
        ):
            return False
        source_checkpoint = Path(recovery_checkpoint).parent / checkpoint_name
        progress_identity = {key: value for key, value in progress.state_dict().items() if key != "schema_version"}
        try:
            self.checkpoint_manager.verify(
                source_checkpoint,
                {
                    **checkpoint_metadata,
                    **progress_identity,
                    "checkpoint_kind": "boundary",
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                },
            )
            return self._completed_marker(source_checkpoint, boundary) is not None
        except Exception:
            # Missing, corrupt, or mismatched proof is never accepted.  It is
            # safe to fall back to completing validation for the recovery
            # checkpoint itself.
            return False

    def _boundary_proof(self, checkpoint: Path, boundary: EvaluationBoundary) -> dict[str, Any]:
        checkpoint_fingerprint: Any = None
        metadata_error: BaseException | None = None
        try:
            checkpoint_fingerprint = _checkpoint_metadata_file(checkpoint)["checkpoint_fingerprint"]
        except BaseException as error:
            metadata_error = error
        self._coordinate_local_phase(metadata_error is None, "boundary proof")
        if metadata_error is not None:
            raise metadata_error
        return {
            "version": 1,
            "checkpoint_name": Path(checkpoint).name,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "boundary": asdict(boundary),
        }

    @staticmethod
    def _proof_matches_progress(proof: Mapping[str, Any] | None, progress: TrainingProgress) -> bool:
        if not isinstance(proof, Mapping):
            return False
        boundary = proof.get("boundary")
        return isinstance(boundary, Mapping) and all(
            boundary.get(key) == value
            for key, value in {
                "stage": progress.stage,
                "epoch": progress.epoch,
                "boundary": progress.boundary,
                "global_step": progress.global_step,
            }.items()
        )

    def _completed_marker_collectively(
        self,
        checkpoint: Path,
        boundary: EvaluationBoundary,
    ) -> Path | None:
        existing: Path | None = None
        local_error: BaseException | None = None
        if self.runtime.rank == 0:
            try:
                existing = self._completed_marker(checkpoint, boundary)
            except BaseException as error:
                local_error = error
        outcome = torch.tensor(
            [0 if local_error is not None else 1, 1 if existing is not None else 0],
            dtype=torch.int8,
            device=self.runtime.device,
        )
        try:
            gathered = self.runtime.gather(outcome).reshape(-1, 2)
        except BaseException as error:
            self._coordinated_recovery = False
            raise self._restart_required(error) from error
        if int(gathered[0, 0].item()) != 1:
            if local_error is not None:
                raise local_error
            raise RuntimeError("main rank failed while checking the completed boundary marker")
        if int(gathered[0, 1].item()) == 1:
            return existing or self._marker_path(checkpoint)
        return None

    def _write_boundary_marker_collectively(
        self,
        checkpoint: Path,
        boundary: EvaluationBoundary,
    ) -> Path:
        marker = self._marker_path(checkpoint)
        local_error: BaseException | None = None
        if self.runtime.rank == 0:
            try:
                marker = (
                    self.boundary_marker(boundary, checkpoint)
                    if self.boundary_marker is not None
                    else self._write_completed_marker(checkpoint, boundary)
                )
            except BaseException as error:
                local_error = error
        status = torch.tensor([0 if local_error is not None else 1], dtype=torch.int8, device=self.runtime.device)
        try:
            gathered = self.runtime.gather(status).reshape(-1)
        except BaseException as error:
            self._coordinated_recovery = False
            raise self._restart_required(error) from error
        if not bool((gathered == 1).all().item()):
            if local_error is not None:
                raise local_error
            raise RuntimeError("main rank failed while publishing the completed boundary marker")
        return Path(marker)

    def _completed_marker(self, checkpoint: Path, boundary: EvaluationBoundary) -> Path | None:
        path = self._marker_path(checkpoint)
        if not path.is_file():
            return None
        expected = self._marker_value(checkpoint, boundary)
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise TrainerConfigurationError(f"cannot read completed boundary marker {path}: {error}") from error
        except json.JSONDecodeError:
            return None
        if actual != expected:
            return None
        return path

    def _write_completed_marker(self, checkpoint: Path, boundary: EvaluationBoundary) -> Path:
        path = self._marker_path(checkpoint)
        atomic_json(path, self._marker_value(checkpoint, boundary))
        return path

    def _marker_value(self, checkpoint: Path, boundary: EvaluationBoundary) -> dict[str, Any]:
        metadata = _checkpoint_metadata_file(checkpoint)
        return {
            "version": 1,
            "status": "complete",
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
            "boundary": asdict(boundary),
        }

    def _marker_path(self, checkpoint: Path) -> Path:
        return Path(self.config.output_dir) / "completed-boundaries" / f"{Path(checkpoint).name}.json"

    def _evaluation_boundary(
        self,
        progress: TrainingProgress,
        total_steps: int,
        boundary_index: int | None = None,
    ) -> EvaluationBoundary:
        return EvaluationBoundary(
            stage=progress.stage,
            epoch=progress.epoch,
            boundary=progress.boundary if boundary_index is None else boundary_index,
            global_step=progress.global_step,
            stage_progress=progress.optimizer_step / total_steps,
        )

    def _checkpoint_metadata(
        self,
        *,
        stage_config: Any,
        geometry: EpochGeometry,
        microbatch: int,
        stage_start_global_step: int,
    ) -> dict[str, Any]:
        return {
            "stage_epochs": int(getattr(stage_config, "epochs")),
            "world_size": self.runtime.world_size,
            "microbatch": microbatch,
            "accumulation": self.accumulation,
            "global_batch_size": self.runtime.world_size * microbatch * self.accumulation,
            "optimizer_steps_per_epoch": geometry.optimizer_steps_per_epoch,
            "optimizer_count": 1,
            "scheduler_count": 1,
            "checkpointable_count": 1,
            "stage_start_global_step": stage_start_global_step,
            **self.identity,
        }

    def _stage2_expected(self) -> dict[str, Any]:
        stage1 = getattr(self.config, "stage1", None)
        if stage1 is None:
            raise TrainerConfigurationError("stage 2 requires stage1 configuration")
        epochs = _positive_integer(getattr(stage1, "epochs"), "stage1.epochs")
        return {
            "source_stage": "stage1",
            "source_stage_epochs": epochs,
            "source_epoch": epochs - 1,
            "source_boundary": 8,
            **{
                key: self.identity[key]
                for key in ("base_revision", "data_fingerprint", "selection_fingerprint", "lora_fingerprint")
            },
        }

    def _resume_preflight_expected(
        self,
        *,
        stage_name: str,
        stage_config: Any,
        geometry: EpochGeometry,
        microbatch: int,
    ) -> dict[str, Any]:
        """Verify every known immutable field before trusting stage-start progress.

        ``stage_start_global_step`` is the one value being obtained from this
        already hash-verified checkpoint. It is included in the subsequent
        complete ``resume_same_stage`` expected identity before mutation.
        """
        return {
            "stage": stage_name,
            "sampler_seed": self.sampler_seed,
            "stage_epochs": int(getattr(stage_config, "epochs")),
            "world_size": self.runtime.world_size,
            "microbatch": microbatch,
            "accumulation": self.accumulation,
            "global_batch_size": self.runtime.world_size * microbatch * self.accumulation,
            "optimizer_steps_per_epoch": geometry.optimizer_steps_per_epoch,
            "optimizer_count": 1,
            "scheduler_count": 1,
            "checkpointable_count": 1,
            **self.identity,
        }

    def _verify_stage1_approval(self) -> None:
        if self.approval_verifier is None or self.approval_path is None or self.approval_expected is None:
            raise ApprovalRequired("stage 1 requires an injected matching manual memorization approval verifier")
        current_identity = {
            key: self.identity[key]
            for key in ("base_revision", "data_fingerprint", "selection_fingerprint", "lora_fingerprint")
        }
        conflicts = {
            key: (self.approval_expected[key], value)
            for key, value in current_identity.items()
            if key in self.approval_expected and self.approval_expected[key] != value
        }
        if conflicts:
            raise ApprovalRequired(f"manual memorization approval identity conflicts with this stage: {conflicts}")
        self.approval_verifier(
            self.approval_path,
            {**self.approval_expected, **current_identity},
        )

    def _epoch_iterator(
        self,
        loader: Iterable[Any],
        progress: TrainingProgress,
        geometry: EpochGeometry,
    ) -> Iterable[Any]:
        consumed = progress.microstep - progress.epoch * geometry.microsteps_per_epoch
        if not 0 <= consumed <= geometry.microsteps_per_epoch:
            raise TrainerConfigurationError("resume microstep is outside the active epoch")
        _set_loader_epoch(loader, progress.sampler_epoch)
        iterator = iter(loader)
        for _ in range(consumed):
            try:
                next(iterator)
            except StopIteration as error:
                raise TrainerConfigurationError("prepared loader is shorter than checkpoint epoch geometry") from error
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                # An AMP-overflow skip consumes data but not optimizer progress.
                # Replay the deterministic epoch until the planned real update
                # count is reached; complete groups are still the only unit.
                _set_loader_epoch(loader, progress.sampler_epoch)
                iterator = iter(loader)

    def _is_exact_validation_boundary(
        self,
        progress: TrainingProgress,
        geometry: EpochGeometry,
    ) -> bool:
        if progress.boundary == 0:
            return False
        return progress.optimizer_step == geometry.validation_steps(progress.epoch)[progress.boundary - 1]

    def _validate_curriculum(self) -> None:
        mandated = {"stage1": (2, 1e-4), "stage2": (3, 5e-5)}
        for name, (epochs, learning_rate) in mandated.items():
            config = getattr(self.config, name, None)
            actual = (
                getattr(config, "epochs", None) if config is not None else None,
                getattr(config, "learning_rate", None) if config is not None else None,
            )
            if actual != (epochs, learning_rate):
                raise TrainerConfigurationError(
                    f"mandated curriculum requires {name} epochs={epochs} and learning_rate={learning_rate}"
                )

    def _restart_required(
        self,
        cause: BaseException,
        recovery_error: BaseException | None = None,
    ) -> TrainingRestartRequired:
        last = str(self.last_durable_checkpoint) if self.last_durable_checkpoint is not None else "none"
        suffix = f"; recovery save also failed: {recovery_error}" if recovery_error is not None else ""
        return TrainingRestartRequired(
            f"distributed training state may be divergent; restart from last durable checkpoint {last}{suffix}"
        )

    @contextmanager
    def _signal_handlers(self):
        if not self.install_signal_handlers:
            yield
            return
        previous: dict[int, Any] = {}

        def request(signum: int, _frame: Any) -> None:
            self.request_stop(signum)

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.getsignal(signum)
                signal.signal(signum, request)
            yield
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def _verified_model_pin(config: Any) -> tuple[dict[str, Any], Path]:
    hub = getattr(config, "hub", None)
    if hub is None:
        raise TrainerConfigurationError("configuration must define pinned Hub inputs")
    hub_root = Path(getattr(hub, "local_dir"))
    manifest_path = hub_root / "hub-pins.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainerConfigurationError(f"cannot read pinned Hub manifest {manifest_path}: {error}") from error
    pin = manifest.get("model") if isinstance(manifest, dict) else None
    if not isinstance(pin, dict):
        raise TrainerConfigurationError("Hub pins must include the VoxCPM2 model")
    repo_id = getattr(hub, "model_repo_id")
    if pin.get("kind") != "model" or pin.get("repo_id") != repo_id:
        raise TrainerConfigurationError("Hub model pin does not match the configured VoxCPM2 repository")
    revision = pin.get("revision")
    if not isinstance(revision, str) or not revision:
        raise TrainerConfigurationError("Hub model pin must contain an immutable revision")
    model_dir = Path(str(pin.get("local_dir", ""))).resolve()
    if model_dir != (hub_root / "model").resolve():
        raise TrainerConfigurationError("Hub model pin local directory does not match the configured cache")
    files = pin.get("files")
    if not isinstance(files, dict) or not files:
        raise TrainerConfigurationError("Hub model pin must hash every local model file")
    actual_files = {
        path.relative_to(model_dir).as_posix(): sha256_file(path)
        for path in sorted(model_dir.rglob("*"))
        if path.is_file()
    }
    if actual_files != files:
        raise TrainerConfigurationError("pinned local VoxCPM2 files changed after download")
    return pin, model_dir


def _default_dataset_factory(config: Any, stage: int, tokenizer: Callable[..., Any]) -> IndexedBalalaikaDataset:
    index_path = Path(config.data.index_dir) / "balalaika-index.sqlite3"
    sidecar_path = (
        Path(config.data.corpus_root)
        / "combined_sidecars"
        / "rover-punctuation-stress-v1"
        / "rover-punctuation-stress.jsonl"
    )
    return IndexedBalalaikaDataset(index_path, sidecar_path, stage=stage, tokenizer=tokenizer)


def _default_loader_factory(
    dataset: Any,
    *,
    microbatch: int,
    workers: int,
    seed: int,
    world_size: int,
    accumulation: int,
    dropped_samples: int,
) -> Any:
    del dropped_samples
    return build_unsharded_dataloader(
        dataset,
        batch_size=microbatch,
        workers=workers,
        seed=seed,
        world_size=world_size,
        accumulation=accumulation,
    )


def _default_batch_processor_factory(
    model: torch.nn.Module,
    audio_vae: torch.nn.Module,
    runtime: Any,
) -> BatchProcessor:
    return BatchProcessor(config=model.config, audio_vae=audio_vae, dataset_cnt=1, device=runtime.device)


def _default_optimizer_factory(
    parameters: Iterable[torch.nn.Parameter],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return AdamW(parameters, lr=lr, weight_decay=weight_decay)


def _default_scheduler_factory(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> Any:
    # Keep this optional dependency at the execution boundary. Some model-unit
    # tests intentionally install a minimal ``transformers`` stub while
    # collecting unrelated modules; importing the trainer must remain safe.
    from transformers import get_cosine_schedule_with_warmup

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def _forward(model: torch.nn.Module, processed: Mapping[str, torch.Tensor], progress: float) -> Mapping[str, Any]:
    standard = (
        "text_tokens",
        "text_mask",
        "audio_feats",
        "audio_mask",
        "loss_mask",
        "position_ids",
        "labels",
    )
    if all(name in processed for name in standard):
        outputs = model(*(processed[name] for name in standard), progress=progress)
    else:
        outputs = model(**processed, progress=progress)
    if not isinstance(outputs, Mapping):
        raise TypeError("VoxCPM2 training forward must return a loss mapping")
    return outputs


def _weighted_loss(outputs: Mapping[str, Any], weights: Mapping[str, float]) -> torch.Tensor:
    total: torch.Tensor | None = None
    for name, value in outputs.items():
        if not str(name).startswith("loss/"):
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"training loss {name!r} must be a tensor")
        weighted = value * float(weights.get(str(name), 1.0))
        total = weighted if total is None else total + weighted
    if total is None:
        raise TrainerConfigurationError("VoxCPM2 forward returned no loss/* tensors")
    return total


def _validated_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(_IDENTITY_FIELDS - set(identity))
    extra = sorted(set(identity) - _IDENTITY_FIELDS)
    if missing or extra:
        raise TrainerConfigurationError(f"training identity must be complete; missing={missing}, extra={extra}")
    value = dict(identity)
    for name in _IDENTITY_FIELDS - {"wandb_group"}:
        if not isinstance(value[name], str) or not value[name]:
            raise TrainerConfigurationError(f"training identity {name} must be a non-empty string")
    if value["wandb_group"] is not None and not isinstance(value["wandb_group"], str):
        raise TrainerConfigurationError("training identity wandb_group must be a string or null")
    return value


def _checkpoint_metadata_file(checkpoint: Path) -> Mapping[str, Any]:
    path = Path(checkpoint) / "metadata.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainerConfigurationError(f"cannot read checkpoint metadata {path}: {error}") from error
    if not isinstance(value, Mapping) or not isinstance(value.get("checkpoint_fingerprint"), str):
        raise TrainerConfigurationError(f"checkpoint metadata has no fingerprint: {path}")
    return value


def _optimizer_step_was_skipped(runtime: Any, optimizer: Any) -> bool:
    value = getattr(runtime, "optimizer_step_was_skipped", None)
    if value is None:
        value = getattr(getattr(runtime, "accelerator", None), "optimizer_step_was_skipped", None)
    if value is None:
        value = getattr(optimizer, "step_was_skipped", False)
    return bool(value)


def _set_loader_epoch(loader: Any, epoch: int) -> None:
    if callable(getattr(loader, "set_epoch", None)):
        loader.set_epoch(epoch)
        return
    sampler = getattr(loader, "sampler", None)
    if callable(getattr(sampler, "set_epoch", None)):
        sampler.set_epoch(epoch)


def _accelerator(runtime: Any) -> Any:
    return getattr(runtime, "accelerator", runtime)


def _stage_name(stage: Any) -> str:
    if stage not in (1, 2) or isinstance(stage, bool):
        raise ValueError("stage must be 1 or 2")
    return f"stage{stage}"


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainerConfigurationError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "ApprovalRequired",
    "BalalaikaTrainer",
    "EvaluationBoundary",
    "TrainerConfigurationError",
    "TrainingRestartRequired",
    "VerifiedAdapterCheckpoint",
    "build_model",
]
