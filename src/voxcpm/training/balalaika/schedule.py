"""Exact optimizer-step geometry for Balalaika fractional validations."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class EpochGeometry:
    """Integer-only geometry for one equally sized, sharded training epoch."""

    dataset_rows: int
    world_size: int
    microbatch: int
    accumulation: int
    global_microbatch_size: int
    available_microsteps_per_epoch: int
    microsteps_per_epoch: int
    optimizer_steps_per_epoch: int
    dropped_accumulation_microsteps: int
    dropped_unbatched_samples: int
    dropped_samples: int

    @classmethod
    def from_counts(
        cls,
        dataset_rows: int,
        world_size: int,
        microbatch: int,
        accumulation: int,
    ) -> "EpochGeometry":
        counts = {
            "dataset_rows": dataset_rows,
            "world_size": world_size,
            "microbatch": microbatch,
            "accumulation": accumulation,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in counts.values()):
            raise ValueError("dataset_rows, world_size, microbatch, and accumulation must be positive integers")

        global_microbatch_size = world_size * microbatch
        available_microsteps, dropped_unbatched_samples = divmod(dataset_rows, global_microbatch_size)
        optimizer_steps, dropped_accumulation_microsteps = divmod(available_microsteps, accumulation)
        if optimizer_steps < 8:
            raise ValueError("epoch geometry must contain at least eight optimizer steps for eight unique boundaries")

        microsteps = optimizer_steps * accumulation
        used_samples = microsteps * global_microbatch_size
        return cls(
            dataset_rows=dataset_rows,
            world_size=world_size,
            microbatch=microbatch,
            accumulation=accumulation,
            global_microbatch_size=global_microbatch_size,
            available_microsteps_per_epoch=available_microsteps,
            microsteps_per_epoch=microsteps,
            optimizer_steps_per_epoch=optimizer_steps,
            dropped_accumulation_microsteps=dropped_accumulation_microsteps,
            dropped_unbatched_samples=dropped_unbatched_samples,
            dropped_samples=dataset_rows - used_samples,
        )

    def validation_steps(self, epoch: int) -> tuple[int, ...]:
        """Return the eight stage-relative completed optimizer steps for *epoch*."""
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        offset = epoch * self.optimizer_steps_per_epoch
        # ceil(numerator / 8), without floating-point rounding.
        boundaries = tuple(offset + (fraction * self.optimizer_steps_per_epoch + 7) // 8 for fraction in range(1, 9))
        if len(set(boundaries)) != 8:
            raise ValueError("epoch geometry cannot produce eight unique validation boundaries")
        return boundaries

    def validation_steps_for_epochs(self, epochs: int) -> tuple[int, ...]:
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        return tuple(step for epoch in range(epochs) for step in self.validation_steps(epoch))


@dataclass
class TrainingProgress:
    """Accelerate-checkpointable stage progress and once-only boundary cursor."""

    stage: str
    epoch: int = 0
    boundary: int = 0
    microstep: int = 0
    optimizer_step: int = 0
    global_step: int = 0
    sampler_seed: int = 0
    sampler_epoch: int = 0

    _SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        self._validate()

    def state_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": self._SCHEMA_VERSION,
            **{field.name: getattr(self, field.name) for field in fields(self)},
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected_keys = {"schema_version", *(field.name for field in fields(self))}
        actual_keys = set(state_dict)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"invalid training progress fields; missing={missing}, extra={extra}")
        if state_dict["schema_version"] != self._SCHEMA_VERSION:
            raise ValueError(f"unsupported training progress schema_version: {state_dict['schema_version']!r}")

        previous = self.state_dict()
        try:
            for field in fields(self):
                setattr(self, field.name, state_dict[field.name])
            self._validate()
        except BaseException:
            for field in fields(self):
                setattr(self, field.name, previous[field.name])
            raise

    def boundaries_due(self, geometry: EpochGeometry) -> tuple[int, ...]:
        """Return and consume boundaries reached by already completed optimizer steps."""
        if geometry.accumulation <= 0:  # defensive against manually constructed instances
            raise ValueError("geometry accumulation must be positive")
        steps = geometry.validation_steps(self.epoch)
        due = tuple(
            index for index, step in enumerate(steps, start=1) if self.boundary < index and step <= self.optimizer_step
        )
        if due:
            self.boundary = due[-1]
        return due

    def complete_optimizer_step(self, geometry: EpochGeometry) -> tuple[int, ...]:
        """Record one real optimizer step and emit every newly crossed boundary."""
        epoch_start = self.epoch * geometry.optimizer_steps_per_epoch
        epoch_end = epoch_start + geometry.optimizer_steps_per_epoch
        if not epoch_start <= self.optimizer_step < epoch_end:
            raise ValueError("optimizer_step is outside the active epoch; call start_next_epoch after epoch end")
        self.microstep += geometry.accumulation
        self.optimizer_step += 1
        self.global_step += 1
        return self.boundaries_due(geometry)

    def start_next_epoch(self) -> None:
        if self.boundary != 8:
            raise ValueError("cannot start the next epoch before all eight boundaries are complete")
        self.epoch += 1
        self.boundary = 0
        self.sampler_epoch += 1

    def reset_for_stage(self, stage: str, *, sampler_seed: int | None = None) -> None:
        """Reset stage-relative counters without rewinding cross-stage global progress."""
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        self.stage = stage
        self.epoch = 0
        self.boundary = 0
        self.microstep = 0
        self.optimizer_step = 0
        self.sampler_epoch = 0
        if sampler_seed is not None:
            self.sampler_seed = sampler_seed
        self._validate()

    def _validate(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be a non-empty string")
        for name in ("epoch", "boundary", "microstep", "optimizer_step", "global_step", "sampler_epoch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.sampler_seed, int) or isinstance(self.sampler_seed, bool):
            raise ValueError("sampler_seed must be an integer")
        if self.boundary > 8:
            raise ValueError("boundary must be between zero and eight")
