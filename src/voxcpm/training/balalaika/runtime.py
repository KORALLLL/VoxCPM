"""Public distributed-training runtime backed by Hugging Face Accelerate."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from os import PathLike
from typing import Any, Protocol, TypeVar, runtime_checkable

import torch
from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_T = TypeVar("_T")


@runtime_checkable
class TrainingRuntime(Protocol):
    """Training and state operations shared by distributed trainer components."""

    @property
    def rank(self) -> int: ...

    @property
    def world_size(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def sync_gradients(self) -> bool: ...

    def prepare(self, *objects: _T) -> _T | tuple[_T, ...]: ...

    def accumulate(self, *models: torch.nn.Module) -> AbstractContextManager[None]: ...

    def backward(self, loss: torch.Tensor) -> None: ...

    def clip_grad_norm_(self, parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor: ...

    def gather(self, value: Any) -> Any: ...

    def barrier(self) -> None: ...

    def unwrap(self, model: _T) -> _T: ...

    def save(self, output_dir: str | PathLike[str]) -> None: ...

    def load(self, input_dir: str | PathLike[str]) -> None: ...


class AccelerateRuntime:
    """Small Accelerate facade that keeps trainer code on supported public APIs."""

    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator

    @classmethod
    def create(cls, config: Any, *, accelerator_cls: type[Accelerator] = Accelerator) -> "AccelerateRuntime":
        """Create the required BF16, seedable, non-padding Accelerate runtime."""
        accumulation = _config_accumulation(config)
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
            even_batches=False,
            use_seedable_sampler=True,
        )
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        return cls(
            accelerator_cls(
                cpu=_config_cpu(config),
                mixed_precision="bf16",
                gradient_accumulation_steps=accumulation,
                # The trainer owns the single scheduler.step() that follows
                # each real global optimizer update.  Accelerate's default
                # wrapper otherwise repeats that call once per process when
                # split_batches=False.
                step_scheduler_with_optimizer=False,
                dataloader_config=dataloader_config,
                kwargs_handlers=[ddp_kwargs],
            )
        )

    @property
    def rank(self) -> int:
        return int(self.accelerator.process_index)

    @property
    def world_size(self) -> int:
        return int(self.accelerator.num_processes)

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def sync_gradients(self) -> bool:
        return bool(self.accelerator.sync_gradients)

    def prepare(self, *objects: _T) -> _T | tuple[_T, ...]:
        for value in objects:
            if isinstance(value, DataLoader):
                _reject_distributed_sampler(value)
        return self.accelerator.prepare(*objects)

    def accumulate(self, *models: torch.nn.Module) -> AbstractContextManager[None]:
        return self.accelerator.accumulate(*models)

    def backward(self, loss: torch.Tensor) -> None:
        self.accelerator.backward(loss)

    def clip_grad_norm_(self, parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
        return self.accelerator.clip_grad_norm_(parameters, max_norm)

    def gather(self, value: Any) -> Any:
        return self.accelerator.gather_for_metrics(value)

    def barrier(self) -> None:
        self.accelerator.wait_for_everyone()

    def unwrap(self, model: _T) -> _T:
        return self.accelerator.unwrap_model(model)

    def save(self, output_dir: str | PathLike[str]) -> None:
        local_error: BaseException | None = None
        try:
            self.accelerator.save_state(str(output_dir))
        except BaseException as error:
            local_error = error
        status = torch.tensor([0 if local_error is not None else 1], dtype=torch.int8, device=self.device)
        try:
            gathered = self.accelerator.gather_for_metrics(status).reshape(-1)
        except BaseException as collective_error:
            raise RuntimeError("state-save outcome collective failed; process restart required") from (
                local_error or collective_error
            )
        if not bool((gathered == 1).all().item()):
            if local_error is not None:
                raise local_error
            raise RuntimeError("peer rank failed during state save")
        self.accelerator.wait_for_everyone()

    def load(self, input_dir: str | PathLike[str]) -> None:
        self.accelerator.load_state(str(input_dir))


def _config_accumulation(config: Any) -> int:
    if isinstance(config, Mapping):
        accumulation = config.get("accumulation")
    else:
        accumulation = getattr(config, "accumulation", None)
    if isinstance(accumulation, bool) or not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError("runtime accumulation must be a positive integer")
    return accumulation


def _config_cpu(config: Any) -> bool:
    value = config.get("cpu", False) if isinstance(config, Mapping) else getattr(config, "cpu", False)
    if not isinstance(value, bool):
        raise ValueError("runtime cpu must be a boolean")
    return value


def _reject_distributed_sampler(loader: DataLoader[Any]) -> None:
    samplers = [getattr(loader, "sampler", None)]
    batch_sampler = getattr(loader, "batch_sampler", None)
    if batch_sampler is not None:
        samplers.append(getattr(batch_sampler, "sampler", None))
    if any(isinstance(sampler, DistributedSampler) for sampler in samplers):
        raise ValueError("input dataloaders must not be pre-wrapped in a DistributedSampler; Accelerate shards them")
