"""Synchronized and side-effect-free startup microbatch probing."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .runtime import TrainingRuntime


class ProbeError(RuntimeError):
    """The distributed microbatch probe cannot select a usable candidate."""


@dataclass(frozen=True)
class ProbeResult:
    microbatch: int
    all_rank_success: bool
    attempted: tuple[int, ...]
    bypassed: bool = False


class ProbeStep(Protocol):
    """One representative forward/backward/optimizer probe operation."""

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __call__(self, runtime: TrainingRuntime, sample: Any) -> None: ...


@dataclass
class _ProbeSnapshot:
    lora: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor | None]
    optimizer: dict[str, Any]
    cpu_rng: torch.Tensor
    cuda_rng: torch.Tensor | None


_OOM = 0
_SUCCESS = 1
_TERMINAL = -1


def probe_microbatch(
    runtime: TrainingRuntime,
    candidates: Iterable[int],
    sample_factory: Callable[[int], Any],
    step_fn: ProbeStep,
    *,
    explicit_microbatch: int | None = None,
) -> ProbeResult:
    """Select the largest candidate that completes on every rank without retaining probe state."""
    if explicit_microbatch is not None:
        _validate_candidate(explicit_microbatch, "explicit_microbatch")
        return ProbeResult(explicit_microbatch, all_rank_success=True, attempted=(), bypassed=True)

    ordered = tuple(sorted(set(candidates)))
    if not ordered:
        raise ValueError("microbatch candidates must not be empty")
    for candidate in ordered:
        _validate_candidate(candidate, "microbatch candidate")

    model = getattr(step_fn, "model", None)
    optimizer = getattr(step_fn, "optimizer", None)
    if not isinstance(model, torch.nn.Module) or not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("step_fn must expose the probe model and optimizer")
    target = runtime.unwrap(model)
    snapshot = _snapshot(target, optimizer, runtime.device)

    largest_success: int | None = None
    attempted: list[int] = []
    for candidate in ordered:
        attempted.append(candidate)
        local_status = _SUCCESS
        terminal_error: Exception | None = None
        try:
            step_fn(runtime, sample_factory(candidate))
        except torch.cuda.OutOfMemoryError:
            local_status = _OOM
        except Exception as error:
            local_status = _TERMINAL
            terminal_error = error
        finally:
            _restore(target, optimizer, snapshot, runtime.device)

        statuses = runtime.gather(torch.tensor([local_status], dtype=torch.int8, device=runtime.device)).reshape(-1)
        runtime.barrier()
        if bool((statuses == _TERMINAL).any().item()):
            if terminal_error is not None:
                raise terminal_error
            raise ProbeError("a non-OOM microbatch probe failure occurred on another rank")
        if bool((statuses == _SUCCESS).all().item()):
            largest_success = candidate
        elif local_status == _OOM and runtime.device.type == "cuda":
            torch.cuda.empty_cache()

    if largest_success is None:
        raise ProbeError("no microbatch candidate succeeded on every rank")
    return ProbeResult(largest_success, all_rank_success=True, attempted=tuple(attempted))


def _snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> _ProbeSnapshot:
    parameters = dict(model.named_parameters())
    lora = {name: parameter.detach().clone() for name, parameter in parameters.items() if "lora_" in name}
    if not lora:
        raise ProbeError("probe model has no LoRA parameters")
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in parameters.items()
    }
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return _ProbeSnapshot(
        lora=lora,
        gradients=gradients,
        optimizer=copy.deepcopy(optimizer.state_dict()),
        cpu_rng=torch.get_rng_state().clone(),
        cuda_rng=cuda_rng,
    )


def _restore(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    snapshot: _ProbeSnapshot,
    device: torch.device,
) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in snapshot.lora.items():
            parameters[name].copy_(value)
    for name, value in snapshot.gradients.items():
        parameters[name].grad = None if value is None else value.clone()
    optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer))
    torch.set_rng_state(snapshot.cpu_rng)
    if snapshot.cuda_rng is not None:
        torch.cuda.set_rng_state(snapshot.cuda_rng, device)


def _validate_candidate(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
