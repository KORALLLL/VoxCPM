#!/usr/bin/env python3
"""Real tiny-GPU smoke for Balalaika Accelerate semantics; never loads VoxCPM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import accelerate
import torch
from torch.utils.data import DataLoader, Dataset

from voxcpm.training.balalaika.probe import probe_microbatch
from voxcpm.training.balalaika.runtime import AccelerateRuntime

_ACCUMULATION = 2
_MICROBATCH = 2
_LEARNING_RATE = 0.025


class _SyntheticSamples(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = 0.125 + (index + 1) / 64.0
        return {
            "id": torch.tensor(index, dtype=torch.int64),
            "input": torch.tensor([value], dtype=torch.float32),
            "target": torch.tensor([0.35 * value - 0.025], dtype=torch.float32),
        }


class _TinyLoRAModel(torch.nn.Module):
    """Two scalar LoRA factors and a frozen base are enough to exercise DDP reduction."""

    def __init__(self):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.tensor([[0.2]]), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.tensor([[0.1]]))
        self.lora_B = torch.nn.Parameter(torch.tensor([[0.15]]))
        self.register_buffer("next_sample_position", torch.tensor(0, dtype=torch.int64))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = self.base_weight + self.lora_B @ self.lora_A
        return inputs @ weight.transpose(0, 1)


class _SyntheticLocalProbeStep:
    def __init__(
        self,
        local_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        rank: int,
        fault_rank: int | None = None,
        fault_candidate: int | None = None,
    ):
        self.local_model = local_model
        self.optimizer = optimizer
        self.rank = rank
        self.fault_rank = fault_rank
        self.fault_candidate = fault_candidate
        self.injected_oom = False

    def __call__(self, local_model: torch.nn.Module, sample: tuple[torch.Tensor, torch.Tensor]) -> None:
        inputs, targets = sample
        torch.rand(1, device=inputs.device)
        loss = torch.nn.functional.mse_loss(local_model(inputs), targets)
        loss.backward()
        candidate = int(inputs.shape[0])
        if self.rank == self.fault_rank and candidate == self.fault_candidate:
            self.injected_oom = True
            raise torch.cuda.OutOfMemoryError(f"injected rank-local OOM for candidate {candidate}")
        self.optimizer.step()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train-checkpoint", "probe-rank-fault"), required=True)
    parser.add_argument("--runtime", choices=("accelerate",), default="accelerate")
    return parser.parse_args()


def _lora_values(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.detach().float().reshape(-1) for name, parameter in model.named_parameters() if "lora_" in name]
    )


def _lora_gradients(model: torch.nn.Module) -> torch.Tensor:
    gradients = []
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            if parameter.grad is None:
                raise AssertionError(f"missing synchronized gradient for {name}")
            gradients.append(parameter.grad.detach().float().reshape(-1))
    return torch.cat(gradients)


def _optimizer_step(optimizer: torch.optim.Optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is not None:
            steps.append(int(value.item() if isinstance(value, torch.Tensor) else value))
    if not steps or len(set(steps)) != 1:
        raise AssertionError(f"optimizer parameter steps are absent or unequal: {steps}")
    return steps[0]


def _make_reference(
    initial_state: dict[str, torch.Tensor],
    dataset: _SyntheticSamples,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _TinyLoRAModel().to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LEARNING_RATE, weight_decay=0.0)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    gradients = _lora_gradients(model).clone()
    optimizer.step()
    return gradients, _lora_values(model)


def _checkpoint_path(runtime: AccelerateRuntime) -> Path:
    nonce = time.time_ns() if runtime.rank == 0 else 0
    gathered = runtime.gather(torch.tensor([nonce], dtype=torch.int64, device=runtime.device))
    shared_nonce = int(gathered.reshape(-1)[0].item())
    root = (
        Path(__file__).resolve().parents[1]
        / ".superpowers"
        / "sdd"
        / "2026-08-02-balalaika-two-stage-lora-training"
        / "task-10-artifacts"
    )
    path = root / f"accelerate-state-{runtime.world_size}x-{shared_nonce}"
    if runtime.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    runtime.barrier()
    return path


def _assert_disjoint_ids(runtime: AccelerateRuntime, local_ids: list[int], total: int) -> list[list[int]]:
    local = torch.tensor(local_ids, dtype=torch.int64, device=runtime.device)
    gathered = runtime.gather(local).reshape(runtime.world_size, -1).cpu()
    partitions = [row.tolist() for row in gathered]
    flattened = [sample_id for partition in partitions for sample_id in partition]
    if sorted(flattened) != list(range(total)) or len(set(flattened)) != total:
        raise AssertionError(f"sample partitions are not disjoint and complete: {partitions}")
    return partitions


def _run_train_checkpoint() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the required Accelerate smoke must run on CUDA")
    runtime = AccelerateRuntime.create(SimpleNamespace(accumulation=_ACCUMULATION))
    torch.cuda.reset_peak_memory_stats(runtime.device)
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)

    model = _TinyLoRAModel().to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LEARNING_RATE, weight_decay=0.0)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_optimizer = optimizer.state_dict()
    initial_cpu_rng = torch.get_rng_state().clone()
    initial_cuda_rng = torch.cuda.get_rng_state(runtime.device).clone()

    probe_step = _SyntheticLocalProbeStep(model, optimizer, rank=runtime.rank)

    def probe_sample(candidate: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.full((candidate, 1), 0.25, dtype=torch.float32, device=runtime.device)
        targets = torch.full((candidate, 1), 0.0625, dtype=torch.float32, device=runtime.device)
        return inputs, targets

    probe = probe_microbatch(runtime, [1, _MICROBATCH], probe_sample, probe_step)
    if probe.microbatch != _MICROBATCH or optimizer.state_dict() != initial_optimizer:
        raise AssertionError("probe selection or optimizer restoration failed")
    if not torch.equal(torch.get_rng_state(), initial_cpu_rng):
        raise AssertionError("probe did not restore CPU RNG")
    if not torch.equal(torch.cuda.get_rng_state(runtime.device), initial_cuda_rng):
        raise AssertionError("probe did not restore CUDA RNG")
    for name, expected in initial_state.items():
        if not torch.equal(model.state_dict()[name], expected):
            raise AssertionError(f"probe did not restore model state {name}")

    model, optimizer = runtime.prepare(model, optimizer)
    target = runtime.unwrap(model)
    total_samples = runtime.world_size * _MICROBATCH * _ACCUMULATION
    dataset = _SyntheticSamples(total_samples)
    loader = DataLoader(dataset, batch_size=_MICROBATCH, shuffle=False, drop_last=True)
    loader = runtime.prepare(loader)
    local_ids: list[int] = []
    sync_pattern: list[bool] = []
    synchronized_gradients: torch.Tensor | None = None
    for batch in loader:
        with runtime.accumulate(model):
            local_ids.extend(int(value) for value in batch["id"].tolist())
            outputs = model(batch["input"])
            loss = torch.nn.functional.mse_loss(outputs, batch["target"])
            runtime.backward(loss)
            sync_pattern.append(runtime.sync_gradients)
            if runtime.sync_gradients:
                synchronized_gradients = _lora_gradients(target).clone()
                runtime.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

    if sync_pattern != [False, True] or synchronized_gradients is None:
        raise AssertionError(f"unexpected accumulation pattern on rank {runtime.rank}: {sync_pattern}")
    target.next_sample_position.fill_(total_samples)
    partitions = _assert_disjoint_ids(runtime, local_ids, total_samples)
    gathered_patterns = runtime.gather(torch.tensor(sync_pattern, dtype=torch.int8, device=runtime.device)).reshape(
        runtime.world_size, -1
    )
    if not torch.equal(gathered_patterns, torch.tensor([[0, 1]] * runtime.world_size, device=runtime.device)):
        raise AssertionError(f"ranks disagree on accumulation: {gathered_patterns.tolist()}")

    gathered_gradients = runtime.gather(synchronized_gradients).reshape(runtime.world_size, -1)
    if not torch.allclose(
        gathered_gradients, gathered_gradients[0].expand_as(gathered_gradients), atol=2e-5, rtol=2e-4
    ):
        raise AssertionError(f"LoRA gradients differ across ranks: {gathered_gradients.tolist()}")
    reference_gradients, reference_parameters = _make_reference(initial_state, dataset, runtime.device)
    if not torch.allclose(synchronized_gradients, reference_gradients, atol=2e-5, rtol=2e-3):
        raise AssertionError(
            f"distributed LoRA gradient differs from reference: {synchronized_gradients.tolist()} != "
            f"{reference_gradients.tolist()}"
        )
    if not torch.allclose(_lora_values(target), reference_parameters, atol=2e-5, rtol=2e-3):
        raise AssertionError("distributed optimizer update differs from the single-batch reference")
    if _optimizer_step(optimizer) != 1:
        raise AssertionError("gradient accumulation produced more than one optimizer update")

    checkpoint = _checkpoint_path(runtime)
    runtime.save(checkpoint)
    saved_parameters = _lora_values(target).clone()
    saved_step = _optimizer_step(optimizer)
    saved_position = int(target.next_sample_position.item())
    with torch.no_grad():
        for name, parameter in target.named_parameters():
            if "lora_" in name:
                parameter.add_(9.0)
        target.next_sample_position.zero_()
        for state in optimizer.state.values():
            if "step" in state:
                state["step"].add_(7)
    torch.rand(11)
    torch.rand(11, device=runtime.device)
    runtime.load(checkpoint)

    restored = (
        torch.equal(_lora_values(target), saved_parameters)
        and _optimizer_step(optimizer) == saved_step
        and int(target.next_sample_position.item()) == saved_position
    )
    if not restored:
        raise AssertionError("checkpoint did not restore parameters, optimizer step, and next sample position")
    runtime.barrier()

    peak_mib = torch.cuda.max_memory_allocated(runtime.device) / (1024 * 1024)
    evidence = {
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "sample_ids": local_ids,
        "sync_pattern": sync_pattern,
        "lora_gradient": synchronized_gradients.tolist(),
        "optimizer_step": saved_step,
        "next_sample_position": saved_position,
        "restored": restored,
        "peak_memory_mib": round(peak_mib, 3),
    }
    print("RANK_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    if runtime.rank == 0:
        summary = {
            "status": "PASS",
            "runtime": "accelerate",
            "accelerate": accelerate.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(runtime.device),
            "world_size": runtime.world_size,
            "partitions": partitions,
            "probe_microbatch": probe.microbatch,
            "checkpoint": str(checkpoint),
        }
        print("SMOKE_RESULT " + json.dumps(summary, sort_keys=True), flush=True)


def _run_probe_rank_fault() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the required rank-fault smoke must run on CUDA")
    runtime = AccelerateRuntime.create(SimpleNamespace(accumulation=_ACCUMULATION))
    if runtime.world_size < 2:
        raise RuntimeError("probe-rank-fault requires multiple processes")
    torch.manual_seed(811)
    torch.cuda.manual_seed(811)

    model = _TinyLoRAModel().to(runtime.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=_LEARNING_RATE, weight_decay=0.0)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    initial_optimizer = optimizer.state_dict()
    fault_rank = runtime.world_size - 1
    step = _SyntheticLocalProbeStep(
        model,
        optimizer,
        rank=runtime.rank,
        fault_rank=fault_rank,
        fault_candidate=2,
    )

    def probe_sample(candidate: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.full((candidate, 1), 0.25, dtype=torch.float32, device=runtime.device)
        targets = torch.full((candidate, 1), 0.0625, dtype=torch.float32, device=runtime.device)
        return inputs, targets

    probe = probe_microbatch(runtime, [1, 2], probe_sample, step)
    if probe.microbatch != 1:
        raise AssertionError(f"rank-local OOM should select previous candidate 1, got {probe.microbatch}")
    if optimizer.state_dict() != initial_optimizer:
        raise AssertionError("rank-fault probe retained optimizer state")
    for name, expected in initial_state.items():
        if not torch.equal(model.state_dict()[name], expected):
            raise AssertionError(f"rank-fault probe retained model state {name}")
    injected = runtime.gather(torch.tensor([int(step.injected_oom)], dtype=torch.int8, device=runtime.device)).reshape(
        -1
    )
    if injected.tolist() != [0] * fault_rank + [1]:
        raise AssertionError(f"OOM injection did not occur on exactly rank {fault_rank}: {injected.tolist()}")

    model, optimizer = runtime.prepare(model, optimizer)
    target = runtime.unwrap(model)
    total_samples = runtime.world_size * probe.microbatch * _ACCUMULATION
    loader = DataLoader(_SyntheticSamples(total_samples), batch_size=probe.microbatch, shuffle=False, drop_last=True)
    loader = runtime.prepare(loader)
    sync_pattern: list[bool] = []
    for batch in loader:
        with runtime.accumulate(model):
            loss = torch.nn.functional.mse_loss(model(batch["input"]), batch["target"])
            runtime.backward(loss)
            sync_pattern.append(runtime.sync_gradients)
            if runtime.sync_gradients:
                runtime.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()

    if sync_pattern != [False, True]:
        raise AssertionError(f"probe advanced accumulation cursor on rank {runtime.rank}: {sync_pattern}")
    gathered_patterns = runtime.gather(torch.tensor(sync_pattern, dtype=torch.int8, device=runtime.device)).reshape(
        runtime.world_size, -1
    )
    if not torch.equal(gathered_patterns, torch.tensor([[0, 1]] * runtime.world_size, device=runtime.device)):
        raise AssertionError(f"post-probe accumulation differs across ranks: {gathered_patterns.tolist()}")
    if _optimizer_step(optimizer) != 1:
        raise AssertionError("post-probe normal training did not take exactly one optimizer step")

    evidence = {
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "fault_rank": fault_rank,
        "injected_oom": step.injected_oom,
        "selected_microbatch": probe.microbatch,
        "sync_pattern": sync_pattern,
        "optimizer_step": _optimizer_step(optimizer),
        "lora_values": _lora_values(target).tolist(),
    }
    print("RANK_FAULT_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    if runtime.rank == 0:
        print(
            "RANK_FAULT_RESULT "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": runtime.world_size,
                    "fault_rank": fault_rank,
                    "selected_microbatch": probe.microbatch,
                    "sync_pattern": sync_pattern,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    if args.mode == "train-checkpoint":
        _run_train_checkpoint()
    elif args.mode == "probe-rank-fault":
        _run_probe_rank_fault()


if __name__ == "__main__":
    main()
