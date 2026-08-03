#!/usr/bin/env python3
"""Real tiny-device smokes for Balalaika Accelerate semantics; never loads VoxCPM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import wave

import accelerate
import torch
from torch.utils.data import DataLoader, Dataset

from voxcpm.training.balalaika.probe import probe_microbatch
from voxcpm.training.balalaika.runtime import AccelerateRuntime
from voxcpm.training.balalaika.artifacts import atomic_json, sha256_file
from voxcpm.training.balalaika.evaluation import DistributedEvaluator
from voxcpm.training.balalaika.ledger import ValidationLedger
from voxcpm.training.balalaika.metrics import BenchmarkRow

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


class _SyntheticValidationModel(torch.nn.Module):
    """Fake VoxCPM surface that records exact evaluator assignments."""

    sample_rate = 16_000

    def __init__(self):
        super().__init__()
        self.audio_vae = None
        self.item_ids: list[int] = []

    def generate(self, **kwargs: object) -> torch.Tensor:
        if self.audio_vae is None:
            raise AssertionError("validation evaluator did not attach the retained AudioVAE")
        item_id = int(str(kwargs["target_text"]).split()[-1])
        if kwargs["prompt_text"] != "синтетический промпт":
            raise AssertionError("validation evaluator changed the fixed prompt text")
        self.item_ids.append(item_id)
        return torch.full((80,), (item_id + 1) / 1_000.0, dtype=torch.float32)


class _SyntheticValidationASR:
    def __init__(self, close_events: list[str]):
        self.close_events = close_events

    def transcribe(self, wav_path: Path) -> str:
        item_id = int(Path(wav_path).stem)
        return "" if item_id == 0 else f"номер {item_id}"

    def close(self) -> None:
        self.close_events.append("closed")


class _RankZeroValidationTracking:
    """Durable local tracking fake; importing or contacting W&B is unnecessary."""

    def __init__(self, root: Path, rank: int):
        if rank != 0:
            raise AssertionError("non-main rank initialized validation tracking")
        self.root = root
        atomic_json(root / "tracking-init.json", {"init_ranks": [rank]})

    def log_validation(self, payload: object, *, global_step: int) -> None:
        path = self.root / "tracking-log.json"
        if path.exists():
            raise AssertionError("validation aggregate was tracked more than once")
        atomic_json(
            path,
            {
                "log_count": 1,
                "global_step": global_step,
                "item_count": payload.metrics.item_count,
                "audio_ids": list(payload.audio_paths),
            },
        )


class _WorkerValidationTracking:
    def log_validation(self, payload: object, *, global_step: int) -> None:
        raise AssertionError("non-main rank attempted validation tracking")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train-checkpoint", "probe-rank-fault", "validation", "trainer-regressions"),
        required=True,
    )
    parser.add_argument("--runtime", choices=("accelerate",), default="accelerate")
    parser.add_argument("--items", type=int, default=32)
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
    runtime.barrier()
    runtime.close()


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


def _run_trainer_regressions() -> None:
    """Exercise scheduler ownership and deterministic overflow replay on eight real ranks."""
    runtime = AccelerateRuntime.create(SimpleNamespace(accumulation=_ACCUMULATION, cpu=True))
    if runtime.world_size != 8:
        raise RuntimeError(f"trainer-regressions smoke requires exactly eight processes, found {runtime.world_size}")
    torch.manual_seed(1907)

    real_step_target = 3
    total_samples = runtime.world_size * _ACCUMULATION * real_step_target
    model = _TinyLoRAModel().to(runtime.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)
    initial_scheduler_epoch = scheduler.last_epoch
    loader = DataLoader(
        _SyntheticSamples(total_samples),
        batch_size=1,
        shuffle=True,
        drop_last=True,
    )

    prepare_calls = 0
    prepare_calls += 1
    model, optimizer, loader, scheduler = runtime.prepare(model, optimizer, loader, scheduler)

    sampler_epoch = 17
    real_steps = 0
    sync_attempts = 0
    passes: list[list[int]] = []
    while real_steps < real_step_target:
        loader.set_epoch(sampler_epoch)
        local_ids: list[int] = []
        for batch in loader:
            local_ids.extend(int(value) for value in batch["id"].tolist())
            with runtime.accumulate(model):
                loss = torch.nn.functional.mse_loss(model(batch["input"]), batch["target"])
                runtime.backward(loss)
                if not runtime.sync_gradients:
                    continue
                sync_attempts += 1
                if sync_attempts == real_step_target:
                    # Synthetic AMP overflow: consume the complete group but
                    # do not advance optimizer, scheduler, or durable progress.
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                real_steps += 1
                if real_steps == real_step_target:
                    break
        passes.append(local_ids)

    if len(passes) != 2 or passes[1] != passes[0][: len(passes[1])]:
        raise AssertionError(f"same-epoch replay changed sample order on rank {runtime.rank}: {passes}")

    loader.set_epoch(sampler_epoch + 1)
    next_epoch_ids = [int(value) for batch in loader for value in batch["id"].tolist()]
    next_epoch_changed = next_epoch_ids != passes[0]
    underlying_scheduler = getattr(scheduler, "scheduler", scheduler)
    scheduler_steps = int(underlying_scheduler.last_epoch - initial_scheduler_epoch)
    local_evidence = torch.tensor(
        [
            real_steps,
            sync_attempts,
            scheduler_steps,
            int(passes[1] == passes[0][: len(passes[1])]),
            int(next_epoch_changed),
            prepare_calls,
        ],
        dtype=torch.int64,
        device=runtime.device,
    )
    gathered = runtime.gather(local_evidence).reshape(runtime.world_size, -1).cpu()
    expected = torch.tensor(
        [real_step_target, real_step_target + 1, real_step_target, 1, 1, 1],
        dtype=torch.int64,
    )
    if not torch.equal(gathered, expected.expand_as(gathered)):
        raise AssertionError(f"trainer regression evidence differs across ranks: {gathered.tolist()}")
    print(
        "TRAINER_REGRESSION_RANK_EVIDENCE "
        + json.dumps(
            {
                "rank": runtime.rank,
                "world_size": runtime.world_size,
                "first_epoch_ids": passes[0],
                "replay_ids": passes[1],
                "next_epoch_ids": next_epoch_ids,
                "real_optimizer_steps": real_steps,
                "sync_attempts": sync_attempts,
                "scheduler_steps": scheduler_steps,
                "prepare_calls": prepare_calls,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if runtime.rank == 0:
        print(
            "TRAINER_REGRESSION_RESULT "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": runtime.world_size,
                    "real_optimizer_steps": real_step_target,
                    "scheduler_steps": real_step_target,
                    "same_epoch_replay": True,
                    "next_epoch_increment": 1,
                    "prepare_calls": prepare_calls,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _write_validation_prompt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 80)


def _validation_root(runtime: AccelerateRuntime) -> Path:
    nonce = time.time_ns() if runtime.rank == 0 else 0
    gathered = runtime.gather(torch.tensor([nonce], dtype=torch.int64, device=runtime.device))
    shared_nonce = int(gathered.reshape(-1)[0].item())
    return (
        Path(__file__).resolve().parents[1]
        / ".superpowers"
        / "sdd"
        / "2026-08-02-balalaika-two-stage-lora-training"
        / "task-11-artifacts"
        / f"validation-{runtime.world_size}x-{shared_nonce}"
    )


def _smoke_payload(**values: object) -> SimpleNamespace:
    return SimpleNamespace(**values)


def _run_validation(items: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the required validation smoke must run on CUDA")
    runtime = AccelerateRuntime.create(SimpleNamespace(accumulation=1))
    if items != 32:
        raise ValueError("validation smoke requires exactly --items 32")
    if runtime.world_size != 8:
        raise RuntimeError("validation smoke requires exactly eight processes")
    root = _validation_root(runtime)
    if runtime.rank == 0:
        root.mkdir(parents=True, exist_ok=False)
        _write_validation_prompt(root / "selection" / "prompt.wav")
    runtime.barrier()

    prompt_path = root / "selection" / "prompt.wav"
    prompt = SimpleNamespace(
        prompt_id="prompt-00",
        text="синтетический промпт",
        wav_path=prompt_path,
        wav_sha256=sha256_file(prompt_path),
    )
    rows = tuple(
        BenchmarkRow(
            id=item_id,
            category="number" if item_id % 2 == 0 else "date",
            text=f"номер {item_id}",
            normalized_gold=f"номер {item_id}",
            stressed=f"но́мер {item_id}",
        )
        for item_id in range(items)
    )
    selection = SimpleNamespace(
        fingerprint="synthetic-selection-fingerprint",
        prompts=[prompt],
        benchmark_prompt_by_id={row.id: prompt.prompt_id for row in rows},
        audio_log_ids=[0, 7, 16, 31],
    )
    ledger = ValidationLedger(
        root / "boundary-01",
        generation_fingerprint="synthetic-generation-fingerprint",
        asr_fingerprint="synthetic-asr-fingerprint",
        max_attempts=2,
    )
    tracking = _RankZeroValidationTracking(root, runtime.rank) if runtime.rank == 0 else _WorkerValidationTracking()
    asr_close_events: list[str] = []
    evaluator = DistributedEvaluator(
        runtime=runtime,
        rows=rows,
        selection=selection,
        ledger=ledger,
        asr_factory=lambda _device_id: _SyntheticValidationASR(asr_close_events),
        run_manager=tracking,
        expected_item_count=items,
        validation_root=root,
        payload_factory=_smoke_payload,
    )
    model = _SyntheticValidationModel()
    audio_vae = torch.nn.Identity()
    payload = evaluator.run(
        model,
        audio_vae,
        {"checkpoint_fingerprint": "synthetic-checkpoint-fingerprint"},
        {
            "stage": "stage1",
            "epoch": 0,
            "boundary": 1,
            "global_step": 1,
            "stage_progress": 0.125,
        },
    )
    if len(model.item_ids) != 4 or len(set(model.item_ids)) != 4:
        raise AssertionError(f"rank {runtime.rank} did not generate exactly four unique IDs: {model.item_ids}")
    if asr_close_events != ["closed"]:
        raise AssertionError(f"rank {runtime.rank} did not release its ASR session: {asr_close_events}")
    if payload.metrics.item_count != items:
        raise AssertionError(f"rank {runtime.rank} did not verify the complete aggregate payload")

    local = torch.tensor(model.item_ids, dtype=torch.int64, device=runtime.device)
    gathered = runtime.gather(local).reshape(runtime.world_size, -1).cpu()
    partitions = [row.tolist() for row in gathered]
    flattened = [item_id for partition in partitions for item_id in partition]
    if sorted(flattened) != list(range(items)) or len(set(flattened)) != items:
        raise AssertionError(f"validation partitions are not disjoint and complete: {partitions}")
    evidence = {
        "rank": runtime.rank,
        "world_size": runtime.world_size,
        "item_ids": model.item_ids,
        "generation_count": len(model.item_ids),
        "asr_released": asr_close_events == ["closed"],
    }
    print("VALIDATION_RANK_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)
    runtime.barrier()
    if runtime.rank == 0:
        tracking_init = json.loads((root / "tracking-init.json").read_text(encoding="utf-8"))
        tracking_log = json.loads((root / "tracking-log.json").read_text(encoding="utf-8"))
        aggregate_files = list(root.rglob("metrics.json"))
        completion_files = list(root.rglob("validation-complete.json"))
        if len(aggregate_files) != 1 or len(completion_files) != 1 or tracking_log["log_count"] != 1:
            raise AssertionError("validation smoke did not publish exactly one rank-zero aggregate")
        print(
            "VALIDATION_SMOKE_RESULT "
            + json.dumps(
                {
                    "status": "PASS",
                    "world_size": runtime.world_size,
                    "items": items,
                    "partitions": partitions,
                    "aggregate_files": len(aggregate_files),
                    "completion_files": len(completion_files),
                    "tracking_init_ranks": tracking_init["init_ranks"],
                    "tracking_log_count": tracking_log["log_count"],
                    "artifact_root": str(root),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    runtime.barrier()
    runtime.close()


def main() -> None:
    args = _parse_args()
    if args.mode == "train-checkpoint":
        _run_train_checkpoint()
    elif args.mode == "probe-rank-fault":
        _run_probe_rank_fault()
    elif args.mode == "validation":
        _run_validation(args.items)
    elif args.mode == "trainer-regressions":
        _run_trainer_regressions()


if __name__ == "__main__":
    main()
