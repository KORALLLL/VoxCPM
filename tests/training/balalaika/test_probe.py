import copy

import pytest
import torch

from voxcpm.training.balalaika.probe import ProbeError, probe_microbatch


class FakeRuntime:
    def __init__(self, rank_results=None):
        self.rank_results = rank_results or {}
        self.current_candidate = None
        self.device = torch.device("cpu")
        self.barriers = 0

    def unwrap(self, model):
        return model

    def gather(self, local_status):
        configured = self.rank_results.get(self.current_candidate)
        if configured is None:
            return local_status
        return torch.tensor(configured, dtype=local_status.dtype)

    def barrier(self):
        self.barriers += 1


class TinyProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([4.0]), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))
        self.lora_B = torch.nn.Parameter(torch.tensor([2.0]))

    def forward(self, value):
        return (self.lora_A * self.lora_B * value + self.frozen).sum()


class ProbeStep:
    def __init__(self, runtime, *, fail_with=None):
        self.model = TinyProbeModel()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        self.runtime = runtime
        self.fail_with = fail_with or {}
        self.calls = []

    def __call__(self, runtime, sample):
        candidate = int(sample.numel())
        runtime.current_candidate = candidate
        self.calls.append(candidate)
        torch.rand(5)
        loss = self.model(sample)
        loss.backward()
        self.optimizer.step()
        if candidate in self.fail_with:
            raise self.fail_with[candidate]


def samples(candidate):
    return torch.ones(candidate)


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_probe_uses_largest_success_shared_by_every_rank():
    runtime = FakeRuntime({1: [1, 1, 1], 2: [1, 1, 1], 4: [1, 0, 1]})
    step = ProbeStep(runtime)

    result = probe_microbatch(runtime, [4, 1, 2], samples, step)

    assert result.microbatch == 2
    assert result.all_rank_success is True
    assert result.attempted == (1, 2, 4)
    assert step.calls == [1, 2, 4]
    assert runtime.barriers == 3


def test_probe_restores_lora_weights_gradients_optimizer_and_cpu_rng_after_every_candidate():
    runtime = FakeRuntime()
    step = ProbeStep(runtime, fail_with={2: torch.cuda.OutOfMemoryError("synthetic OOM")})
    step.model.lora_A.grad = torch.tensor([17.0])
    original_lora = {
        name: parameter.detach().clone() for name, parameter in step.model.named_parameters() if "lora_" in name
    }
    original_gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in step.model.named_parameters()
    }
    original_optimizer = copy.deepcopy(step.optimizer.state_dict())
    original_rng = torch.get_rng_state().clone()

    result = probe_microbatch(runtime, [1, 2], samples, step)

    assert result.microbatch == 1
    for name, parameter in step.model.named_parameters():
        if name in original_lora:
            assert torch.equal(parameter, original_lora[name])
        expected_grad = original_gradients[name]
        if expected_grad is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, expected_grad)
    _assert_nested_equal(step.optimizer.state_dict(), original_optimizer)
    assert torch.equal(torch.get_rng_state(), original_rng)


def test_probe_restores_state_then_reraises_non_oom_exception():
    runtime = FakeRuntime()
    step = ProbeStep(runtime, fail_with={1: ValueError("bad synthetic sample")})
    original_lora = step.model.lora_A.detach().clone()
    original_rng = torch.get_rng_state().clone()

    with pytest.raises(ValueError, match="bad synthetic sample"):
        probe_microbatch(runtime, [1, 2], samples, step)

    assert step.calls == [1]
    assert torch.equal(step.model.lora_A, original_lora)
    assert torch.equal(torch.get_rng_state(), original_rng)


def test_explicit_microbatch_bypasses_probe_work_and_validation():
    runtime = FakeRuntime()
    calls = []

    result = probe_microbatch(
        runtime,
        [1, 2, 4],
        lambda candidate: calls.append(candidate),
        object(),
        explicit_microbatch=3,
    )

    assert result.microbatch == 3
    assert result.all_rank_success is True
    assert result.attempted == ()
    assert result.bypassed is True
    assert calls == []


def test_probe_fails_when_no_candidate_succeeds_on_every_rank():
    runtime = FakeRuntime({1: [0, 0], 2: [1, 0]})
    step = ProbeStep(runtime, fail_with={1: torch.cuda.OutOfMemoryError("local OOM")})

    with pytest.raises(ProbeError, match="no microbatch candidate"):
        probe_microbatch(runtime, [1, 2], samples, step)
