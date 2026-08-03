from __future__ import annotations

from contextlib import contextmanager

import torch

from voxcpm.training.balalaika.generation import generation_autocast


def test_generation_autocast_enters_cuda_bf16_and_is_a_cpu_noop(monkeypatch) -> None:
    """Catches low-level CUDA generation losing BF16 autocast or enabling it on CPU."""
    active: list[tuple[str, torch.dtype]] = []

    @contextmanager
    def fake_autocast(*, device_type: str, dtype: torch.dtype):
        active.append((device_type, dtype))
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(torch, "autocast", fake_autocast)

    with generation_autocast(torch.device("cuda", 0)):
        assert active == [("cuda", torch.bfloat16)]
    with generation_autocast(torch.device("cpu")):
        assert active == []

    assert active == []
