"""Shared inference context for low-level VoxCPM2 generation."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch


def generation_autocast(device: torch.device | str) -> AbstractContextManager[None]:
    """Use the BF16 context required by unwrapped CUDA generation."""
    if torch.device(device).type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()
