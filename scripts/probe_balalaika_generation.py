#!/usr/bin/env python3
"""Generate one WAV from a verified Balalaika memorization recovery adapter."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import soundfile as sf
import torch

from voxcpm.training.balalaika.artifacts import fingerprint
from voxcpm.training.balalaika.asr import GigaAMRNNT
from voxcpm.training.balalaika.checkpoint import CheckpointManager
from voxcpm.training.balalaika.config import BalalaikaConfig
from voxcpm.training.balalaika.generation import generation_autocast
from voxcpm.training.balalaika.memorization import _load_selection
from voxcpm.training.balalaika.trainer import build_model


class GenerationProbeError(RuntimeError):
    """The production generation probe cannot run without mutating protected state."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="probe-balalaika-generation")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gigaam",
        action="store_true",
        help=(
            "Run the pinned GigaAM diagnostic after successful generation; "
            "diagnostic failure is reported but non-gating."
        ),
    )
    return parser


def run_probe(
    config: Any,
    checkpoint: Path,
    output: Path,
    *,
    gigaam: bool,
) -> dict[str, object]:
    """Load one verified recovery adapter and write exactly one diagnostic WAV."""
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise GenerationProbeError(f"memorization recovery checkpoint does not exist: {checkpoint}")
    output = Path(output).resolve()
    _require_safe_output(config, checkpoint, output)
    device = _require_one_cuda_device()

    _selection_path, selection_fingerprint, samples = _load_selection(config)
    sample = min(samples, key=lambda item: (len(item.text), item.source_relative_path))
    model, audio_vae, _tokenizer = build_model(config, 1, adapter_checkpoint=None)
    manager = CheckpointManager(checkpoint.parent)
    metadata = manager.load_verified_adapter(
        model,
        checkpoint,
        expected={
            "checkpoint_kind": "recovery",
            "stage": "stage1",
            "optimizer_step": config.memorization.updates,
            "base_revision": getattr(model, "balalaika_base_revision", None),
            "selection_fingerprint": selection_fingerprint,
            "lora_fingerprint": fingerprint(config.lora.model_dump(mode="json")),
        },
    )

    model.to(device)
    audio_vae.to(device=device, dtype=torch.float32)
    model_training = model.training
    vae_training = audio_vae.training
    had_audio_vae = hasattr(model, "audio_vae")
    previous_audio_vae = getattr(model, "audio_vae", None)
    try:
        model.eval()
        audio_vae.eval()
        setattr(model, "audio_vae", audio_vae)
        with torch.no_grad():
            with generation_autocast(device):
                generated = model.generate(
                    target_text=sample.text,
                    prompt_text="",
                    prompt_wav_path="",
                    seed=config.memorization.seed,
                    cfg_value=config.generation.cfg_value,
                    inference_timesteps=config.generation.inference_timesteps,
                    max_len=config.generation.max_length,
                )
    finally:
        if had_audio_vae:
            setattr(model, "audio_vae", previous_audio_vae)
        elif hasattr(model, "audio_vae"):
            delattr(model, "audio_vae")
        model.train(model_training)
        audio_vae.train(vae_training)

    audio = _audio_array(generated)
    sample_rate = _sample_rate(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), audio, sample_rate, format="WAV")
    diagnostic = _gigaam_diagnostic(config, output, device) if gigaam else None
    return {
        "status": "complete",
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
        "sample_id": sample.source_relative_path,
        "text": sample.text,
        "output": str(output),
        "sample_rate": sample_rate,
        "samples": int(audio.size),
        "gigaam_diagnostic": diagnostic,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[..., dict[str, object]] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = BalalaikaConfig.load(args.config)
        result = (runner or run_probe)(
            config,
            args.checkpoint,
            args.output,
            gigaam=args.gigaam,
        )
    except (GenerationProbeError, OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


def _require_safe_output(config: Any, checkpoint: Path, output: Path) -> None:
    if output.suffix.casefold() != ".wav":
        raise GenerationProbeError("probe output must be an explicit .wav path")
    if output.exists():
        raise GenerationProbeError(f"probe output already exists: {output}")
    protected = {
        "checkpoint": checkpoint,
        "corpus": Path(config.data.corpus_root).resolve(),
        "index": Path(config.data.index_dir).resolve(),
        "selection": Path(config.selection_dir).resolve(),
    }
    for label, root in protected.items():
        if output == root or root in output.parents:
            raise GenerationProbeError(f"probe output is inside protected {label} state: {output}")


def _require_one_cuda_device() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise GenerationProbeError("probe requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def _audio_array(generated: object) -> np.ndarray:
    value = generated[0] if isinstance(generated, tuple) and generated else generated
    if isinstance(value, torch.Tensor):
        audio = value.detach().float().cpu().numpy()
    else:
        audio = np.asarray(value, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise GenerationProbeError("VoxCPM2 generation returned empty or non-finite audio")
    return audio


def _sample_rate(model: Any) -> int:
    value = getattr(model, "sample_rate", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationProbeError("VoxCPM2 model has no valid output sample rate")
    return value


def _gigaam_diagnostic(config: Any, output: Path, device: torch.device) -> str:
    try:
        pins = json.loads((Path(config.hub.local_dir) / "hub-pins.json").read_text(encoding="utf-8"))
        pin = pins.get("gigaam") if isinstance(pins, Mapping) else None
        if not isinstance(pin, Mapping):
            raise GenerationProbeError("Hub pins have no GigaAM model")
        model_dir = Path(str(pin.get("local_dir", "")))
        device_id = device.index if device.index is not None else 0
        with GigaAMRNNT(model_dir, device_id, fingerprint(pin)) as asr:
            return asr.transcribe(output)
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"


if __name__ == "__main__":
    raise SystemExit(main())
