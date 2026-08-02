"""Pinned, GPU-only GigaAM v3 RNN-T transcription adapter."""

from __future__ import annotations

import gc
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .artifacts import fingerprint, sha256_file

_MODEL_NAME = "gigaam-v3-rnnt"
_TARGET_SAMPLE_RATE = 16_000


class GigaAMProviderError(RuntimeError):
    """Raised when the requested CUDA execution provider cannot be used."""


class GigaAMRNNT:
    """Lazily load one locally pinned GigaAM v3 RNN-T session on one CUDA GPU.

    ``model_dir`` must be the ``gigaam`` directory described by the adjacent
    Hub pin manifest.  Passing that directory as ``path=`` keeps
    :func:`onnx_asr.load_model` from resolving or downloading a floating model
    revision at validation time.
    """

    def __init__(self, model_dir: Path, device_id: int, model_fingerprint: str):
        self.model_dir = Path(model_dir)
        self.device_id = int(device_id)
        self.model_fingerprint = str(model_fingerprint)
        self._session: Any | None = None

    def __enter__(self) -> "GigaAMRNNT":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def transcribe(self, wav_path: Path) -> str:
        """Transcribe a WAV after converting it to mono float32 16 kHz audio."""
        waveform = _load_wav(Path(wav_path))
        result = self._load_session().recognize(waveform, sample_rate=_TARGET_SAMPLE_RATE)
        return _normalize_result(result)

    def close(self) -> None:
        """Release the loaded adapter/session before validation returns to training."""
        session, self._session = self._session, None
        if session is None:
            return
        _close_session(session)
        del session
        gc.collect()

    def _load_session(self) -> Any:
        if self._session is not None:
            return self._session

        self._validate_pinned_model()
        onnxruntime = _import_dependency("onnxruntime")
        providers = _cuda_providers(onnxruntime, self.device_id)
        onnx_asr = _import_dependency("onnx_asr")
        try:
            self._session = onnx_asr.load_model(_MODEL_NAME, path=self.model_dir, providers=providers)
        except BaseException as error:
            raise GigaAMProviderError(
                f"Could not load pinned {_MODEL_NAME} with CUDAExecutionProvider on device {self.device_id}."
            ) from error
        return self._session

    def _validate_pinned_model(self) -> None:
        model_dir = self.model_dir.resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Pinned GigaAM model directory does not exist: {model_dir}")
        manifest_path = model_dir.parent / "hub-pins.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pin = manifest["gigaam"]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Missing valid GigaAM Hub pin manifest at {manifest_path}.") from error
        if not isinstance(pin, dict):
            raise ValueError("GigaAM Hub pin must be a JSON object.")
        if pin.get("kind") != "model" or not isinstance(pin.get("revision"), str) or not pin["revision"]:
            raise ValueError("GigaAM Hub pin must identify an immutable model revision.")
        declared_dir = Path(str(pin.get("local_dir", "")))
        if not declared_dir.is_absolute():
            declared_dir = manifest_path.parent / declared_dir
        if declared_dir.resolve() != model_dir:
            raise ValueError("GigaAM Hub pin local directory does not match the requested model directory.")
        if fingerprint(pin) != self.model_fingerprint:
            raise ValueError("Pinned GigaAM model fingerprint does not match the Hub pin manifest.")
        files = pin.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("GigaAM Hub pin must include hashes for local model files.")
        for relative_path, expected_hash in files.items():
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                raise ValueError("GigaAM Hub pin contains an invalid model file hash.")
            candidate = (model_dir / relative_path).resolve()
            if model_dir not in candidate.parents or not candidate.is_file():
                raise ValueError(f"Pinned GigaAM model file is missing: {relative_path}")
            if sha256_file(candidate) != expected_hash:
                raise ValueError(f"pinned GigaAM file hash does not match: {relative_path}")


def _import_dependency(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"{name} is required for pinned GigaAM validation.") from error


def _cuda_providers(onnxruntime: Any, device_id: int) -> list[tuple[str, dict[str, int]] | str]:
    available = getattr(onnxruntime, "get_available_providers", None)
    if not callable(available):
        raise GigaAMProviderError("onnxruntime cannot report available execution providers.")
    if "CUDAExecutionProvider" not in set(available()):
        raise GigaAMProviderError(
            f"CUDAExecutionProvider is unavailable for requested GigaAM device {device_id}; refusing CPU-only ASR."
        )
    # CPU is intentionally a secondary provider only for individual unsupported
    # graph nodes.  CUDA remains the required first provider for the session.
    return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]


def _load_wav(wav_path: Path) -> np.ndarray:
    if not wav_path.is_file():
        raise FileNotFoundError(f"GigaAM input WAV does not exist: {wav_path}")
    try:
        waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Could not read GigaAM input WAV {wav_path}.") from error
    if sample_rate <= 0 or waveform.size == 0:
        raise ValueError(f"GigaAM input WAV {wav_path} has no audio samples.")
    mono = np.asarray(waveform.mean(axis=1), dtype=np.float32)
    if sample_rate == _TARGET_SAMPLE_RATE:
        return mono
    output_samples = max(1, round(len(mono) * _TARGET_SAMPLE_RATE / sample_rate))
    source_positions = np.arange(len(mono), dtype=np.float64)
    target_positions = np.linspace(0, len(mono) - 1, output_samples, dtype=np.float64)
    return np.interp(target_positions, source_positions, mono).astype(np.float32, copy=False)


def _normalize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and len(result) == 1 and isinstance(result[0], str):
        return result[0]
    raise TypeError(f"Pinned GigaAM returned an invalid single-item result: {type(result).__name__}.")


def _close_session(session: Any) -> None:
    """Close known adapter/session shapes without relying on garbage collection."""
    seen: set[int] = set()
    candidates = [
        session,
        getattr(session, "session", None),
        getattr(session, "asr", None),
        getattr(session, "resampler", None),
    ]
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        close = getattr(candidate, "close", None)
        if callable(close):
            close()
            continue
        release = getattr(candidate, "release", None)
        if callable(release):
            release()
