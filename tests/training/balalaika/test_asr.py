"""Pinned, GPU-only GigaAM v3 RNN-T adapter behavior."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import pytest

from voxcpm.training.balalaika.artifacts import fingerprint, sha256_file
from voxcpm.training.balalaika.asr import GigaAMRNNT, GigaAMProviderError


class FakeSession:
    def __init__(self, result: str | list[str] = "распознано"):
        self.result = result
        self.calls: list[tuple[object, int]] = []
        self.closed = 0

    def recognize(self, waveform, *, sample_rate: int):
        self.calls.append((waveform, sample_rate))
        return self.result

    def close(self) -> None:
        self.closed += 1


class FakeOnnxAsr:
    def __init__(self):
        self.providers = None
        self.calls: list[tuple[str, Path | None]] = []
        self.session = FakeSession()

    def load_model(self, model: str, path: Path | None = None, *, providers=None):
        self.calls.append((model, path))
        self.providers = providers
        return self.session


def _write_wav(path: Path, *, sample_rate: int = 8_000, channels: int = 2) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes((b"\x10\x00\xf0\xff") * 20)


@pytest.fixture
def pinned_model(tmp_path: Path) -> tuple[Path, str]:
    model_dir = tmp_path / "gigaam"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"pinned-onnx")
    pin = {
        "kind": "model",
        "repo_id": "example/gigaam-v3-rnnt",
        "revision": "immutable-revision",
        "local_dir": str(model_dir),
        "files": {"model.onnx": sha256_file(model_dir / "model.onnx")},
    }
    (tmp_path / "hub-pins.json").write_text(json.dumps({"gigaam": pin}), encoding="utf-8")
    return model_dir, fingerprint(pin)


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    path = tmp_path / "input.wav"
    _write_wav(path)
    return path


@pytest.fixture
def fake_onnx_asr(monkeypatch) -> FakeOnnxAsr:
    fake = FakeOnnxAsr()
    monkeypatch.setitem(sys.modules, "onnx_asr", fake)
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]),
    )
    return fake


def test_gigaam_preserves_successful_empty_hypothesis_and_resamples_mono(
    fake_onnx_asr: FakeOnnxAsr, pinned_model: tuple[Path, str], wav_path: Path
):
    """Catches empty output being treated as an ASR failure or an unpinned CPU session."""
    model_dir, model_fingerprint = pinned_model
    fake_onnx_asr.session.result = ""
    model = GigaAMRNNT(model_dir, device_id=3, model_fingerprint=model_fingerprint)

    assert fake_onnx_asr.calls == []
    assert model.transcribe(wav_path) == ""
    assert fake_onnx_asr.calls == [("gigaam-v3-rnnt", model_dir)]
    assert fake_onnx_asr.providers[0][0] == "CUDAExecutionProvider"
    assert fake_onnx_asr.providers[0][1]["device_id"] == 3
    assert fake_onnx_asr.providers[1] == "CPUExecutionProvider"
    waveform, sample_rate = fake_onnx_asr.session.calls[0]
    assert waveform.ndim == 1
    assert str(waveform.dtype) == "float32"
    assert sample_rate == 16_000


def test_gigaam_uses_only_the_verified_local_hub_pin(
    fake_onnx_asr: FakeOnnxAsr, pinned_model: tuple[Path, str], wav_path: Path
):
    """Catches onnx-asr being allowed to resolve a latest remote GigaAM revision."""
    model_dir, model_fingerprint = pinned_model

    GigaAMRNNT(model_dir, device_id=0, model_fingerprint=model_fingerprint).transcribe(wav_path)

    assert fake_onnx_asr.calls == [("gigaam-v3-rnnt", model_dir)]


def test_gigaam_rejects_requested_cuda_provider_when_unavailable(
    fake_onnx_asr: FakeOnnxAsr, monkeypatch, pinned_model: tuple[Path, str], wav_path: Path
):
    """Catches production validation silently falling back to CPU-only ASR."""
    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])
    )
    model_dir, model_fingerprint = pinned_model

    with pytest.raises(GigaAMProviderError, match="CUDAExecutionProvider.*device 2"):
        GigaAMRNNT(model_dir, device_id=2, model_fingerprint=model_fingerprint).transcribe(wav_path)

    assert fake_onnx_asr.calls == []


def test_gigaam_rejects_a_tampered_pin_before_opening_a_session(
    fake_onnx_asr: FakeOnnxAsr, pinned_model: tuple[Path, str], wav_path: Path
):
    """Catches model-file or revision drift being hidden behind a stale fingerprint."""
    model_dir, model_fingerprint = pinned_model
    (model_dir / "model.onnx").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="pinned GigaAM file hash"):
        GigaAMRNNT(model_dir, device_id=0, model_fingerprint=model_fingerprint).transcribe(wav_path)

    assert fake_onnx_asr.calls == []


def test_gigaam_normalizes_single_item_results_and_releases_its_session(
    fake_onnx_asr: FakeOnnxAsr, pinned_model: tuple[Path, str], wav_path: Path
):
    """Catches adapter/session lifetimes leaking over successive validation boundaries."""
    model_dir, model_fingerprint = pinned_model
    fake_onnx_asr.session.result = ["готово"]

    with GigaAMRNNT(model_dir, device_id=0, model_fingerprint=model_fingerprint) as model:
        assert model.transcribe(wav_path) == "готово"

    assert fake_onnx_asr.session.closed == 1
