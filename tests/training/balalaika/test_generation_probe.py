from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import soundfile as sf
import torch

from voxcpm.training.balalaika.artifacts import sha256_file
from voxcpm.training.balalaika.config import LoRAConfig

_SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "probe_balalaika_generation.py"
_SPEC = importlib.util.spec_from_file_location("probe_balalaika_generation", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


class FakeProbeModel(torch.nn.Module):
    sample_rate = 16_000

    def __init__(self, active: list[bool]):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))
        self.balalaika_base_revision = "base-sha"
        self.active = active
        self.adapter_loaded = False
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        assert self.adapter_loaded
        assert hasattr(self, "audio_vae")
        assert self.active == [True]
        self.calls.append(dict(kwargs))
        return np.full(160, 0.125, dtype=np.float32)


class FakeProbeVAE(torch.nn.Module):
    pass


class FakeCheckpointManager:
    instances: list["FakeCheckpointManager"] = []

    def __init__(self, root: Path):
        self.root = Path(root)
        self.loads: list[tuple[Path, dict[str, object]]] = []
        type(self).instances.append(self)

    def load_verified_adapter(self, model, checkpoint: Path, *, expected):
        self.loads.append((Path(checkpoint), dict(expected)))
        model.adapter_loaded = True
        return {"checkpoint_fingerprint": "checkpoint-sha"}


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 80)


def _fixture(tmp_path: Path):
    selection_dir = tmp_path / "selection"
    texts = ("это более длинный текст", "кратко", "средний текст", "еще один длинный текст")
    samples = []
    for index, text in enumerate(texts):
        wav_path = selection_dir / "audio" / f"item-{index:02d}.wav"
        _write_wav(wav_path)
        samples.append(
            {
                "source_relative_path": f"000000/item-{index:02d}.wav",
                "agreement": 0.99,
                "stage": 2,
                "text": text,
                "wav_path": str(wav_path),
                "wav_sha256": sha256_file(wav_path),
            }
        )
    (selection_dir / "memorization.json").write_text(
        json.dumps(
            {"schema_version": 1, "fingerprint": "selection-sha", "seed": 17, "memorization": samples},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    hub_dir = tmp_path / "hub"
    (hub_dir / "gigaam").mkdir(parents=True)
    (hub_dir / "hub-pins.json").write_text(
        json.dumps(
            {
                "gigaam": {
                    "kind": "model",
                    "revision": "asr-sha",
                    "local_dir": str(hub_dir / "gigaam"),
                    "files": {"model.onnx": "unused-in-fake"},
                }
            }
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        output_dir=tmp_path / "runs",
        selection_dir=selection_dir,
        data=SimpleNamespace(corpus_root=tmp_path / "corpus", index_dir=tmp_path / "index"),
        hub=SimpleNamespace(local_dir=hub_dir),
        lora=LoRAConfig(r=8, alpha=16),
        memorization=SimpleNamespace(updates=256, seed=101),
        generation=SimpleNamespace(cfg_value=4.0, inference_timesteps=19, max_length=321),
    )
    checkpoint = tmp_path / "runs" / "memorization" / "memorization-final"
    checkpoint.mkdir(parents=True)
    return config, checkpoint, samples


def _install_generation_fakes(monkeypatch, active: list[bool]):
    model = FakeProbeModel(active)
    vae = FakeProbeVAE()

    def build_model(config, stage, adapter_checkpoint=None):
        del config, adapter_checkpoint
        assert stage == 1
        return model, vae, lambda text: [len(text)]

    @contextmanager
    def required_context(device):
        assert device == torch.device("cpu")
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    FakeCheckpointManager.instances.clear()
    monkeypatch.setattr(probe, "build_model", build_model)
    monkeypatch.setattr(probe, "CheckpointManager", FakeCheckpointManager)
    monkeypatch.setattr(probe, "generation_autocast", required_context)
    monkeypatch.setattr(probe, "_require_one_cuda_device", lambda: torch.device("cpu"))
    return model, vae


def test_probe_verifies_recovery_adapter_enters_shared_context_and_writes_only_output(tmp_path, monkeypatch):
    """Catches bypassed adapter verification, wrong generation settings, or writes into protected inputs."""
    config, checkpoint, samples = _fixture(tmp_path)
    active: list[bool] = []
    model, _vae = _install_generation_fakes(monkeypatch, active)
    output = config.output_dir / "probes" / "memorization.wav"

    result = probe.run_probe(config, checkpoint, output, gigaam=False)

    assert result["status"] == "complete"
    assert result["output"] == str(output.resolve())
    assert result["sample_id"] == samples[1]["source_relative_path"]
    assert result["text"] == samples[1]["text"]
    assert result["gigaam_diagnostic"] is None
    assert sf.info(output).samplerate == 16_000
    assert len(FakeCheckpointManager.instances) == 1
    loaded_path, expected = FakeCheckpointManager.instances[0].loads[0]
    assert loaded_path == checkpoint.resolve()
    assert expected == {
        "checkpoint_kind": "recovery",
        "stage": "stage1",
        "optimizer_step": 256,
        "base_revision": "base-sha",
        "selection_fingerprint": "selection-sha",
        "lora_fingerprint": probe.fingerprint(config.lora.model_dump(mode="json")),
    }
    assert model.calls == [
        {
            "target_text": samples[1]["text"],
            "prompt_text": "",
            "prompt_wav_path": "",
            "seed": 101,
            "cfg_value": 4.0,
            "inference_timesteps": 19,
            "max_len": 321,
        }
    ]
    assert not hasattr(model, "audio_vae")
    assert list(checkpoint.iterdir()) == []


def test_probe_refuses_output_inside_checkpoint_or_prepared_inputs(tmp_path, monkeypatch):
    """Catches the diagnostic overwriting checkpoint, corpus, index, or selection state."""
    config, checkpoint, _samples = _fixture(tmp_path)
    monkeypatch.setattr(probe, "_require_one_cuda_device", lambda: pytest.fail("GPU check must follow path safety"))

    protected_outputs = (
        checkpoint / "probe.wav",
        config.data.corpus_root / "probe.wav",
        config.data.index_dir / "probe.wav",
        config.selection_dir / "probe.wav",
    )
    for output in protected_outputs:
        with pytest.raises(probe.GenerationProbeError, match="protected"):
            probe.run_probe(config, checkpoint, output, gigaam=False)
        assert not output.exists()


def test_probe_optionally_reports_gigaam_without_changing_generation_result(tmp_path, monkeypatch):
    """Catches the optional ASR diagnostic being omitted or gating a valid generated WAV."""
    config, checkpoint, _samples = _fixture(tmp_path)
    active: list[bool] = []
    _install_generation_fakes(monkeypatch, active)
    events: list[str] = []

    class FakeGigaAM:
        def __init__(self, model_dir, device_id, model_fingerprint):
            events.append(f"init:{Path(model_dir).name}:{device_id}:{bool(model_fingerprint)}")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append("close")

        def transcribe(self, wav_path):
            events.append(f"transcribe:{Path(wav_path).name}")
            return "диагностическая гипотеза"

    monkeypatch.setattr(probe, "GigaAMRNNT", FakeGigaAM)
    output = config.output_dir / "probe-with-asr.wav"

    result = probe.run_probe(config, checkpoint, output, gigaam=True)

    assert result["status"] == "complete"
    assert result["gigaam_diagnostic"] == "диагностическая гипотеза"
    assert events == ["init:gigaam:0:True", "transcribe:probe-with-asr.wav", "close"]


def test_probe_cli_requires_and_forwards_explicit_paths_and_gigaam_flag(tmp_path, monkeypatch, capsys):
    """Catches implicit checkpoint/output selection or a dropped diagnostic request."""
    config_path = tmp_path / "config.yaml"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "probe.wav"
    config = object()
    captured: list[tuple[object, Path, Path, bool]] = []
    monkeypatch.setattr(probe.BalalaikaConfig, "load", lambda path: config if path == config_path else None)

    def runner(loaded_config, loaded_checkpoint, loaded_output, *, gigaam):
        captured.append((loaded_config, loaded_checkpoint, loaded_output, gigaam))
        return {"status": "complete", "output": str(loaded_output)}

    exit_code = probe.main(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--gigaam",
        ],
        runner=runner,
    )

    assert exit_code == 0
    assert captured == [(config, checkpoint, output, True)]
    assert json.loads(capsys.readouterr().out) == {"output": str(output), "status": "complete"}
