import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from voxcpm.training.balalaika.config import HubConfig
from voxcpm.training.balalaika.hub import pin_hub_inputs


MODEL = "OpenBMB/VoxCPM2"
DATASET = "bitmanagerai/hard_number_eval_for_tts"
GIGAAM = "example/gigaam-v3-rnnt"


class FakeRunner:
    def __init__(self, responses: dict[tuple[str, ...], str]):
        self.responses = responses
        self.calls: list[tuple[str, ...]] = []

    def run(self, argv: tuple[str, ...]) -> CompletedProcess[str]:
        call = tuple(argv)
        self.calls.append(call)
        if call not in self.responses:
            raise AssertionError(f"Unexpected command: {call}")
        return CompletedProcess(call, 0, self.responses[call], "")


def hub_config(tmp_path: Path, *, gigaam_repo_id: str | None = None) -> HubConfig:
    return HubConfig(local_dir=tmp_path, gigaam_repo_id=gigaam_repo_id)


def write_downloaded_files(root: Path) -> None:
    (root / "model").mkdir(parents=True)
    (root / "model" / "config.json").write_text('{"architectures": ["VoxCPM2"]}')
    (root / "benchmark").mkdir()
    (root / "benchmark" / "data.jsonl").write_text('{"id": 1}\n')


def test_pin_private_benchmark_uses_hf_cli_and_immutable_sha(tmp_path):
    """Catches a mutable revision or explicit token added to dataset download."""
    write_downloaded_files(tmp_path)
    runner = FakeRunner(
        {
            ("hf", "models", "info", MODEL): json.dumps({"sha": "model-sha"}),
            (
                "hf",
                "download",
                MODEL,
                "--revision",
                "model-sha",
                "--local-dir",
                str(tmp_path / "model"),
            ): str(tmp_path / "model"),
            ("hf", "datasets", "info", DATASET): json.dumps({"sha": "abc123"}),
            (
                "hf",
                "download",
                DATASET,
                "--repo-type",
                "dataset",
                "--revision",
                "abc123",
                "--local-dir",
                str(tmp_path / "benchmark"),
            ): str(tmp_path / "benchmark"),
        }
    )

    pins = pin_hub_inputs(hub_config(tmp_path), runner)

    assert pins["benchmark"].revision == "abc123"
    assert pins["benchmark"].files == {
        "data.jsonl": "82b0cf5da91b6a7e02f031e2da2fa5ed1261dce4c85ae20d63db9d1e84a4c384"
    }
    assert all("--token" not in argv for argv in runner.calls)
    assert json.loads((tmp_path / "hub-pins.json").read_text())["model"]["revision"] == "model-sha"


def test_pin_model_uses_model_info_and_hashes_downloaded_files(tmp_path):
    """Catches model repositories being inspected as datasets or left unhashed."""
    write_downloaded_files(tmp_path)
    runner = FakeRunner(
        {
            ("hf", "models", "info", MODEL): json.dumps({"sha": "model-sha"}),
            (
                "hf",
                "download",
                MODEL,
                "--revision",
                "model-sha",
                "--local-dir",
                str(tmp_path / "model"),
            ): str(tmp_path / "model"),
            ("hf", "datasets", "info", DATASET): json.dumps({"sha": "dataset-sha"}),
            (
                "hf",
                "download",
                DATASET,
                "--repo-type",
                "dataset",
                "--revision",
                "dataset-sha",
                "--local-dir",
                str(tmp_path / "benchmark"),
            ): str(tmp_path / "benchmark"),
        }
    )

    pin = pin_hub_inputs(hub_config(tmp_path), runner)["model"]

    assert pin.kind == "model"
    assert pin.revision == "model-sha"
    assert pin.files == {
        "config.json": "47533c64f873613d78ff1f675cd82e890623da220788751cc4bb31f3db952407"
    }
    assert runner.calls[0] == ("hf", "models", "info", MODEL)


def test_pin_rejects_info_without_immutable_sha(tmp_path):
    """Catches a response without a commit SHA being treated as immutable."""
    runner = FakeRunner({("hf", "models", "info", MODEL): json.dumps({"id": MODEL})})

    with pytest.raises(ValueError, match="immutable sha"):
        pin_hub_inputs(hub_config(tmp_path), runner)

    assert runner.calls == [("hf", "models", "info", MODEL)]


def test_pin_reuses_manifest_only_when_repo_revision_and_file_hashes_match(tmp_path):
    """Catches stale local data being reused after its content changes."""
    write_downloaded_files(tmp_path)
    first_runner = FakeRunner(
        {
            ("hf", "models", "info", MODEL): json.dumps({"sha": "model-sha"}),
            (
                "hf",
                "download",
                MODEL,
                "--revision",
                "model-sha",
                "--local-dir",
                str(tmp_path / "model"),
            ): str(tmp_path / "model"),
            ("hf", "datasets", "info", DATASET): json.dumps({"sha": "dataset-sha"}),
            (
                "hf",
                "download",
                DATASET,
                "--repo-type",
                "dataset",
                "--revision",
                "dataset-sha",
                "--local-dir",
                str(tmp_path / "benchmark"),
            ): str(tmp_path / "benchmark"),
        }
    )
    pin_hub_inputs(hub_config(tmp_path), first_runner)

    reused_runner = FakeRunner(
        {
            ("hf", "models", "info", MODEL): json.dumps({"sha": "model-sha"}),
            ("hf", "datasets", "info", DATASET): json.dumps({"sha": "dataset-sha"}),
        }
    )
    pin_hub_inputs(hub_config(tmp_path), reused_runner)
    assert all("download" not in argv for argv in reused_runner.calls)

    (tmp_path / "benchmark" / "data.jsonl").write_text('{"id": 2}\n')
    stale_runner = FakeRunner(
        {
            ("hf", "models", "info", MODEL): json.dumps({"sha": "model-sha"}),
            ("hf", "datasets", "info", DATASET): json.dumps({"sha": "dataset-sha"}),
            (
                "hf",
                "download",
                DATASET,
                "--repo-type",
                "dataset",
                "--revision",
                "dataset-sha",
                "--local-dir",
                str(tmp_path / "benchmark"),
            ): str(tmp_path / "benchmark"),
        }
    )
    pin_hub_inputs(hub_config(tmp_path), stale_runner)
    assert stale_runner.calls[-1][1] == "download"


def test_pin_configured_gigaam_as_model(tmp_path):
    """Catches configured ONNX ASR weights being omitted from the pin manifest."""
    write_downloaded_files(tmp_path)
    (tmp_path / "gigaam").mkdir()
    (tmp_path / "gigaam" / "model.onnx").write_bytes(b"onnx")
    runner = FakeRunner(
        {
            ("hf", "models", "info", MODEL): json.dumps({"sha": "model-sha"}),
            (
                "hf", "download", MODEL, "--revision", "model-sha", "--local-dir", str(tmp_path / "model")
            ): str(tmp_path / "model"),
            ("hf", "datasets", "info", DATASET): json.dumps({"sha": "dataset-sha"}),
            (
                "hf", "download", DATASET, "--repo-type", "dataset", "--revision", "dataset-sha", "--local-dir", str(tmp_path / "benchmark")
            ): str(tmp_path / "benchmark"),
            ("hf", "models", "info", GIGAAM): json.dumps({"sha": "gigaam-sha"}),
            (
                "hf", "download", GIGAAM, "--revision", "gigaam-sha", "--local-dir", str(tmp_path / "gigaam")
            ): str(tmp_path / "gigaam"),
        }
    )

    pins = pin_hub_inputs(hub_config(tmp_path, gigaam_repo_id=GIGAAM), runner)

    assert pins["gigaam"].repo_id == GIGAAM
    assert pins["gigaam"].revision == "gigaam-sha"
