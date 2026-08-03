from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from voxcpm.training.balalaika.cli import CommandResult, main
from voxcpm.training.balalaika.config import BalalaikaConfig
from voxcpm.training.balalaika.workflow import ProductionCommands, load_build_expectations


@dataclass
class RecordingCommands:
    calls: list[tuple[str, object]] = field(default_factory=list)

    def pin(self, config):
        self.calls.append(("pin", config))
        return CommandResult("pin", {"status": "complete"})

    def prepare(self, config):
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "prepared.txt").write_text("prepared", encoding="utf-8")
        self.calls.append(("prepare", config))
        return CommandResult("prepare", {"status": "complete"})

    def memorize(self, config, *, smoke):
        self.calls.append(("memorize", smoke))
        return CommandResult("memorize", {"status": "complete", "large_training_started": False})

    def approve(self, config, *, wandb_run_id):
        self.calls.append(("approve", wandb_run_id))
        return CommandResult("approve", {"status": "approved"})

    def train(self, config, *, stage, smoke, resume, stage1_checkpoint):
        self.calls.append(("train", (stage, smoke, resume, stage1_checkpoint)))
        return CommandResult("train", {"status": "complete", "stage": stage})

    def validate(self, config, *, checkpoint, smoke):
        self.calls.append(("validate", (checkpoint, smoke)))
        return CommandResult("validate", {"status": "complete"})

    def audit(self, config):
        self.calls.append(("audit", config))
        return CommandResult(
            "audit",
            {
                "status": "complete",
                "source_shards": config.data.expected_shards,
                "joined_rows": config.data.expected_rows,
                "excluded_null_agreement": config.data.expected_null_agreement,
            },
        )


def _config_value(tmp_path: Path) -> dict[str, object]:
    return {
        "data": {
            "corpus_root": str(tmp_path / "corpus"),
            "index_dir": str(tmp_path / "index"),
            "verification_manifest": str(tmp_path / "corpus" / "verification.json"),
            "combined_metadata": str(tmp_path / "corpus" / "combined.meta.json"),
            "expected_shards": 519,
            "expected_rows": 4_075_032,
            "expected_null_agreement": 309,
        },
        "output_dir": str(tmp_path / "runs"),
        "selection_dir": str(tmp_path / "selection"),
        "runtime": {"accumulation": 4, "workers": 4, "batch_candidates": [1, 2, 4]},
        "memorization": {"updates": 64, "learning_rate": 0.0001},
    }


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config_value(tmp_path)), encoding="utf-8")
    return path


def _invoke(config_path: Path, argv: list[str], commands: RecordingCommands, capsys, *, world_size: int = 1):
    code = main(
        ["--config", str(config_path), *argv],
        commands=commands,
        environ={"WORLD_SIZE": str(world_size), "USER": "operator"},
    )
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.parametrize("command", ["pin", "prepare", "audit"])
def test_non_distributed_commands_route_to_the_named_operation(config_path, command, capsys):
    """Catches a subparser dispatching a safe command to the wrong operation."""
    commands = RecordingCommands()

    code, stdout, stderr = _invoke(config_path, [command], commands, capsys)

    assert code == 0
    assert stderr == ""
    assert json.loads(stdout)["command"] == command
    assert [call[0] for call in commands.calls] == [command]


def test_prepare_does_not_mutate_corpus(config_path, capsys):
    """Catches preparation writing generated state beneath the immutable corpus root."""
    corpus = config_path.parent / "corpus"
    corpus.mkdir()
    (corpus / "immutable.bin").write_bytes(b"source-corpus")
    before = _file_hashes(corpus)

    code, _, _ = _invoke(config_path, ["prepare"], RecordingCommands(), capsys)

    assert code == 0
    assert _file_hashes(corpus) == before


def test_memorize_and_approve_route_without_starting_large_training(config_path, capsys):
    """Catches memorization or approval falling through into a stage-training route."""
    commands = RecordingCommands()

    memorize_code, memorize_stdout, _ = _invoke(config_path, ["memorize"], commands, capsys, world_size=8)
    approve_code, approve_stdout, _ = _invoke(config_path, ["approve", "--wandb-run-id", "run-123"], commands, capsys)

    assert memorize_code == approve_code == 0
    assert json.loads(memorize_stdout)["large_training_started"] is False
    assert json.loads(approve_stdout)["status"] == "approved"
    assert commands.calls == [("memorize", False), ("approve", "run-123")]


def test_train_stage1_requires_matching_approval(config_path, capsys):
    """Catches stage 1 reaching setup without an operator approval artifact."""
    code, _, stderr = _invoke(config_path, ["train", "--stage", "1"], RecordingCommands(), capsys, world_size=8)

    assert code != 0
    assert "memorization approval" in stderr.lower()


def test_train_routes_each_stage_after_required_artifacts_exist(config_path, capsys):
    """Catches a legal stage number being routed with the wrong stage or transition input."""
    commands = RecordingCommands()
    approval = config_path.parent / "runs" / "memorization" / "memorization-approval.json"
    approval.parent.mkdir(parents=True)
    approval.write_text("{}", encoding="utf-8")
    stage1 = config_path.parent / "stage1-final"
    stage1.mkdir()
    (stage1 / "adapter_model.safetensors").write_bytes(b"adapter")
    (stage1 / "metadata.json").write_text(
        json.dumps({"stage": "stage1", "epoch": 1, "boundary": 8, "checkpoint_kind": "boundary"}),
        encoding="utf-8",
    )

    first_code, _, first_error = _invoke(config_path, ["train", "--stage", "1"], commands, capsys, world_size=8)
    second_code, _, second_error = _invoke(
        config_path,
        ["train", "--stage", "2", "--stage1-checkpoint", str(stage1)],
        commands,
        capsys,
        world_size=8,
    )

    assert first_code == second_code == 0
    assert first_error == second_error == ""
    assert commands.calls[0][1][0] == 1
    assert commands.calls[1][1][0] == 2
    assert commands.calls[1][1][3] == stage1.resolve()


def test_train_stage2_requires_completed_stage1_boundary_adapter(config_path, tmp_path, capsys):
    """Catches stage 2 accepting a missing or non-final stage-1 adapter."""
    incomplete = tmp_path / "incomplete-stage1"
    incomplete.mkdir()
    (incomplete / "adapter_model.safetensors").write_bytes(b"adapter")
    (incomplete / "metadata.json").write_text(
        json.dumps({"stage": "stage1", "epoch": 0, "boundary": 7, "checkpoint_kind": "boundary"}),
        encoding="utf-8",
    )

    code, _, stderr = _invoke(
        config_path,
        ["train", "--stage", "2", "--stage1-checkpoint", str(incomplete)],
        RecordingCommands(),
        capsys,
        world_size=8,
    )

    assert code != 0
    assert "completed stage-1" in stderr.lower()


def test_validate_requires_checkpoint(config_path, capsys):
    """Catches standalone validation running without an immutable checkpoint identity."""
    code, _, stderr = _invoke(config_path, ["validate"], RecordingCommands(), capsys, world_size=8)

    assert code != 0
    assert "checkpoint" in stderr.lower()


def test_validate_routes_an_existing_checkpoint(config_path, tmp_path, capsys):
    """Catches validation dropping the operator-selected checkpoint path."""
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    commands = RecordingCommands()

    code, _, stderr = _invoke(
        config_path,
        ["validate", "--checkpoint", str(checkpoint)],
        commands,
        capsys,
        world_size=8,
    )

    assert code == 0
    assert stderr == ""
    assert commands.calls == [("validate", (checkpoint.resolve(), False))]


def test_large_command_rejects_non_eight_processes_unless_smoke(config_path, capsys):
    """Catches an accidental one-rank production launch while retaining explicit smoke mode."""
    approval = config_path.parent / "runs" / "memorization" / "memorization-approval.json"
    approval.parent.mkdir(parents=True)
    approval.write_text("{}", encoding="utf-8")
    commands = RecordingCommands()

    rejected, _, rejected_error = _invoke(config_path, ["train", "--stage", "1"], commands, capsys, world_size=4)
    accepted, _, accepted_error = _invoke(
        config_path, ["train", "--stage", "1", "--smoke"], commands, capsys, world_size=1
    )

    assert rejected != 0
    assert "exactly 8 processes" in rejected_error
    assert accepted == 0
    assert accepted_error == ""
    assert commands.calls == [("train", (1, True, None, None))]


def test_audit_surfaces_production_null_row_expectation(config_path, capsys):
    """Catches audit output omitting the fixed production null-agreement exclusion."""
    code, stdout, _ = _invoke(config_path, ["audit"], RecordingCommands(), capsys)

    payload = json.loads(stdout)
    assert code == 0
    assert payload["source_shards"] == 519
    assert payload["joined_rows"] == 4_075_032
    assert payload["excluded_null_agreement"] == 309


def test_load_build_expectations_binds_every_verified_shard(tmp_path):
    """Catches preparation trusting row totals without binding the complete source inventory."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    verification = corpus / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "status": "ok",
                "stats": {"samples": 3},
                "shards": [
                    {"shard": "shard_000000.tar", "samples": 2, "sha256": "a" * 64},
                    {"shard": "shard_000001.tar", "samples": 1, "sha256": "b" * 64},
                ],
            }
        ),
        encoding="utf-8",
    )
    combined = corpus / "combined.meta.json"
    combined.write_text(
        json.dumps(
            {
                "complete": True,
                "rows": 3,
                "inputs": {"provenance_binding": {"trusted_release": True, "rover_archive_sha256": "c" * 64}},
                "output": {"sha256": "d" * 64},
            }
        ),
        encoding="utf-8",
    )
    config = BalalaikaConfig.model_validate(
        {
            "data": {
                "corpus_root": corpus,
                "index_dir": tmp_path / "index",
                "verification_manifest": verification,
                "combined_metadata": combined,
                "expected_shards": 2,
                "expected_rows": 3,
                "expected_null_agreement": 1,
            },
            "output_dir": tmp_path / "runs",
        }
    )

    expectations = load_build_expectations(config.data)

    assert expectations.source_shard_count == 2
    assert expectations.source_row_count == expectations.rover_row_count == expectations.combined_row_count == 3
    assert expectations.source_tar_sha256 == {
        "train/shard_000000.tar": "a" * 64,
        "train/shard_000001.tar": "b" * 64,
    }
    assert expectations.rover_archive_sha256 == "c" * 64
    assert expectations.combined_sidecar_sha256 == "d" * 64


def test_production_prepare_builds_index_then_fixed_selection(config_path, tmp_path):
    """Catches preparation publishing selections before its verified index or using an unpinned benchmark."""
    config = BalalaikaConfig.load(config_path)
    benchmark = config.hub.local_dir / "benchmark" / config.hub.benchmark_file
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"id": 0}\n', encoding="utf-8")
    events: list[tuple[str, object]] = []
    audit = SimpleNamespace(
        index_path=config.data.index_dir / "balalaika-index.sqlite3",
        audit_path=config.data.index_dir / "balalaika-index-audit.json",
        fingerprint="index-fingerprint",
        total_rows=4_075_032,
        eligible_rows=4_074_723,
        stage1_rows=1_000_000,
        stage2_rows=3_074_723,
        excluded_null_agreement=309,
    )
    selection = SimpleNamespace(
        fingerprint="selection-fingerprint",
        memorization=[1, 2, 3, 4],
        prompts=list(range(20)),
        benchmark_prompt_by_id={item: "prompt-00" for item in range(2_000)},
    )

    def expectation_loader(data):
        events.append(("expectations", data))
        return "trusted-expectations"

    def index_builder(data, expectations):
        events.append(("index", (data, expectations)))
        return audit

    def selection_builder(index_path, benchmark_path, output_dir, seed):
        events.append(("selection", (index_path, benchmark_path, output_dir, seed)))
        return selection

    commands = ProductionCommands(
        expectation_loader=expectation_loader,
        index_builder=index_builder,
        selection_builder=selection_builder,
    )

    result = commands.prepare(config)

    assert [event[0] for event in events] == ["expectations", "index", "selection"]
    assert events[2][1] == (audit.index_path, benchmark, config.selection_dir, config.runtime.seed)
    assert result.values["excluded_null_agreement"] == 309
    assert result.values["memorization_samples"] == 4
    assert result.values["validation_prompts"] == 20
    assert result.values["benchmark_assignments"] == 2_000


def test_checked_in_config_loads_complete_operator_defaults():
    """Catches the shipped operator config omitting an input identity or required runtime default."""
    root = Path(__file__).resolve().parents[3]
    config = BalalaikaConfig.load(root / "conf" / "voxcpm_v2" / "balalaika_lora.yaml")

    assert config.data.corpus_root == Path("/workspace/balalaika_proprietary_v2")
    assert config.data.expected_shards == 519
    assert config.data.expected_rows == 4_075_032
    assert config.data.expected_null_agreement == 309
    assert config.hub.model_repo_id == "OpenBMB/VoxCPM2"
    assert config.hub.benchmark_repo_id == "bitmanagerai/hard_number_eval_for_tts"
    assert config.hub.gigaam_repo_id == "istupakov/gigaam-v3-onnx"
    assert config.runtime.batch_candidates == [1, 2, 4]
    assert config.stage1.epochs == 2 and config.stage2.epochs == 3
    assert config.lora.r == config.lora.alpha == 32
    assert config.wandb.project == "voxcpm-balalaika"
    assert config.wandb.mode in {"online", "offline"}


def test_production_audit_rejects_a_changed_pinned_file(tmp_path):
    """Catches audit reporting success after a locally pinned Hub artifact changes."""
    value = _config_value(tmp_path)
    value["hub"] = {"local_dir": str(tmp_path / "hub"), "gigaam_repo_id": "istupakov/gigaam-v3-onnx"}
    config = BalalaikaConfig.model_validate(value)
    config.data.index_dir.mkdir(parents=True)
    index_path = config.data.index_dir / "balalaika-index.sqlite3"
    index_path.write_bytes(b"verified-index")
    (config.data.index_dir / "balalaika-index-audit.json").write_text(
        json.dumps(
            {
                "index_sha256": hashlib.sha256(b"verified-index").hexdigest(),
                "fingerprint": "index-fingerprint",
                "total_rows": 4_075_032,
                "eligible_rows": 4_074_723,
                "stage1_rows": 1_000_000,
                "stage2_rows": 3_074_723,
                "excluded_null_agreement": 309,
            }
        ),
        encoding="utf-8",
    )
    config.selection_dir.mkdir(parents=True)
    common = {"fingerprint": "selection-fingerprint", "seed": 20_260_802}
    selections = {
        "memorization.json": {**common, "memorization": list(range(4))},
        "prompts.json": {**common, "prompts": list(range(20))},
        "benchmark-prompts.json": {
            **common,
            "benchmark_prompt_by_id": {str(item): "prompt-00" for item in range(2_000)},
        },
        "audio-log-ids.json": {**common, "audio_log_ids": [0, 1, 2, 3]},
    }
    for name, payload in selections.items():
        (config.selection_dir / name).write_text(json.dumps(payload), encoding="utf-8")
    pin_values = {}
    for name, kind, repo_id in (
        ("model", "model", config.hub.model_repo_id),
        ("benchmark", "dataset", config.hub.benchmark_repo_id),
        ("gigaam", "model", config.hub.gigaam_repo_id),
    ):
        local_dir = config.hub.local_dir / name
        local_dir.mkdir(parents=True)
        artifact = local_dir / "artifact.bin"
        artifact.write_bytes(name.encode())
        pin_values[name] = {
            "kind": kind,
            "repo_id": repo_id,
            "revision": f"{name}-revision",
            "local_dir": str(local_dir.resolve()),
            "files": {"artifact.bin": hashlib.sha256(name.encode()).hexdigest()},
        }
    (config.hub.local_dir / "hub-pins.json").write_text(json.dumps(pin_values), encoding="utf-8")
    commands = ProductionCommands()

    assert commands.audit(config).values["status"] == "complete"
    (config.hub.local_dir / "model" / "artifact.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed after download"):
        commands.audit(config)
