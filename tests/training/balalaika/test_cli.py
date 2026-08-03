from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import voxcpm.training.balalaika.workflow as workflow_module
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


def _mark_generation(root: Path, role: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".balalaika-generation").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner": "voxcpm-balalaika",
                "role": role,
                "canonical_root": str(root.resolve()),
                "generation_id": "previous",
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )


def test_runbook_uses_project_accelerate_launcher():
    repository_root = Path(__file__).resolve().parents[3]
    runbook = (repository_root / "docs" / "balalaika_training.md").read_text(encoding="utf-8")

    assert "rtk accelerate launch" not in runbook
    assert "rtk uv run accelerate launch" in runbook


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


def test_memorize_smoke_rejects_before_runtime_or_runner_side_effects(config_path, tmp_path, capsys):
    """Catches a smoke flag bypassing the process guard and entering real memorization setup."""
    side_effect = tmp_path / "real-memorization-started"

    def runtime_factory(_config):
        side_effect.write_text("runtime-created", encoding="utf-8")
        raise AssertionError("real runtime must not be created by smoke")

    commands = ProductionCommands(runtime_factory=runtime_factory)

    code, _, stderr = _invoke(config_path, ["memorize", "--smoke"], commands, capsys)

    assert code == 2
    assert "scripts/smoke_balalaika_accelerate.py" in stderr
    assert not side_effect.exists()


@pytest.mark.parametrize("runner_fails", [False, True])
def test_real_memorization_closes_its_runtime_once_on_success_and_failure(config_path, runner_fails, monkeypatch):
    """Catches the real memorization runtime owner leaking Accelerate resources on either exit path."""
    config = BalalaikaConfig.load(config_path)
    events: list[str] = []

    class Runtime:
        def close(self):
            events.append("close")

    def run(_config, _runtime):
        events.append("run")
        if runner_fails:
            raise RuntimeError("memorization failed")
        return SimpleNamespace(
            status="stopped",
            result_path=config.output_dir / "result.json",
            checkpoint=config.output_dir / "checkpoint",
            wandb_run_id="run-123",
            large_training_started=False,
        )

    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})
    monkeypatch.setattr(workflow_module, "_verify_current_prepared_generation", lambda _config: ("index", object()))
    commands = ProductionCommands(runtime_factory=lambda _config: Runtime(), memorization_runner=run)

    if runner_fails:
        with pytest.raises(RuntimeError, match="memorization failed"):
            commands.memorize(config, smoke=False)
    else:
        assert commands.memorize(config, smoke=False).values["status"] == "stopped"

    assert events == ["run", "close"]


def test_memorization_preserves_runner_failure_when_runtime_close_also_fails(config_path, monkeypatch):
    """Catches teardown masking the operational failure that operators must diagnose."""
    config = BalalaikaConfig.load(config_path)

    class Runtime:
        def close(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})
    monkeypatch.setattr(workflow_module, "_verify_current_prepared_generation", lambda _config: ("index", object()))
    commands = ProductionCommands(
        runtime_factory=lambda _config: Runtime(),
        memorization_runner=lambda *_args: (_ for _ in ()).throw(ValueError("runner failed")),
    )

    with pytest.raises(ValueError, match="runner failed"):
        commands.memorize(config, smoke=False)


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


def test_load_build_expectations_binds_every_augmented_shard_to_its_verified_source(tmp_path):
    """Catches hashing augmented tar files against their pre-augmentation source digests."""
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
    manifests = corpus / "manifests"
    manifests.mkdir()
    for shard, source_digest, augmented_digest, samples in (
        ("000000", "a" * 64, "e" * 64, 2),
        ("000001", "b" * 64, "f" * 64, 1),
    ):
        (manifests / f"shard_{shard}.json").write_text(
            json.dumps(
                {
                    "shard_id": shard,
                    "samples": samples,
                    "members": samples * 2,
                    "source_sha256": source_digest,
                    "sha256": augmented_digest,
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
        "train/shard_000000.tar": "e" * 64,
        "train/shard_000001.tar": "f" * 64,
    }
    assert expectations.rover_archive_sha256 == "c" * 64
    assert expectations.combined_sidecar_sha256 == "d" * 64


def test_production_prepare_builds_index_then_fixed_selection(config_path, tmp_path, monkeypatch):
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
        generation_validator=lambda *_args: None,
    )
    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})

    result = commands.prepare(config)

    assert [event[0] for event in events] == ["expectations", "index", "selection"]
    selected_index, selected_benchmark, selected_output, selected_seed = events[2][1]
    assert selected_index == audit.index_path
    assert selected_benchmark == benchmark
    assert selected_output != config.selection_dir
    assert selected_output.parent == config.selection_dir.parent
    assert selected_seed == config.runtime.seed
    assert result.values["excluded_null_agreement"] == 309
    assert result.values["memorization_samples"] == 4
    assert result.values["validation_prompts"] == 20
    assert result.values["benchmark_assignments"] == 2_000


def test_production_prepare_verifies_all_pins_before_reading_inputs(config_path, monkeypatch):
    """Catches preparation consuming corpus or benchmark state before rehashing every immutable Hub pin."""
    config = BalalaikaConfig.load(config_path)
    benchmark = config.hub.local_dir / "benchmark" / config.hub.benchmark_file
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"id": 0}\n', encoding="utf-8")
    events: list[str] = []

    def verify_pins(_config):
        events.append("pins")
        return {"model": {}, "benchmark": {}, "gigaam": {}}

    def build_index(data, _expectations):
        events.append("index")
        data.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = data.index_dir / "balalaika-index.sqlite3"
        index_path.write_bytes(b"new-index")
        audit_path = data.index_dir / "balalaika-index-audit.json"
        audit_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            index_path=index_path,
            audit_path=audit_path,
            fingerprint="index-fingerprint",
            total_rows=4_075_032,
            eligible_rows=4_074_723,
            stage1_rows=1_000_000,
            stage2_rows=3_074_723,
            excluded_null_agreement=309,
        )

    def build_selection(*_args):
        events.append("selection")
        return SimpleNamespace(
            fingerprint="selection-fingerprint",
            memorization=[1, 2, 3, 4],
            prompts=list(range(20)),
            benchmark_prompt_by_id={item: "prompt-00" for item in range(2_000)},
        )

    monkeypatch.setattr(workflow_module, "_verified_pins", verify_pins)
    commands = ProductionCommands(
        expectation_loader=lambda _data: events.append("expectations") or "expectations",
        index_builder=build_index,
        selection_builder=build_selection,
        generation_validator=lambda *_args: None,
    )

    commands.prepare(config)

    assert events == ["pins", "expectations", "index", "selection"]


def test_prepare_selection_failure_restores_previous_index_without_copying(config_path, monkeypatch):
    """Catches a failed selection leaving a newly published index paired with the prior selection generation."""
    config = BalalaikaConfig.load(config_path)
    benchmark = config.hub.local_dir / "benchmark" / config.hub.benchmark_file
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"id": 0}\n', encoding="utf-8")
    _mark_generation(config.data.index_dir, "index")
    previous_index = config.data.index_dir / "balalaika-index.sqlite3"
    previous_index.write_bytes(b"previous-complete-index")
    previous_inode = previous_index.stat().st_ino
    (config.data.index_dir / "balalaika-index-audit.json").write_text('{"generation":"previous"}', encoding="utf-8")
    _mark_generation(config.selection_dir, "selection")
    (config.selection_dir / "complete-generation.txt").write_text("previous", encoding="utf-8")

    def build_index(data, _expectations):
        data.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = data.index_dir / "balalaika-index.sqlite3"
        index_path.write_bytes(b"new-uncommitted-index")
        audit_path = data.index_dir / "balalaika-index-audit.json"
        audit_path.write_text('{"generation":"new"}', encoding="utf-8")
        return SimpleNamespace(
            index_path=index_path,
            audit_path=audit_path,
            fingerprint="new-index-fingerprint",
            total_rows=4_075_032,
            eligible_rows=4_074_723,
            stage1_rows=1_000_000,
            stage2_rows=3_074_723,
            excluded_null_agreement=309,
        )

    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})
    commands = ProductionCommands(
        expectation_loader=lambda _data: "expectations",
        index_builder=build_index,
        selection_builder=lambda *_args: (_ for _ in ()).throw(RuntimeError("selection publication failed")),
    )

    with pytest.raises(RuntimeError, match="selection publication failed"):
        commands.prepare(config)

    assert previous_index.read_bytes() == b"previous-complete-index"
    assert previous_index.stat().st_ino == previous_inode
    assert (config.data.index_dir / "balalaika-index-audit.json").read_text(
        encoding="utf-8"
    ) == '{"generation":"previous"}'
    assert (config.selection_dir / "complete-generation.txt").read_text(encoding="utf-8") == "previous"


def test_prepare_refuses_to_commit_missing_generation_outputs_and_restores_previous(config_path, monkeypatch):
    """Catches a successful-returning fake or broken builder deleting the prior complete generation."""
    config = BalalaikaConfig.load(config_path)
    benchmark = config.hub.local_dir / "benchmark" / config.hub.benchmark_file
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"id": 0}\n', encoding="utf-8")
    _mark_generation(config.data.index_dir, "index")
    previous_index = config.data.index_dir / "balalaika-index.sqlite3"
    previous_index.write_bytes(b"previous-complete-index")
    previous_inode = previous_index.stat().st_ino
    (config.data.index_dir / "balalaika-index-audit.json").write_text('{"generation":"previous"}', encoding="utf-8")
    _mark_generation(config.selection_dir, "selection")
    previous_selection = config.selection_dir / "memorization.json"
    previous_selection.write_text('{"generation":"previous"}', encoding="utf-8")
    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})
    commands = ProductionCommands(
        expectation_loader=lambda _data: "expectations",
        index_builder=lambda data, _expectations: SimpleNamespace(
            index_path=data.index_dir / "balalaika-index.sqlite3",
            audit_path=data.index_dir / "balalaika-index-audit.json",
            fingerprint="missing-generation",
            total_rows=4_075_032,
            eligible_rows=4_074_723,
            stage1_rows=2_486_821,
            stage2_rows=1_587_902,
            excluded_null_agreement=309,
        ),
        selection_builder=lambda *_args: SimpleNamespace(
            fingerprint="missing-generation",
            memorization=[1, 2, 3, 4],
            prompts=list(range(20)),
            benchmark_prompt_by_id={item: "prompt-00" for item in range(2_000)},
        ),
    )

    with pytest.raises(ValueError, match="prepared generation.*missing"):
        commands.prepare(config)

    assert previous_index.read_bytes() == b"previous-complete-index"
    assert previous_index.stat().st_ino == previous_inode
    assert previous_selection.read_text(encoding="utf-8") == '{"generation":"previous"}'


def test_production_config_routes_the_pinned_hard_number_benchmark(tmp_path, monkeypatch):
    """Catches preparation assuming a benchmark filename absent from the pinned dataset revision."""
    root = Path(__file__).resolve().parents[3]
    config = BalalaikaConfig.load(root / "conf" / "voxcpm_v2" / "balalaika_lora.yaml")
    hub_root = tmp_path / "hub"
    benchmark = hub_root / "benchmark" / "hard_number_eval.jsonl"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text('{"id": 0}\n', encoding="utf-8")
    config = config.model_copy(
        update={
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
            "data": config.data.model_copy(update={"index_dir": tmp_path / "index"}),
            "hub": config.hub.model_copy(update={"local_dir": hub_root}),
        }
    )
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
    selected_paths: list[Path] = []

    def selection_builder(index_path, benchmark_path, output_dir, seed):
        del index_path, output_dir, seed
        benchmark_path = Path(benchmark_path)
        benchmark_path.read_text(encoding="utf-8")
        selected_paths.append(benchmark_path)
        return selection

    commands = ProductionCommands(
        expectation_loader=lambda data: "trusted-expectations",
        index_builder=lambda data, expectations: audit,
        selection_builder=selection_builder,
        generation_validator=lambda *_args: None,
    )
    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})

    commands.prepare(config)

    assert selected_paths == [benchmark]
    assert config.output_dir.is_relative_to(tmp_path)
    assert config.data.index_dir.is_relative_to(tmp_path)
    assert config.selection_dir.is_relative_to(tmp_path)
    assert config.hub.local_dir.is_relative_to(tmp_path)


def test_checked_in_config_loads_complete_operator_defaults():
    """Catches the shipped operator config omitting an input identity or required runtime default."""
    root = Path(__file__).resolve().parents[3]
    config = BalalaikaConfig.load(root / "conf" / "voxcpm_v2" / "balalaika_lora.yaml")

    assert config.data.corpus_root == Path("/workspace/balalaika_proprietary_v2")
    assert config.data.expected_shards == 519
    assert config.data.expected_rows == 4_075_032
    assert config.data.expected_null_agreement == 309
    assert config.data.expected_stage1_rows == 2_486_821
    assert config.data.expected_stage2_rows == 1_587_902
    assert config.hub.model_repo_id == "OpenBMB/VoxCPM2"
    assert config.hub.benchmark_repo_id == "bitmanagerai/hard_number_eval_for_tts"
    assert config.hub.gigaam_repo_id == "istupakov/gigaam-v3-onnx"
    assert config.runtime.batch_candidates == [1, 2, 4]
    assert config.stage1.epochs == 2 and config.stage2.epochs == 3
    assert config.lora.r == config.lora.alpha == 32
    assert config.wandb.project == "voxcpm-balalaika"
    assert config.wandb.mode in {"online", "offline"}
