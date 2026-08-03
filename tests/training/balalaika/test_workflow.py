from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest

import voxcpm.training.balalaika.workflow as workflow_module
from voxcpm.training.balalaika.artifacts import sha256_file
from voxcpm.training.balalaika.config import BalalaikaConfig, DataConfig
from voxcpm.training.balalaika.index import build_index
from voxcpm.training.balalaika.selection import create_selection_manifests
from voxcpm.training.balalaika.workflow import ProductionCommands


@pytest.fixture
def prepared_deep_audit(synthetic_corpus, tmp_path) -> BalalaikaConfig:
    identities = [f"{index // 13:06d}/item-{index:02d}.wav" for index in range(25)]
    synthetic_corpus.source_rows = [
        {"source_relative_path": identity, "shard": identity.split("/", maxsplit=1)[0], "include_audio": True}
        for identity in identities
    ]
    synthetic_corpus.rover_rows = [
        {
            "source_relative_path": identity,
            "asr_agreement_mean": 0.99 if index < 14 else (0.5 if index < 24 else None),
        }
        for index, identity in enumerate(identities)
    ]
    synthetic_corpus.combined_rows = [
        {"source_relative_path": identity, "rover_punctuated_accented": f"тест {index}"}
        for index, identity in enumerate(identities)
    ]
    synthetic_corpus._write_source_tars()
    synthetic_corpus._write_rover()
    synthetic_corpus._write_combined()
    synthetic_corpus._refresh_hashes()
    synthetic_corpus.expectations = replace(
        synthetic_corpus.expectations,
        source_row_count=25,
        rover_row_count=25,
        combined_row_count=25,
    )

    verification = synthetic_corpus.root / "verification.json"
    shard_rows = []
    manifests = synthetic_corpus.root / "manifests"
    manifests.mkdir()
    for tar_path in synthetic_corpus.source_tar_paths:
        shard_id = tar_path.stem.removeprefix("shard_")
        samples = sum(identity.startswith(f"{shard_id}/") for identity in identities)
        digest = sha256_file(tar_path)
        shard_rows.append({"shard": tar_path.name, "samples": samples, "sha256": digest})
        (manifests / f"shard_{shard_id}.json").write_text(
            json.dumps(
                {
                    "shard_id": shard_id,
                    "samples": samples,
                    "members": samples * 2,
                    "source_sha256": digest,
                    "sha256": digest,
                }
            ),
            encoding="utf-8",
        )
    verification.write_text(
        json.dumps({"status": "ok", "stats": {"samples": 25}, "shards": shard_rows}), encoding="utf-8"
    )
    combined_metadata = synthetic_corpus.root / "combined.meta.json"
    combined_metadata.write_text(
        json.dumps(
            {
                "complete": True,
                "rows": 25,
                "inputs": {
                    "provenance_binding": {
                        "trusted_release": True,
                        "rover_archive_sha256": sha256_file(synthetic_corpus.rover_archive),
                    }
                },
                "output": {"sha256": sha256_file(synthetic_corpus.combined_sidecar)},
            }
        ),
        encoding="utf-8",
    )
    data = DataConfig(
        corpus_root=synthetic_corpus.root,
        index_dir=tmp_path / "prepared",
        verification_manifest=verification,
        combined_metadata=combined_metadata,
        expected_shards=2,
        expected_rows=25,
        expected_null_agreement=1,
        expected_stage1_rows=10,
        expected_stage2_rows=14,
    )
    hub_root = tmp_path / "hub"
    benchmark = hub_root / "benchmark" / "data.jsonl"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(
        "".join(
            json.dumps(
                {
                    "id": identifier,
                    "category": "number",
                    "text": f"номер {identifier}",
                    "normalized_gold": str(identifier),
                    "stressed": f"номер {identifier}",
                },
                ensure_ascii=False,
            )
            + "\n"
            for identifier in range(2_000)
        ),
        encoding="utf-8",
    )
    config = BalalaikaConfig.model_validate(
        {
            "data": data.model_dump(),
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
            "hub": {"local_dir": hub_root, "gigaam_repo_id": "istupakov/gigaam-v3-onnx"},
            "runtime": {"seed": 123, "accumulation": 1, "workers": 0},
        }
    )
    audit = build_index(data, synthetic_corpus.expectations)
    create_selection_manifests(audit.index_path, benchmark, config.selection_dir, config.runtime.seed)

    pins = {}
    for name, kind, repo_id in (
        ("model", "model", config.hub.model_repo_id),
        ("benchmark", "dataset", config.hub.benchmark_repo_id),
        ("gigaam", "model", config.hub.gigaam_repo_id),
    ):
        local_dir = hub_root / name
        local_dir.mkdir(parents=True, exist_ok=True)
        if name != "benchmark":
            (local_dir / "artifact.bin").write_bytes(name.encode())
        files = {
            path.relative_to(local_dir).as_posix(): sha256_file(path)
            for path in sorted(local_dir.rglob("*"))
            if path.is_file()
        }
        pins[name] = {
            "kind": kind,
            "repo_id": repo_id,
            "revision": f"{name}-revision",
            "local_dir": str(local_dir.resolve()),
            "files": files,
        }
    (hub_root / "hub-pins.json").write_text(json.dumps(pins), encoding="utf-8")
    return config


def _refresh_declared_index_hash(config: BalalaikaConfig) -> None:
    index_path = config.data.index_dir / "balalaika-index.sqlite3"
    audit_path = config.data.index_dir / "balalaika-index-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["index_sha256"] = sha256_file(index_path)
    audit_path.write_text(json.dumps(audit), encoding="utf-8")


def test_benchmark_loader_accepts_pinned_rows_with_additional_metadata(tmp_path):
    """Catches the production benchmark's descriptive fields being splatted into the scoring dataclass."""
    benchmark = tmp_path / "hard_number_eval.jsonl"
    benchmark.write_text(
        "".join(
            json.dumps(
                {
                    "id": identifier,
                    "category": "agree",
                    "hard_number": "1004",
                    "why_hard": "agreement",
                    "text": f"номер {identifier}",
                    "normalized_runorm": str(identifier),
                    "normalized_gold": str(identifier),
                    "runorm_wrong": False,
                    "error_type": "",
                    "stressed": f"номер {identifier}",
                }
            )
            + "\n"
            for identifier in range(1, 2_001)
        ),
        encoding="utf-8",
    )

    rows = workflow_module._load_benchmark_rows(benchmark)

    assert len(rows) == 2_000
    assert rows[0].id == 1
    assert rows[-1].id == 2_000


def test_deep_audit_rejects_sqlite_stage_tampering_with_preserved_declarations(prepared_deep_audit):
    """Catches audit trusting declared stage totals and a refreshed self-declared index hash."""
    commands = ProductionCommands()
    assert commands.audit(prepared_deep_audit).values["status"] == "complete"
    index_path = prepared_deep_audit.data.index_dir / "balalaika-index.sqlite3"
    with sqlite3.connect(index_path) as database:
        sample_id = database.execute("SELECT sample_id FROM samples WHERE stage = 1 LIMIT 1").fetchone()[0]
        database.execute("UPDATE samples SET stage = 2, agreement = 0.99 WHERE sample_id = ?", (sample_id,))
    _refresh_declared_index_hash(prepared_deep_audit)

    with pytest.raises(ValueError, match="stage-1 row count"):
        commands.audit(prepared_deep_audit)


def test_deep_audit_rejects_selection_content_with_unchanged_declared_fingerprint(prepared_deep_audit):
    """Catches audit trusting a shared selection fingerprint without recomputing manifest content."""
    commands = ProductionCommands()
    path = prepared_deep_audit.selection_dir / "prompts.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["prompts"][0]["text"] = "tampered but same count"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="selection fingerprint"):
        commands.audit(prepared_deep_audit)


def test_deep_audit_rejects_ineligible_memorization_identity_before_fingerprint(prepared_deep_audit):
    """Catches four unique declared rows bypassing the stage-2 high-agreement eligibility gate."""
    commands = ProductionCommands()
    path = prepared_deep_audit.selection_dir / "memorization.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["memorization"][0]["stage"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="stage-2 high-agreement"):
        commands.audit(prepared_deep_audit)


def test_deep_audit_rejects_assignment_to_unknown_prompt(prepared_deep_audit):
    """Catches cardinality-only assignment validation accepting a nonexistent prompt reference."""
    commands = ProductionCommands()
    path = prepared_deep_audit.selection_dir / "benchmark-prompts.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["benchmark_prompt_by_id"]["0"] = "prompt-does-not-exist"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt reference"):
        commands.audit(prepared_deep_audit)


def test_deep_audit_rejects_changed_extracted_wav(prepared_deep_audit):
    """Catches audit accepting a selected audio artifact whose manifest hash is stale."""
    commands = ProductionCommands()
    manifest = json.loads((prepared_deep_audit.selection_dir / "memorization.json").read_text(encoding="utf-8"))
    wav_path = Path(manifest["memorization"][0]["wav_path"])
    wav_path.write_bytes(wav_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="WAV SHA-256"):
        commands.audit(prepared_deep_audit)


def test_deep_audit_rehashes_current_corpus_sources(prepared_deep_audit):
    """Catches audit accepting an index after an immutable source archive changes in place."""
    commands = ProductionCommands()
    source_tar = next((prepared_deep_audit.data.corpus_root / "train").glob("shard_*.tar"))
    source_tar.write_bytes(source_tar.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="source tar SHA-256"):
        commands.audit(prepared_deep_audit)


def test_deep_audit_rejects_changed_pinned_file(prepared_deep_audit):
    """Catches deep audit consuming a locally changed model after immutable pinning."""
    commands = ProductionCommands()
    (prepared_deep_audit.hub.local_dir / "model" / "artifact.bin").write_bytes(b"changed")

    with pytest.raises(ValueError, match="changed after download"):
        commands.audit(prepared_deep_audit)


@pytest.mark.parametrize("trainer_fails", [False, True])
def test_configured_training_finishes_tracking_then_closes_runtime_once(tmp_path, monkeypatch, trainer_fails):
    """Catches production training leaking Accelerate or closing it before W&B cleanup on either exit path."""
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": tmp_path / "corpus", "index_dir": tmp_path / "index"},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
            "hub": {"local_dir": tmp_path / "hub", "gigaam_repo_id": "gigaam/repo"},
            "wandb": {"group": "qualification"},
        }
    )
    events: list[str] = []

    class Runtime:
        rank = 0

        def barrier(self):
            events.append("barrier")

        def close(self):
            events.append("close")
            if trainer_fails:
                raise RuntimeError("runtime close failed")

    class RunManager:
        def finish(self):
            events.append("finish")
            if trainer_fails:
                raise RuntimeError("tracking finish failed")

    class Trainer:
        def __init__(self, *_args, **_kwargs):
            events.append("trainer-init")

        def run_stage(self, _stage):
            events.append("trainer-run")
            if trainer_fails:
                raise RuntimeError("training failed")
            return tmp_path / "checkpoint"

    monkeypatch.setattr(workflow_module.AccelerateRuntime, "create", lambda _config: Runtime())
    monkeypatch.setattr(
        workflow_module,
        "_verified_pins",
        lambda _config: {
            "model": {"revision": "model-revision"},
            "gigaam": {"revision": "gigaam-revision", "local_dir": str(tmp_path / "gigaam")},
        },
    )
    monkeypatch.setattr(workflow_module, "_index_fingerprint", lambda _config: "index-fingerprint")
    monkeypatch.setattr(
        workflow_module,
        "_load_selection",
        lambda _root: SimpleNamespace(fingerprint="selection-fingerprint"),
    )
    monkeypatch.setattr(workflow_module, "create_run_manager", lambda **_kwargs: RunManager())
    monkeypatch.setattr(workflow_module, "_read_json", lambda *_args: {"run_id": "run-123"})
    monkeypatch.setattr(workflow_module, "_load_benchmark_rows", lambda _path: ())
    monkeypatch.setattr(workflow_module, "BalalaikaTrainer", Trainer)

    if trainer_fails:
        with pytest.raises(RuntimeError, match="training failed"):
            workflow_module._run_configured_training(
                config, stage=2, smoke=False, resume=None, stage1_checkpoint=tmp_path / "stage1"
            )
    else:
        assert (
            workflow_module._run_configured_training(
                config, stage=2, smoke=False, resume=None, stage1_checkpoint=tmp_path / "stage1"
            )
            == tmp_path / "checkpoint"
        )

    assert events[-2:] == ["finish", "close"]
    assert events.count("close") == 1


def test_configured_training_closes_runtime_when_tracking_setup_fails(tmp_path, monkeypatch):
    """Catches a runtime leak before a W&B manager exists."""
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": tmp_path / "corpus", "index_dir": tmp_path / "index"},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
            "hub": {"local_dir": tmp_path / "hub", "gigaam_repo_id": "gigaam/repo"},
            "wandb": {"group": "qualification"},
        }
    )
    events: list[str] = []

    class Runtime:
        rank = 0

        def close(self):
            events.append("close")

    monkeypatch.setattr(workflow_module.AccelerateRuntime, "create", lambda _config: Runtime())
    monkeypatch.setattr(
        workflow_module,
        "_verified_pins",
        lambda _config: {
            "model": {"revision": "model-revision"},
            "gigaam": {"revision": "gigaam-revision", "local_dir": str(tmp_path / "gigaam")},
        },
    )
    monkeypatch.setattr(workflow_module, "_index_fingerprint", lambda _config: "index-fingerprint")
    monkeypatch.setattr(
        workflow_module,
        "_load_selection",
        lambda _root: SimpleNamespace(fingerprint="selection-fingerprint"),
    )
    monkeypatch.setattr(
        workflow_module,
        "create_run_manager",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tracking setup failed")),
    )

    with pytest.raises(RuntimeError, match="tracking setup failed"):
        workflow_module._run_configured_training(
            config, stage=2, smoke=False, resume=None, stage1_checkpoint=tmp_path / "stage1"
        )

    assert events == ["close"]
