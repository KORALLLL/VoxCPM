from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import torch

import voxcpm.training.balalaika.workflow as workflow_module
from voxcpm.training.balalaika.artifacts import fingerprint, sha256_file
from voxcpm.training.balalaika.config import BalalaikaConfig, DataConfig
from voxcpm.training.balalaika.index import IndexAudit, build_index
from voxcpm.training.balalaika.selection import create_selection_manifests
from voxcpm.training.balalaika.workflow import ProductionCommands


def _generation_marker(root: Path, role: str, generation_id: str = "previous") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / ".balalaika-generation").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "owner": "voxcpm-balalaika",
                "role": role,
                "canonical_root": str(root.resolve()),
                "generation_id": generation_id,
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def prepared_deep_audit(synthetic_corpus, tmp_path) -> BalalaikaConfig:
    identities = [f"{index // 23:06d}/item-{index:02d}.wav" for index in range(45)]
    synthetic_corpus.source_rows = [
        {"source_relative_path": identity, "shard": identity.split("/", maxsplit=1)[0], "include_audio": True}
        for identity in identities
    ]
    synthetic_corpus.rover_rows = [
        {
            "source_relative_path": identity,
            "asr_agreement_mean": 0.99 if index < 30 else (0.5 if index < 44 else None),
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
        source_row_count=45,
        rover_row_count=45,
        combined_row_count=45,
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
        json.dumps({"status": "ok", "stats": {"samples": 45}, "shards": shard_rows}), encoding="utf-8"
    )
    combined_metadata = synthetic_corpus.root / "combined.meta.json"
    combined_metadata.write_text(
        json.dumps(
            {
                "complete": True,
                "rows": 45,
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
        expected_rows=45,
        expected_null_agreement=1,
        expected_stage1_rows=14,
        expected_stage2_rows=30,
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


def _replace_with_self_consistent_noncanonical_selection(config: BalalaikaConfig) -> None:
    """Replace every chosen row while preserving all old audit invariants."""
    root = config.selection_dir
    manifests = {
        name: json.loads((root / filename).read_text(encoding="utf-8"))
        for name, filename in (
            ("memorization", "memorization.json"),
            ("prompts", "prompts.json"),
            ("assignments", "benchmark-prompts.json"),
            ("audio_ids", "audio-log-ids.json"),
        )
    }
    old_memorization = manifests["memorization"]["memorization"]
    old_prompts = manifests["prompts"]["prompts"]
    excluded_memorization = {item["source_relative_path"] for item in old_memorization}
    excluded_prompts = {item["source_relative_path"] for item in old_prompts}
    index_path = config.data.index_dir / "balalaika-index.sqlite3"
    with sqlite3.connect(index_path) as database:
        database.row_factory = sqlite3.Row
        sidecar_path = Path(database.execute("SELECT value FROM metadata WHERE key = 'sidecar_path'").fetchone()[0])
        index_fingerprint = database.execute("SELECT value FROM metadata WHERE key = 'fingerprint'").fetchone()[0]
        memorization_rows = [row for row in database.execute("""
                SELECT source_relative_path, agreement, stage, sidecar_offset, sidecar_size
                FROM samples WHERE stage = 2 AND agreement >= 0.95
                ORDER BY source_relative_path
                """) if row["source_relative_path"] not in excluded_memorization][:4]
        prompt_rows = [row for row in database.execute("""
                SELECT source_relative_path, agreement, stage, sidecar_offset, sidecar_size
                FROM samples WHERE stage IN (1, 2)
                ORDER BY source_relative_path
                """) if row["source_relative_path"] not in excluded_prompts][:20]
    assert len(memorization_rows) == 4
    assert len(prompt_rows) == 20

    def replacement(old: dict[str, object], row: sqlite3.Row) -> dict[str, object]:
        with sidecar_path.open("rb") as source:
            source.seek(row["sidecar_offset"])
            sidecar = json.loads(source.read(row["sidecar_size"]))
        return {
            **old,
            "source_relative_path": row["source_relative_path"],
            "agreement": row["agreement"],
            "stage": row["stage"],
            "text": sidecar["rover_punctuated_accented"],
        }

    memorization = [replacement(old, row) for old, row in zip(old_memorization, memorization_rows, strict=True)]
    prompts = [replacement(old, row) for old, row in zip(old_prompts, prompt_rows, strict=True)]
    prompt_ids = [item["prompt_id"] for item in reversed(prompts)]
    assignment_by_id = {identifier: prompt_ids[identifier % len(prompt_ids)] for identifier in range(2_000)}
    audio_log_ids = [3, 503, 1_003, 1_503]

    def sample_fingerprint_value(item: dict[str, object]) -> dict[str, object]:
        value = dict(item)
        value.pop("wav_path")
        return value

    replacement_fingerprint = fingerprint(
        {
            "schema_version": 1,
            "seed": config.runtime.seed,
            "index_fingerprint": index_fingerprint,
            "memorization": [sample_fingerprint_value(item) for item in memorization],
            "prompts": [sample_fingerprint_value(item) for item in prompts],
            "benchmark_prompt_by_id": assignment_by_id,
            "audio_log_ids": audio_log_ids,
        }
    )
    for manifest in manifests.values():
        manifest["fingerprint"] = replacement_fingerprint
    manifests["memorization"]["memorization"] = memorization
    manifests["prompts"]["prompts"] = prompts
    manifests["assignments"]["benchmark_prompt_by_id"] = assignment_by_id
    manifests["audio_ids"]["audio_log_ids"] = audio_log_ids
    for name, filename in (
        ("memorization", "memorization.json"),
        ("prompts", "prompts.json"),
        ("assignments", "benchmark-prompts.json"),
        ("audio_ids", "audio-log-ids.json"),
    ):
        (root / filename).write_text(json.dumps(manifests[name]), encoding="utf-8")


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


def test_deep_audit_rejects_self_consistent_noncanonical_selection(prepared_deep_audit):
    """Catches audit accepting a different eligible bundle whose declarations and hashes were rebuilt."""
    commands = ProductionCommands()
    assert commands.audit(prepared_deep_audit).values["status"] == "complete"
    _replace_with_self_consistent_noncanonical_selection(prepared_deep_audit)

    with pytest.raises(ValueError, match="canonical deterministic selection"):
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


@pytest.mark.parametrize("relation", ["equal", "descendant", "ancestor"])
def test_preparation_rejects_generation_roots_overlapping_immutable_corpus(tmp_path, relation):
    """Catches preparation gaining a recursive rename/delete path into the immutable corpus."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    candidate = {"equal": corpus, "descendant": corpus / "generated", "ancestor": tmp_path}[relation]
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": candidate},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
        }
    )

    with pytest.raises(ValueError, match="immutable corpus"):
        workflow_module._PreparationTransaction(config)


def test_preparation_rejects_symlink_and_nested_generation_roots(tmp_path):
    """Catches resolved symlinks and ancestor roots bypassing destructive-path validation."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-index"
    link.symlink_to(target, target_is_directory=True)
    symlink_config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": link},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
        }
    )
    nested_config = symlink_config.model_copy(
        update={
            "data": symlink_config.data.model_copy(update={"index_dir": tmp_path / "generation"}),
            "selection_dir": tmp_path / "generation" / "selection",
        }
    )

    with pytest.raises(ValueError, match="symlink"):
        workflow_module._PreparationTransaction(symlink_config)
    with pytest.raises(ValueError, match="overlap"):
        workflow_module._PreparationTransaction(nested_config)


def test_preparation_refuses_to_replace_unmarked_existing_directory(tmp_path):
    """Catches arbitrary operator directories being treated as disposable pipeline generations."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    index_root = tmp_path / "index"
    index_root.mkdir()
    (index_root / "operator.txt").write_text("keep", encoding="utf-8")
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": index_root},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
        }
    )

    with pytest.raises(ValueError, match="generation marker"):
        with workflow_module._PreparationTransaction(config):
            pass

    assert (index_root / "operator.txt").read_text(encoding="utf-8") == "keep"


def test_prepare_keeps_previous_generation_live_until_validated_publish(tmp_path, monkeypatch):
    """Catches a long build hiding both live roots before its replacement is ready."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    index_root = tmp_path / "index"
    selection_root = tmp_path / "selection"
    _generation_marker(index_root, "index")
    _generation_marker(selection_root, "selection")
    (index_root / "old.txt").write_text("old-index", encoding="utf-8")
    (selection_root / "old.txt").write_text("old-selection", encoding="utf-8")
    benchmark = tmp_path / "hub" / "benchmark" / "data.jsonl"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("{}\n", encoding="utf-8")
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": index_root},
            "output_dir": tmp_path / "runs",
            "selection_dir": selection_root,
            "hub": {"local_dir": tmp_path / "hub"},
        }
    )

    def build_index(data, _expectations):
        assert (index_root / "old.txt").read_text(encoding="utf-8") == "old-index"
        assert (selection_root / "old.txt").read_text(encoding="utf-8") == "old-selection"
        assert data.index_dir != index_root
        data.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = data.index_dir / "balalaika-index.sqlite3"
        audit_path = data.index_dir / "balalaika-index-audit.json"
        index_path.write_bytes(b"new-index")
        audit_path.write_text("{}", encoding="utf-8")
        return IndexAudit(index_path, audit_path, "index-fingerprint", 45, 44, 14, 30, 1)

    def build_selection(_index, _benchmark, output_dir, _seed):
        assert output_dir != selection_root
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "new.txt").write_text("new-selection", encoding="utf-8")
        return SimpleNamespace(
            fingerprint="selection-fingerprint",
            memorization=[1, 2, 3, 4],
            prompts=list(range(20)),
            benchmark_prompt_by_id={item: "prompt-00" for item in range(2_000)},
        )

    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})
    commands = ProductionCommands(
        expectation_loader=lambda _data: object(),
        index_builder=build_index,
        selection_builder=build_selection,
        generation_validator=lambda *_args: None,
    )

    commands.prepare(config)

    assert not (index_root / "old.txt").exists()
    assert (index_root / "balalaika-index.sqlite3").read_bytes() == b"new-index"
    assert (selection_root / "new.txt").read_text(encoding="utf-8") == "new-selection"


def test_prepare_startup_recovers_interrupted_build_without_hiding_live_generation(tmp_path, monkeypatch):
    """Catches host-loss staging residue blocking restart or replacing the last complete generation."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    index_root = tmp_path / "index"
    selection_root = tmp_path / "selection"
    _generation_marker(index_root, "index")
    _generation_marker(selection_root, "selection")
    (index_root / "old.txt").write_text("old-index", encoding="utf-8")
    (selection_root / "old.txt").write_text("old-selection", encoding="utf-8")
    output = tmp_path / "runs"
    output.mkdir()
    transaction_id = "interrupted"
    stage_index = tmp_path / f".index.generation-{transaction_id}"
    stage_selection = tmp_path / f".selection.generation-{transaction_id}"
    for root, role, canonical in (
        (stage_index, "index", index_root),
        (stage_selection, "selection", selection_root),
    ):
        root.mkdir()
        (root / ".balalaika-generation").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "owner": "voxcpm-balalaika",
                    "role": role,
                    "canonical_root": str(canonical.resolve()),
                    "generation_id": transaction_id,
                    "status": "building",
                }
            ),
            encoding="utf-8",
        )
        (root / "partial.txt").write_text("partial", encoding="utf-8")
    journal = {
        "schema_version": 1,
        "owner": "voxcpm-balalaika",
        "transaction_id": transaction_id,
        "phase": "building",
        "roots": {
            "index": {
                "live": str(index_root.resolve()),
                "stage": str(stage_index.resolve()),
                "backup": str((tmp_path / f".index.previous-{transaction_id}").resolve()),
            },
            "selection": {
                "live": str(selection_root.resolve()),
                "stage": str(stage_selection.resolve()),
                "backup": str((tmp_path / f".selection.previous-{transaction_id}").resolve()),
            },
        },
    }
    (output / ".prepare-transaction.json").write_text(json.dumps(journal), encoding="utf-8")
    benchmark = tmp_path / "hub" / "benchmark" / "data.jsonl"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text("{}\n", encoding="utf-8")
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": index_root},
            "output_dir": output,
            "selection_dir": selection_root,
            "hub": {"local_dir": tmp_path / "hub"},
        }
    )
    monkeypatch.setattr(workflow_module, "_verified_pins", lambda _config: {})
    commands = ProductionCommands(
        expectation_loader=lambda _data: object(),
        index_builder=lambda *_args: (_ for _ in ()).throw(RuntimeError("stop after recovery")),
    )

    with pytest.raises(RuntimeError, match="stop after recovery"):
        commands.prepare(config)

    assert not stage_index.exists()
    assert not stage_selection.exists()
    assert not (output / ".prepare-transaction.json").exists()
    assert (index_root / "old.txt").read_text(encoding="utf-8") == "old-index"
    assert (selection_root / "old.txt").read_text(encoding="utf-8") == "old-selection"


def test_preparation_journals_before_creating_any_staging_directory(tmp_path, monkeypatch):
    """Catches SIGKILL leaving an unjournaled sibling generation before recovery can identify it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": tmp_path / "index"},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
        }
    )
    observed: dict[str, bool] = {}

    def stop_at_journal(transaction, phase):
        observed["stage_exists"] = any(path.exists() for path in transaction._stage.values())
        raise RuntimeError(f"stop at {phase} journal")

    monkeypatch.setattr(workflow_module._PreparationTransaction, "_write_journal", stop_at_journal)

    with pytest.raises(RuntimeError, match="stop at building journal"):
        with workflow_module._PreparationTransaction(config):
            pass

    assert observed == {"stage_exists": False}
    assert not tuple(tmp_path.glob(".*.generation-*"))


def test_preparation_startup_finishes_interrupted_publication_before_new_build(tmp_path):
    """Catches a host loss between the two generation renames leaving a mixed live pair."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    index_root = tmp_path / "index"
    selection_root = tmp_path / "selection"
    output = tmp_path / "runs"
    output.mkdir()
    transaction_id = "publishing"
    stage_index = tmp_path / f".index.generation-{transaction_id}"
    stage_selection = tmp_path / f".selection.generation-{transaction_id}"
    backup_index = tmp_path / f".index.previous-{transaction_id}"
    backup_selection = tmp_path / f".selection.previous-{transaction_id}"

    def marker(root: Path, role: str, generation: str, status: str = "complete") -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / ".balalaika-generation").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "owner": "voxcpm-balalaika",
                    "role": role,
                    "canonical_root": str((index_root if role == "index" else selection_root).resolve()),
                    "generation_id": generation,
                    "status": status,
                }
            ),
            encoding="utf-8",
        )

    marker(index_root, "index", transaction_id)
    (index_root / "new.txt").write_text("new-index", encoding="utf-8")
    marker(backup_index, "index", "previous")
    (backup_index / "old.txt").write_text("old-index", encoding="utf-8")
    marker(selection_root, "selection", "previous")
    (selection_root / "old.txt").write_text("old-selection", encoding="utf-8")
    marker(stage_selection, "selection", transaction_id)
    (stage_selection / "new.txt").write_text("new-selection", encoding="utf-8")
    journal = {
        "schema_version": 1,
        "owner": "voxcpm-balalaika",
        "transaction_id": transaction_id,
        "phase": "publishing",
        "roots": {
            "index": {
                "live": str(index_root.resolve()),
                "stage": str(stage_index.resolve()),
                "backup": str(backup_index.resolve()),
            },
            "selection": {
                "live": str(selection_root.resolve()),
                "stage": str(stage_selection.resolve()),
                "backup": str(backup_selection.resolve()),
            },
        },
    }
    (output / ".prepare-transaction.json").write_text(json.dumps(journal), encoding="utf-8")
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": corpus, "index_dir": index_root},
            "output_dir": output,
            "selection_dir": selection_root,
        }
    )

    with pytest.raises(RuntimeError, match="stop new build"):
        with workflow_module._PreparationTransaction(config):
            raise RuntimeError("stop new build")

    assert (index_root / "new.txt").read_text(encoding="utf-8") == "new-index"
    assert (selection_root / "new.txt").read_text(encoding="utf-8") == "new-selection"
    assert not backup_index.exists()
    assert not backup_selection.exists()
    assert not (output / ".prepare-transaction.json").exists()


def test_precommit_validation_enforces_configured_exact_counts(prepared_deep_audit):
    """Catches builder declarations bypassing the independently configured count gate."""
    summary = ProductionCommands().audit(prepared_deep_audit).values
    audit = IndexAudit(
        prepared_deep_audit.data.index_dir / "balalaika-index.sqlite3",
        prepared_deep_audit.data.index_dir / "balalaika-index-audit.json",
        summary["fingerprint"],
        summary["joined_rows"],
        summary["eligible_rows"],
        summary["stage1_rows"],
        summary["stage2_rows"],
        summary["excluded_null_agreement"],
    )
    selection = workflow_module._load_selection(prepared_deep_audit.selection_dir)
    wrong = prepared_deep_audit.model_copy(
        update={
            "data": prepared_deep_audit.data.model_copy(
                update={"expected_stage1_rows": prepared_deep_audit.data.expected_stage1_rows - 1}
            )
        }
    )

    with pytest.raises(ValueError, match="stage-1 row count"):
        workflow_module._validate_prepared_generation(wrong, audit, selection)


@pytest.mark.parametrize("tamper", ["sqlite", "selection"])
def test_memorize_verifies_complete_prepared_generation_before_runtime(prepared_deep_audit, tamper):
    """Catches self-declared prepared artifacts reaching model/runtime setup after tampering."""
    if tamper == "sqlite":
        index_path = prepared_deep_audit.data.index_dir / "balalaika-index.sqlite3"
        index_path.write_bytes(index_path.read_bytes() + b"tampered")
    else:
        path = prepared_deep_audit.selection_dir / "prompts.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["prompts"][0]["text"] = "tampered canonical content"
        path.write_text(json.dumps(value), encoding="utf-8")
    runtime_calls = []
    commands = ProductionCommands(
        runtime_factory=lambda _config: runtime_calls.append("runtime") or SimpleNamespace(close=lambda: None),
        memorization_runner=lambda *_args: SimpleNamespace(
            status="complete",
            result_path=Path("unused"),
            checkpoint=Path("unused"),
            wandb_run_id="unused",
            large_training_started=False,
        ),
    )

    with pytest.raises(ValueError, match="index|selection|SHA-256|hash"):
        commands.memorize(prepared_deep_audit, smoke=False)

    assert runtime_calls == []


def test_production_microbatch_selector_probes_longest_rows_and_honors_fixed_bypass(tmp_path, monkeypatch):
    """Catches production leaving candidate/fixed microbatch configuration disconnected from startup probing."""
    index_root = tmp_path / "index"
    index_root.mkdir()
    with sqlite3.connect(index_root / "balalaika-index.sqlite3") as database:
        database.executescript("""
            CREATE TABLE samples (sample_id INTEGER PRIMARY KEY, source_relative_path TEXT, duration REAL);
            CREATE TABLE stage_ordinals (stage INTEGER, ordinal INTEGER, sample_id INTEGER);
            """)
        database.executemany(
            "INSERT INTO samples VALUES (?, ?, ?)",
            [(1, "short.wav", 1.0), (2, "long.wav", 9.0)],
        )
        database.executemany("INSERT INTO stage_ordinals VALUES (1, ?, ?)", [(0, 1), (1, 2)])
    config = BalalaikaConfig.model_validate(
        {
            "data": {"corpus_root": tmp_path / "corpus", "index_dir": index_root},
            "output_dir": tmp_path / "runs",
            "selection_dir": tmp_path / "selection",
            "runtime": {"batch_candidates": [1, 2], "accumulation": 1},
        }
    )
    monkeypatch.setattr(workflow_module, "VoxCPMCollator", lambda: lambda rows: torch.tensor(rows, dtype=torch.float32))

    class Runtime:
        device = torch.device("cpu")

        @staticmethod
        def unwrap(model):
            return model

        @staticmethod
        def gather(value):
            return value

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value, *, progress):
            del progress
            return {"loss/diff": self.lora_A * value.sum()}

    model = Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    dataset = [1.0, 9.0]
    selector = workflow_module._production_microbatch_selector(config, 1)

    assert selector(Runtime(), model, optimizer, config.stage1, dataset, lambda batch: {"value": batch}) == 2

    fixed = config.model_copy(update={"runtime": config.runtime.model_copy(update={"fixed_microbatch": 3})})
    bypass = workflow_module._production_microbatch_selector(fixed, 1)
    assert bypass(Runtime(), model, optimizer, fixed.stage1, object(), lambda _batch: {}) == 3


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
            assert callable(_kwargs["microbatch_selector"])
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
        "_verify_current_prepared_generation",
        lambda _config: ("index-fingerprint", SimpleNamespace(fingerprint="selection-fingerprint")),
    )
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
        "_verify_current_prepared_generation",
        lambda _config: ("index-fingerprint", SimpleNamespace(fingerprint="selection-fingerprint")),
    )
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
