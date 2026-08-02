"""Rank-zero W&B boundary logging for the Balalaika evaluator."""

from __future__ import annotations

import builtins
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from voxcpm.training.balalaika import tracking as tracking_module
from voxcpm.training.balalaika.artifacts import sha256_file
from voxcpm.training.balalaika.metrics import BenchmarkRow, aggregate_scores, score_utterance
from voxcpm.training.balalaika.tracking import (
    NullRunManager,
    ValidationItem,
    ValidationPayload,
    WandbRunManager,
    create_run_manager,
)


class FakeTable:
    def __init__(self, *, columns: list[str], data: list[list[object]]):
        self.columns = columns
        self.data = data


class FakeAudio:
    def __init__(self, path: str, *, caption: str):
        self.path = path
        self.caption = caption


class FakeRun:
    def __init__(self, directory: Path, *, fail_log: bool = False):
        self.dir = str(directory)
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fail_log = fail_log
        self.logged: list[dict[str, object]] = []
        self.finish_calls = 0

    def log(self, payload: dict[str, object], *, step: int | None = None) -> None:
        if self.fail_log:
            raise RuntimeError("injected W&B upload failure")
        self.logged.append({**payload, "_step": step})

    def finish(self) -> None:
        self.finish_calls += 1


class FakeWandb:
    Table = FakeTable
    Audio = FakeAudio

    def __init__(self, directory: Path, *, fail_log: bool = False):
        self.directory = directory
        self.fail_log = fail_log
        self.init_calls: list[dict[str, object]] = []
        self.run: FakeRun | None = None
        self.on_init = None

    def init(self, **kwargs: object) -> FakeRun:
        if self.on_init is not None:
            self.on_init()
        self.init_calls.append(kwargs)
        self.run = FakeRun(self.directory, fail_log=self.fail_log)
        return self.run


def wandb_config(tmp_path: Path, *, mode: str = "offline") -> dict[str, object]:
    return {
        "project": "voxcpm-balalaika",
        "entity": "acme",
        "mode": mode,
        "group": "experiment-17",
        "dir": tmp_path / "wandb",
        "config": {"learning_rate": 1e-4},
        "config_fingerprint": "config-sha",
    }


def validation_payload(tmp_path: Path) -> ValidationPayload:
    items: list[ValidationItem] = []
    for item_id in range(2_000):
        row = BenchmarkRow(
            id=item_id,
            category="date" if item_id % 2 else "money",
            text=f"Код {item_id + 1}.",
            normalized_gold=f"Код {item_id + 1}.",
            stressed=f"Ко\u0301д {item_id + 1}.",
        )
        items.append(
            ValidationItem(
                row=row,
                score=score_utterance(row, row.normalized_gold),
                prompt_id=f"prompt-{item_id % 20:02d}",
            )
        )
    audio_paths: dict[int, Path] = {}
    for item_id in (11, 29, 301, 1701):
        path = tmp_path / "audio" / f"{item_id:05d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFfake-wav")
        audio_paths[item_id] = path
    metrics = aggregate_scores([item.score for item in items])
    return ValidationPayload(
        metrics=metrics,
        category_metrics=metrics.category_metrics,
        items=items,
        audio_paths=audio_paths,
        artifact_dir=tmp_path / "artifacts",
        timings={"generation_seconds": 12.5, "asr_seconds": 2.5},
        failure_counts={"generation": 0, "asr": 0},
        stage_progress=0.375,
        input_fingerprint="inputs-sha",
        expected_audio_ids=(11, 29, 301, 1701),
        selection_fingerprint="selection-sha",
    )


def rebuild_payload(payload: ValidationPayload, **changes: object) -> ValidationPayload:
    values: dict[str, object] = {
        "metrics": payload.metrics,
        "category_metrics": payload.category_metrics,
        "items": payload.items,
        "audio_paths": payload.audio_paths,
        "artifact_dir": payload.artifact_dir,
        "timings": payload.timings,
        "failure_counts": payload.failure_counts,
        "stage_progress": payload.stage_progress,
        "input_fingerprint": payload.input_fingerprint,
        "expected_audio_ids": payload.expected_audio_ids,
        "selection_fingerprint": payload.selection_fingerprint,
    }
    values.update(changes)
    return ValidationPayload(**values)


def test_validation_payload_rejects_wrong_boundary_cardinalities(tmp_path: Path) -> None:
    """Catches an evaluator handing W&B a partial 2,000-item boundary or wrong audio set."""
    payload = validation_payload(tmp_path)

    with pytest.raises(ValueError, match="exactly 2,000"):
        ValidationPayload(
            payload.metrics,
            payload.category_metrics,
            payload.items[:-1],
            payload.audio_paths,
            payload.artifact_dir,
            expected_audio_ids=payload.expected_audio_ids,
            selection_fingerprint=payload.selection_fingerprint,
        )
    with pytest.raises(ValueError, match="exactly four"):
        ValidationPayload(
            payload.metrics,
            payload.category_metrics,
            payload.items,
            dict(list(payload.audio_paths.items())[:3]),
            payload.artifact_dir,
            expected_audio_ids=payload.expected_audio_ids[:3],
            selection_fingerprint=payload.selection_fingerprint,
        )


def test_validation_payload_rejects_audio_ids_reordered_or_substituted(tmp_path: Path) -> None:
    """Catches W&B examples drifting from the Task-5 selection-owned ordered ID contract."""
    payload = validation_payload(tmp_path)
    reversed_paths = dict(reversed(tuple(payload.audio_paths.items())))
    substituted_paths = {7: next(iter(payload.audio_paths.values())), **dict(list(payload.audio_paths.items())[1:])}

    with pytest.raises(ValueError, match="ordered expected audio IDs"):
        rebuild_payload(payload, audio_paths=reversed_paths)
    with pytest.raises(ValueError, match="ordered expected audio IDs"):
        rebuild_payload(payload, audio_paths=substituted_paths)


def test_validation_payload_recomputes_item_and_category_aggregates(tmp_path: Path) -> None:
    """Catches a caller supplying aggregate totals or category totals that contradict the 2,000 scored rows."""
    payload = validation_payload(tmp_path)
    wrong_metrics = replace(payload.metrics, utt_word_deletions=1)
    wrong_categories = {
        **payload.category_metrics,
        "date": replace(payload.category_metrics["date"], item_count=999),
    }

    with pytest.raises(ValueError, match="aggregate metrics"):
        rebuild_payload(payload, metrics=wrong_metrics)
    with pytest.raises(ValueError, match="category metrics"):
        rebuild_payload(payload, category_metrics=wrong_categories)


def test_validation_payload_requires_normalized_gold_linkage(tmp_path: Path) -> None:
    """Catches an audit row whose benchmark gold text no longer matches the score that W&B will display."""
    payload = validation_payload(tmp_path)
    mismatched = replace(payload.items[0], row=replace(payload.items[0].row, normalized_gold="Другая норма"))

    with pytest.raises(ValueError, match="normalized gold"):
        rebuild_payload(payload, items=(mismatched, *payload.items[1:]))


def test_run_id_is_persisted_before_wandb_init_and_resumed(tmp_path: Path) -> None:
    """Catches an interrupted run creating a new W&B identity after its first upload starts."""
    state_path = tmp_path / "state" / "stage1.json"
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    seen_during_init: list[dict[str, object]] = []
    fake_wandb.on_init = lambda: seen_during_init.append(json.loads(state_path.read_text(encoding="utf-8")))

    first = WandbRunManager.start("stage1", state_path, wandb_config(tmp_path), wandb_module=fake_wandb)
    second = WandbRunManager.start("stage1", state_path, wandb_config(tmp_path), wandb_module=fake_wandb)

    assert seen_during_init[0]["run_id"] == first.run_id
    assert first.run_id == second.run_id
    assert all(call["resume"] == "allow" for call in fake_wandb.init_calls)
    assert all(call["group"] == "experiment-17" for call in fake_wandb.init_calls)
    assert all(call["job_type"] == "stage1" for call in fake_wandb.init_calls)


def test_run_state_rejects_a_changed_config_before_wandb_init(tmp_path: Path) -> None:
    """Catches a resumed run silently mixing a new training configuration into its old W&B identity."""
    state_path = tmp_path / "state.json"
    first_wandb = FakeWandb(tmp_path / "first-run")
    WandbRunManager.start("stage1", state_path, wandb_config(tmp_path), wandb_module=first_wandb)
    changed_config = {**wandb_config(tmp_path), "config_fingerprint": "different-config-sha"}
    resumed_wandb = FakeWandb(tmp_path / "resumed-run")

    with pytest.raises(RuntimeError, match="config fingerprint"):
        WandbRunManager.start("stage1", state_path, changed_config, wandb_module=resumed_wandb)

    assert resumed_wandb.init_calls == []


def test_job_types_share_group_but_use_distinct_persisted_run_ids(tmp_path: Path) -> None:
    """Catches memorization/stage runs being accidentally merged into one resumable W&B run."""
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    config = wandb_config(tmp_path)
    managers = [
        WandbRunManager.start(kind, tmp_path / f"{kind}.json", config, wandb_module=fake_wandb)
        for kind in ("memorization", "stage1", "stage2")
    ]

    assert len({manager.run_id for manager in managers}) == 3
    assert [call["job_type"] for call in fake_wandb.init_calls] == ["memorization", "stage1", "stage2"]
    assert {call["group"] for call in fake_wandb.init_calls} == {"experiment-17"}


@pytest.mark.parametrize("mode", ["online", "offline", "disabled"])
def test_configured_wandb_mode_is_forwarded_without_network_test_dependency(tmp_path: Path, mode: str) -> None:
    """Catches the run lifecycle overriding an operator-selected W&B transport mode."""
    fake_wandb = FakeWandb(tmp_path / f"{mode}-run")

    WandbRunManager.start(
        "stage1", tmp_path / f"{mode}.json", wandb_config(tmp_path, mode=mode), wandb_module=fake_wandb
    )

    assert fake_wandb.init_calls[0]["mode"] == mode


def test_non_main_process_is_a_strict_wandb_noop(tmp_path: Path, monkeypatch) -> None:
    """Catches worker ranks importing W&B or requiring credentials when they only generate validation rows."""
    original_import = builtins.__import__

    def reject_wandb(name: str, *args: Any, **kwargs: Any):
        if name == "wandb":
            raise AssertionError("non-main rank imported wandb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_wandb)
    manager = create_run_manager(is_main_process=False, config=wandb_config(tmp_path))

    assert isinstance(manager, NullRunManager)
    manager.log_train({"loss": 1.0}, global_step=1)
    manager.log_validation(validation_payload(tmp_path), global_step=1)
    manager.finish()


def test_validation_logs_required_metrics_table_and_fixed_audio_order(tmp_path: Path) -> None:
    """Catches a complete boundary omitting its metrics, audit rows, or deterministic audio examples."""
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    payload = validation_payload(tmp_path)
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)

    manager.log_validation(payload, global_step=10)

    assert fake_wandb.run is not None
    logged = fake_wandb.run.logged[-1]
    assert {"val/num_cer", "val/num_wer", "val/utt_cer", "val/utt_wer"} <= logged.keys()
    assert logged["val/category/date/count"] == 1_000
    assert logged["val/timing/generation_seconds"] == 12.5
    assert logged["val/failures/asr"] == 0
    assert logged["stage/progress"] == 0.375
    assert logged["train/global_step"] == 10
    assert logged["config/fingerprint"] == "config-sha"
    assert logged["input/fingerprint"] == "inputs-sha"
    table = logged["val/items"]
    assert isinstance(table, FakeTable)
    assert len(table.data) == 2_000
    assert {
        "id",
        "category",
        "input",
        "normalized_gold",
        "stressed",
        "prompt_id",
        "asr_hypothesis",
        "gold_number_span",
        "hypothesis_number_span",
        "num_word_errors",
        "utt_word_errors",
    } <= set(table.columns)
    assert [audio.path for audio in logged["val/examples"]] == [str(path) for path in payload.audio_paths.values()]
    assert [audio.caption.split(" | ")[0] for audio in logged["val/examples"]] == [
        f"id={item_id}" for item_id in payload.audio_paths
    ]


def test_validation_table_retains_the_original_asr_hypothesis(tmp_path: Path) -> None:
    """Catches the audit table replacing the recognizer's raw hypothesis with scorer-normalized text."""
    payload = validation_payload(tmp_path)
    raw_hypothesis = "КОД!!! ОДИН"
    items = (replace(payload.items[0], asr_hypothesis=raw_hypothesis), *payload.items[1:])
    payload = rebuild_payload(payload, items=items)
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)

    manager.log_validation(payload, global_step=10)

    assert fake_wandb.run is not None
    table = fake_wandb.run.logged[-1]["val/items"]
    assert isinstance(table, FakeTable)
    assert table.data[0][table.columns.index("asr_hypothesis")] == raw_hypothesis


def test_validation_persists_and_validates_full_snapshot_before_wandb_log(tmp_path: Path, monkeypatch) -> None:
    """Catches W&B being called before the complete local table/audio boundary can be recovered after a crash."""
    payload = validation_payload(tmp_path)
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)
    assert fake_wandb.run is not None
    snapshot_path = payload.artifact_dir / "wandb-boundaries" / "stage1-step-000000010.snapshot.json"
    original_log = fake_wandb.run.log

    def inspect_snapshot_before_log(logged: dict[str, object], *, step: int | None = None) -> None:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert len(snapshot["table"]["rows"]) == 2_000
        assert snapshot["audio"][0] == {
            "id": 11,
            "path": str(payload.audio_paths[11]),
            "sha256": sha256_file(payload.audio_paths[11]),
            "caption": logged["val/examples"][0].caption,
        }
        assert snapshot["selection"] == {"audio_ids": [11, 29, 301, 1701], "fingerprint": "selection-sha"}
        original_log(logged, step=step)

    monkeypatch.setattr(fake_wandb.run, "log", inspect_snapshot_before_log)
    manager.log_validation(payload, global_step=10)

    manifest = json.loads((payload.artifact_dir / "wandb-boundaries" / "stage1-step-000000010.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["snapshot_fingerprint"] == tracking_module.fingerprint(json.loads(snapshot_path.read_text()))


def test_snapshot_log_and_completion_failures_preserve_safe_retry_order(tmp_path: Path, monkeypatch) -> None:
    """Catches snapshot, log, or completion failures reporting a boundary complete without a verified recovery record."""
    payload = validation_payload(tmp_path)
    snapshot_path = payload.artifact_dir / "wandb-boundaries" / "stage1-step-000000010.snapshot.json"
    manifest_path = payload.artifact_dir / "wandb-boundaries" / "stage1-step-000000010.json"
    snapshot_wandb = FakeWandb(tmp_path / "snapshot-run")
    snapshot_manager = WandbRunManager.start(
        "stage1", tmp_path / "snapshot.json", wandb_config(tmp_path), wandb_module=snapshot_wandb
    )
    real_atomic_json = tracking_module.atomic_json

    def fail_snapshot(path: Path, value: dict[str, object]) -> None:
        if path == snapshot_path:
            raise OSError("injected snapshot failure")
        real_atomic_json(path, value)

    monkeypatch.setattr(tracking_module, "atomic_json", fail_snapshot)
    with pytest.raises(OSError, match="snapshot failure"):
        snapshot_manager.log_validation(payload, global_step=10)
    assert not snapshot_path.exists()
    assert not manifest_path.exists()
    assert snapshot_wandb.run is not None and snapshot_wandb.run.logged == []

    monkeypatch.setattr(tracking_module, "atomic_json", real_atomic_json)
    failed_wandb = FakeWandb(tmp_path / "log-run", fail_log=True)
    failed_manager = WandbRunManager.start(
        "stage1", tmp_path / "log.json", wandb_config(tmp_path), wandb_module=failed_wandb
    )
    with pytest.raises(RuntimeError, match="upload failure"):
        failed_manager.log_validation(payload, global_step=10)
    assert snapshot_path.is_file()
    assert json.loads(manifest_path.read_text())["status"] == "pending"

    completion_wandb = FakeWandb(tmp_path / "completion-run")
    completion_manager = WandbRunManager.start(
        "stage1", tmp_path / "log.json", wandb_config(tmp_path), wandb_module=completion_wandb
    )

    def fail_completion(path: Path, value: dict[str, object]) -> None:
        if path == manifest_path and value.get("status") == "complete":
            raise OSError("injected completion failure")
        real_atomic_json(path, value)

    monkeypatch.setattr(tracking_module, "atomic_json", fail_completion)
    with pytest.raises(OSError, match="completion failure"):
        completion_manager.log_validation(payload, global_step=10)
    assert json.loads(manifest_path.read_text())["status"] == "pending"
    assert completion_wandb.run is not None and len(completion_wandb.run.logged) == 1


def test_completed_boundary_rejects_any_changed_full_snapshot_content(tmp_path: Path) -> None:
    """Catches same-step resume silently accepting changed transcript, scores, audio, or training progress."""
    payload = validation_payload(tmp_path)
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)
    manager.log_validation(payload, global_step=10)

    changed_text = rebuild_payload(
        payload, items=(replace(payload.items[0], asr_hypothesis="другой текст"), *payload.items[1:])
    )
    changed_score = replace(payload.items[0].score, utt_word_deletions=payload.items[0].score.utt_word_deletions + 1)
    changed_items = (replace(payload.items[0], score=changed_score), *payload.items[1:])
    changed_metrics = aggregate_scores([item.score for item in changed_items])
    changed_metric = rebuild_payload(
        payload,
        items=changed_items,
        metrics=changed_metrics,
        category_metrics=changed_metrics.category_metrics,
    )
    changed_progress = rebuild_payload(payload, stage_progress=0.5)

    for changed in (changed_text, changed_metric, changed_progress):
        with pytest.raises(RuntimeError, match="different snapshot"):
            manager.log_validation(changed, global_step=10)
    payload.audio_paths[11].write_bytes(b"RIFFchanged-wav")
    with pytest.raises(ValueError, match="content"):
        manager.log_validation(payload, global_step=10)


def test_completed_boundary_rejects_a_corrupt_local_snapshot_on_resume(tmp_path: Path) -> None:
    """Catches a complete manifest trusting a missing or corrupted snapshot instead of retaining retry evidence."""
    payload = validation_payload(tmp_path)
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)
    manager.log_validation(payload, global_step=10)
    snapshot_path = payload.artifact_dir / "wandb-boundaries" / "stage1-step-000000010.snapshot.json"
    snapshot_path.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="snapshot"):
        manager.log_validation(payload, global_step=10)


def test_pending_boundary_rejects_a_different_snapshot_before_retry(tmp_path: Path) -> None:
    """Catches a resumed failed boundary overwriting the durable recovery snapshot with different inputs."""
    payload = validation_payload(tmp_path)
    fake_wandb = FakeWandb(tmp_path / "wandb-run", fail_log=True)
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)
    with pytest.raises(RuntimeError, match="upload failure"):
        manager.log_validation(payload, global_step=10)

    with pytest.raises(RuntimeError, match="different snapshot"):
        manager.log_validation(rebuild_payload(payload, stage_progress=0.5), global_step=10)


def test_offline_boundary_is_marked_complete_only_after_successful_log_and_durable_run_dir(tmp_path: Path) -> None:
    """Catches a failed offline upload being recorded as a complete validation boundary."""
    payload = validation_payload(tmp_path)
    failed_wandb = FakeWandb(tmp_path / "failed-run", fail_log=True)
    manager = WandbRunManager.start(
        "stage1", tmp_path / "failed.json", wandb_config(tmp_path), wandb_module=failed_wandb
    )

    with pytest.raises(RuntimeError, match="upload failure"):
        manager.log_validation(payload, global_step=10)

    manifest = payload.artifact_dir / "wandb-boundaries" / "stage1-step-000000010.json"
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "pending"
    successful_wandb = FakeWandb(tmp_path / "successful-run")
    manager = WandbRunManager.start(
        "stage1", tmp_path / "failed.json", wandb_config(tmp_path), wandb_module=successful_wandb
    )
    manager.log_validation(payload, global_step=10)

    complete = json.loads(manifest.read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    assert complete["mode"] == "offline"
    assert Path(successful_wandb.run.dir).is_dir()  # type: ignore[union-attr]


def test_completed_boundary_is_not_uploaded_again_after_resume(tmp_path: Path) -> None:
    """Catches a resumed offline process duplicating a boundary already marked durable and complete."""
    payload = validation_payload(tmp_path)
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    manager = WandbRunManager.start("stage1", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)

    manager.log_validation(payload, global_step=10)
    manager.log_validation(payload, global_step=10)

    assert fake_wandb.run is not None
    assert len(fake_wandb.run.logged) == 1


def test_finish_is_rank_zero_only_and_idempotent(tmp_path: Path) -> None:
    """Catches W&B runs being finished twice or worker ranks participating in run teardown."""
    fake_wandb = FakeWandb(tmp_path / "wandb-run")
    manager = WandbRunManager.start("stage2", tmp_path / "state.json", wandb_config(tmp_path), wandb_module=fake_wandb)

    manager.finish()
    manager.finish()

    assert fake_wandb.run is not None
    assert fake_wandb.run.finish_calls == 1
