"""Distributed generation, transcription, publication, and retention."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest
import torch

from voxcpm.training.balalaika import evaluation as evaluation_module
from voxcpm.training.balalaika.artifacts import atomic_json, fingerprint, sha256_file
from voxcpm.training.balalaika.evaluation import (
    DistributedEvaluator,
    EvaluationIntegrityError,
    IncompleteValidation,
    partition_ids,
    require_exact_complete,
)
from voxcpm.training.balalaika.ledger import ValidationLedger
from voxcpm.training.balalaika.metrics import BenchmarkRow


def _write_wav(path: Path, *, value: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(int(value).to_bytes(2, "little", signed=True) * 80)


class FakeRuntime:
    def __init__(self, *, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device("cpu")
        self.barriers = 0
        self.gather_calls = 0

    def unwrap(self, model):
        return model

    def barrier(self) -> None:
        self.barriers += 1

    def gather(self, value):
        self.gather_calls += 1
        return value


class FakeVAE(torch.nn.Module):
    pass


class FakeModel(torch.nn.Module):
    sample_rate = 16_000

    def __init__(self, *, failures: dict[str, int] | None = None):
        super().__init__()
        self.audio_vae = None
        self.failures = dict(failures or {})
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        assert self.audio_vae is not None
        self.calls.append(dict(kwargs))
        target = str(kwargs["target_text"])
        if self.failures.get(target, 0):
            self.failures[target] -= 1
            raise RuntimeError(f"injected generation failure for {target}")
        return np.full(80, 0.01 * (len(self.calls) + 1), dtype=np.float32)


class FakeASR:
    def __init__(self, events: list[str], *, hypotheses: dict[int, str] | None = None):
        self.events = events
        self.hypotheses = dict(hypotheses or {})

    def transcribe(self, wav_path: Path) -> str:
        item_id = int(Path(wav_path).stem)
        self.events.append(f"asr:{item_id}")
        return self.hypotheses.get(item_id, f"номер {item_id}")

    def close(self) -> None:
        self.events.append("asr-close")


class FakeRunManager:
    def __init__(self, events: list[str], *, fail: bool = False):
        self.events = events
        self.fail = fail
        self.payloads: list[object] = []

    def log_validation(self, payload: object, *, global_step: int) -> None:
        self.events.append(f"wandb:{global_step}")
        if self.fail:
            raise RuntimeError("injected W&B failure")
        self.payloads.append(payload)


@dataclass
class Fixture:
    evaluator: DistributedEvaluator
    runtime: FakeRuntime
    ledger: ValidationLedger
    model: FakeModel
    vae: FakeVAE
    events: list[str]
    manager: FakeRunManager
    rows: tuple[BenchmarkRow, ...]
    selection: object
    checkpoint: dict[str, str]
    boundary: dict[str, object]


def _fake_payload(**values: object) -> SimpleNamespace:
    return SimpleNamespace(**values)


def _fixture(
    tmp_path: Path,
    *,
    item_count: int = 4,
    rank: int = 0,
    world_size: int = 1,
    boundary_index: int = 1,
    global_step: int = 10,
    failures: dict[str, int] | None = None,
    fail_tracking: bool = False,
) -> Fixture:
    prompt_path = tmp_path / "selection" / "prompt.wav"
    _write_wav(prompt_path, value=7)
    prompt = SimpleNamespace(
        prompt_id="prompt-00",
        text="фиксированный промпт",
        wav_path=prompt_path,
        wav_sha256=sha256_file(prompt_path),
    )
    rows = tuple(
        BenchmarkRow(
            id=item_id,
            category="number" if item_id % 2 == 0 else "date",
            text=f"номер {item_id}",
            normalized_gold=f"номер {item_id}",
            stressed=f"но́мер {item_id}",
        )
        for item_id in range(item_count)
    )
    audio_ids = list(range(min(4, item_count)))
    if len(audio_ids) < 4:
        raise ValueError("test evaluator fixtures require at least four items")
    selection = SimpleNamespace(
        fingerprint="selection-fingerprint",
        prompts=[prompt],
        benchmark_prompt_by_id={item.id: prompt.prompt_id for item in rows},
        audio_log_ids=audio_ids,
    )
    runtime = FakeRuntime(rank=rank, world_size=world_size)
    generation_settings = {"cfg_value": 3.5, "inference_timesteps": 17, "max_length": 777}
    ledger = ValidationLedger(
        tmp_path / f"boundary-{boundary_index:02d}",
        generation_fingerprint=fingerprint(generation_settings),
        asr_fingerprint="asr-fingerprint",
        max_attempts=3,
        claim_timeout_seconds=3_600,
    )
    events: list[str] = []
    manager = FakeRunManager(events, fail=fail_tracking)
    evaluator = DistributedEvaluator(
        runtime=runtime,
        rows=rows,
        selection=selection,
        ledger=ledger,
        asr_factory=lambda _device_id: FakeASR(events, hypotheses={0: ""}),
        run_manager=manager,
        expected_item_count=item_count,
        validation_root=tmp_path,
        payload_factory=_fake_payload,
        generation_settings=generation_settings,
    )
    return Fixture(
        evaluator=evaluator,
        runtime=runtime,
        ledger=ledger,
        model=FakeModel(failures=failures),
        vae=FakeVAE(),
        events=events,
        manager=manager,
        rows=rows,
        selection=selection,
        checkpoint={"checkpoint_fingerprint": "checkpoint-fingerprint"},
        boundary={
            "stage": "stage1",
            "epoch": 0,
            "boundary": boundary_index,
            "global_step": global_step,
            "stage_progress": boundary_index / 8,
        },
    )


def _publish_without_retention(fixture: Fixture) -> None:
    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    payload = fixture.evaluator.aggregate(fixture.checkpoint, fixture.boundary)
    context = fixture.evaluator._context(fixture.checkpoint, fixture.boundary)
    atomic_json(
        fixture.ledger.root / "validation-complete.json",
        fixture.evaluator._completion_record(context, payload),
    )


def _rewrite_completion(fixture: Fixture, mutate) -> None:
    path = fixture.ledger.root / "validation-complete.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    mutate(record)
    identity = {key: value for key, value in record.items() if key not in {"snapshot_fingerprint", "published_at"}}
    record["snapshot_fingerprint"] = evaluation_module.fingerprint(identity)
    atomic_json(path, record)


def test_partition_sorted_ids_by_position_is_disjoint_and_complete() -> None:
    """Catches value-based sharding, overlap, omissions, or input-order dependence."""
    ids = tuple(range(1, 2_001))
    partitions = [partition_ids(reversed(ids), rank=rank, world_size=8) for rank in range(8)]

    assert partitions[0][:3] == (1, 9, 17)
    assert partitions[7][-3:] == (1_984, 1_992, 2_000)
    assert set().union(*map(set, partitions)) == set(ids)
    assert sum(map(len, partitions)) == 2_000


def test_exact_completion_rejects_1999_of_2000_and_unexpected_ids() -> None:
    """Catches rank zero publishing a partial or substituted benchmark boundary."""
    with pytest.raises(IncompleteValidation, match="1999/2000"):
        require_exact_complete(range(2_000), range(1_999))
    with pytest.raises(IncompleteValidation, match="missing=.*1999.*unexpected=.*2000"):
        require_exact_complete(range(2_000), [*range(1_999), 2_000])


def test_generation_uses_current_prompt_seed_claim_and_restores_model_after_asr(tmp_path: Path) -> None:
    """Catches wrong generation arguments, claimless writes, leaked VAE/mode, or ASR teardown after training resumes."""
    fixture = _fixture(tmp_path)
    fixture.vae.train()
    fixture.model.train()
    original_train = fixture.model.train

    def record_train(mode: bool = True):
        fixture.events.append(f"model-mode:{mode}")
        return original_train(mode)

    fixture.model.train = record_train  # type: ignore[method-assign]
    completed = fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert completed == {0, 1, 2, 3}
    assert [call["target_text"] for call in fixture.model.calls] == [row.stressed for row in fixture.rows]
    assert {call["prompt_text"] for call in fixture.model.calls} == {"фиксированный промпт"}
    assert {call["prompt_wav_path"] for call in fixture.model.calls} == {str(fixture.selection.prompts[0].wav_path)}
    assert [call["seed"] for call in fixture.model.calls] == [
        fixture.evaluator.item_seed(row.id, fixture.checkpoint, fixture.boundary) for row in fixture.rows
    ]
    assert fixture.model.audio_vae is None
    assert fixture.model.training is True
    assert fixture.vae.training is True
    assert fixture.events.index("asr-close") < fixture.events.index("model-mode:True")
    first = json.loads(fixture.ledger.item_path(0).read_text(encoding="utf-8"))
    assert first["claim_epoch"] == 1
    assert first["asr"]["status"] == "success"
    assert first["asr"]["hypothesis"] == ""


def test_generation_retries_are_bounded_and_reuse_the_deterministic_seed(tmp_path: Path) -> None:
    """Catches retry drift, unbounded generation loops, or failure records that lose the final success."""
    fixture = _fixture(tmp_path, failures={"но́мер 0": 2})

    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    calls = [call for call in fixture.model.calls if call["target_text"] == "но́мер 0"]
    assert len(calls) == 3
    assert len({call["seed"] for call in calls}) == 1
    record = json.loads(fixture.ledger.item_path(0).read_text(encoding="utf-8"))
    assert record["attempts"]["generation"] == 3
    assert record["generation"]["status"] == "success"


def test_asr_reacquires_an_expired_generation_claim_before_transcription(tmp_path: Path, monkeypatch) -> None:
    """Catches ASR mutating with the generation token after its lease expired between phases."""
    fixture = _fixture(tmp_path)
    generate_local = fixture.evaluator._generate_local

    def generate_then_expire(model, local_ids, context):
        claims = generate_local(model, local_ids, context)
        for item_id in local_ids:
            record = json.loads(fixture.ledger.item_path(item_id).read_text(encoding="utf-8"))
            record["owner"]["claimed_at"] = "2000-01-01T00:00:00Z"
            atomic_json(fixture.ledger.item_path(item_id), record)
        return claims

    monkeypatch.setattr(fixture.evaluator, "_generate_local", generate_then_expire)

    completed = fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert completed == {0, 1, 2, 3}
    for item_id in completed:
        record = json.loads(fixture.ledger.item_path(item_id).read_text(encoding="utf-8"))
        assert record["claim_epoch"] == 2
        assert record["asr"]["status"] == "success"


def test_generation_and_complete_resume_restore_identical_cpu_rng(tmp_path: Path, monkeypatch) -> None:
    """Catches fresh validation advancing training RNG while an already-complete resume does not."""
    fixture = _fixture(tmp_path)
    generate = fixture.model.generate

    def random_generate(**kwargs):
        torch.rand(7)
        return generate(**kwargs)

    monkeypatch.setattr(fixture.model, "generate", random_generate)
    torch.manual_seed(8_113)
    expected = torch.rand(6)

    torch.manual_seed(8_113)
    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    fresh_continuation = torch.rand(6)

    torch.manual_seed(8_113)
    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    resumed_continuation = torch.rand(6)

    assert torch.equal(fresh_continuation, expected)
    assert torch.equal(resumed_continuation, expected)


def test_rng_is_restored_when_asr_teardown_raises(tmp_path: Path, monkeypatch) -> None:
    """Catches exceptional validation leaking CPU or rank-local CUDA RNG into resumed training."""
    fixture = _fixture(tmp_path)
    fixture.runtime.device = torch.device("cuda", 0)
    cuda_before = torch.tensor([19, 23, 29], dtype=torch.uint8)
    restored_cuda: list[tuple[torch.Tensor, torch.device]] = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_before.clone())
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: restored_cuda.append((state.clone(), torch.device(device))),
    )

    class RaisingCloseASR(FakeASR):
        def transcribe(self, wav_path: Path) -> str:
            torch.rand(3)
            return super().transcribe(wav_path)

        def close(self) -> None:
            torch.rand(5)
            super().close()
            raise RuntimeError("injected ASR close failure")

    fixture.evaluator.asr_factory = lambda _device_id: RaisingCloseASR(fixture.events)
    torch.manual_seed(9_007)
    cpu_before = torch.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="ASR close failure"):
        fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert torch.equal(torch.get_rng_state(), cpu_before)
    assert len(restored_cuda) == 1
    assert torch.equal(restored_cuda[0][0], cuda_before)
    assert restored_cuda[0][1] == torch.device("cuda", 0)


def test_resume_reuses_only_current_hash_matching_items(tmp_path: Path) -> None:
    """Catches a completed WAV being regenerated, or stale row/WAV content being reused."""
    fixture = _fixture(tmp_path)
    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    fixture.model.calls.clear()

    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    assert fixture.model.calls == []

    fixture.ledger.wav_path(2, rank=0).write_bytes(b"tampered")
    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    assert [call["target_text"] for call in fixture.model.calls] == ["но́мер 2"]


def test_generation_passes_all_configured_voxcpm2_arguments_and_binds_identity(tmp_path: Path) -> None:
    """Catches generation defaults replacing configured VoxCPM2 inference settings or using max_length."""
    fixture = _fixture(tmp_path)

    fixture.evaluator.run_rank(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert fixture.model.calls
    for call in fixture.model.calls:
        assert call["cfg_value"] == 3.5
        assert call["inference_timesteps"] == 17
        assert call["max_len"] == 777
        assert "max_length" not in call
    context = fixture.evaluator._context(fixture.checkpoint, fixture.boundary)
    assert context.item_inputs[0]["generation_settings"] == {
        "cfg_value": 3.5,
        "inference_timesteps": 17,
        "max_length": 777,
    }


def test_rank_status_write_failure_reaches_outcome_collective_before_error(tmp_path: Path, monkeypatch) -> None:
    """Catches one rank leaving before peers when its durable status file cannot be written."""
    fixture = _fixture(tmp_path)
    real_atomic_json = evaluation_module.atomic_json

    def fail_rank_status(path: Path, value: dict[str, object]) -> None:
        if Path(path).parent.name == "rank-status":
            raise OSError("injected rank status write failure")
        real_atomic_json(path, value)

    monkeypatch.setattr(evaluation_module, "atomic_json", fail_rank_status)

    with pytest.raises(RuntimeError, match="status write failure|status publication"):
        fixture.evaluator.run(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert fixture.runtime.gather_calls >= 1


def test_full_run_scores_empty_asr_logs_then_completes_in_fixed_audio_order(tmp_path: Path, monkeypatch) -> None:
    """Catches empty ASR being dropped, W&B/completion inversion, or drift in the four fixed audio IDs."""
    fixture = _fixture(tmp_path)
    real_atomic_json = evaluation_module.atomic_json

    def recording_atomic_json(path: Path, value: dict[str, object]) -> None:
        if Path(path).name == "validation-complete.json":
            fixture.events.append("boundary-complete")
        real_atomic_json(path, value)

    monkeypatch.setattr(evaluation_module, "atomic_json", recording_atomic_json)
    payload = fixture.evaluator.run(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert payload.metrics.item_count == 4
    assert payload.items[0].asr_hypothesis == ""
    assert payload.items[0].score.utt_word_deletions == 2
    assert tuple(payload.audio_paths) == (0, 1, 2, 3)
    assert fixture.events.index("wandb:10") < fixture.events.index("boundary-complete")
    assert fixture.runtime.barriers == 2
    complete = json.loads((fixture.ledger.root / "validation-complete.json").read_text(encoding="utf-8"))
    assert complete["status"] == "complete"
    assert complete["item_count"] == 4


def test_wandb_failure_prevents_boundary_completion_and_retention(tmp_path: Path) -> None:
    """Catches a failed tracking transaction being treated as a durable evaluator boundary."""
    fixture = _fixture(tmp_path, fail_tracking=True)

    with pytest.raises(RuntimeError, match="W&B failure"):
        fixture.evaluator.run(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)

    assert not (fixture.ledger.root / "validation-complete.json").exists()
    assert (fixture.ledger.root / "validation-failed.json").is_file()
    assert (fixture.ledger.root / "wavs").is_dir()
    assert not (tmp_path / "retained-audio" / fixture.ledger.root.name).exists()


def test_retention_prunes_only_older_completed_full_wavs_after_new_completion(tmp_path: Path) -> None:
    """Catches eager pruning, loss of fixed examples, or deletion of an incomplete boundary."""
    first = _fixture(tmp_path, boundary_index=1, global_step=10)
    first.evaluator.run(first.model, first.vae, first.checkpoint, first.boundary)
    incomplete_wavs = tmp_path / "boundary-incomplete" / "wavs"
    _write_wav(incomplete_wavs / "00000.wav")

    second = _fixture(tmp_path, boundary_index=2, global_step=20)
    second.evaluator.run(second.model, second.vae, second.checkpoint, second.boundary)

    assert not (first.ledger.root / "wavs").exists()
    assert (second.ledger.root / "wavs").is_dir()
    assert incomplete_wavs.is_dir()
    for boundary in (first.ledger.root.name, second.ledger.root.name):
        retained = tmp_path / "retained-audio" / boundary
        assert sorted(path.name for path in retained.glob("*.wav")) == [
            "00000.wav",
            "00001.wav",
            "00002.wav",
            "00003.wav",
        ]


def test_retention_rejects_a_historical_noncanonical_snapshot_before_pruning(tmp_path: Path) -> None:
    """Catches retention trusting a completion whose canonical self-fingerprint changed."""
    first = _fixture(tmp_path, boundary_index=1, global_step=10)
    first.evaluator.run(first.model, first.vae, first.checkpoint, first.boundary)
    second = _fixture(tmp_path, boundary_index=2, global_step=20)
    _publish_without_retention(second)
    completion_path = first.ledger.root / "validation-complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["snapshot_fingerprint"] = "corrupt-snapshot"
    atomic_json(completion_path, completion)

    with pytest.raises(EvaluationIntegrityError, match="snapshot fingerprint"):
        second.evaluator._apply_retention()

    assert (first.ledger.root / "wavs").is_dir()
    assert (second.ledger.root / "wavs").is_dir()


def test_retention_rejects_recanonicalized_wrong_audio_cardinality_before_pruning(tmp_path: Path) -> None:
    """Catches an attacker self-signing three historical audio entries and triggering deletion."""
    first = _fixture(tmp_path, boundary_index=1, global_step=10)
    first.evaluator.run(first.model, first.vae, first.checkpoint, first.boundary)
    second = _fixture(tmp_path, boundary_index=2, global_step=20)
    _publish_without_retention(second)
    _rewrite_completion(first, lambda record: record.__setitem__("audio", record["audio"][:3]))

    with pytest.raises(EvaluationIntegrityError, match="exactly four"):
        second.evaluator._apply_retention()

    assert (first.ledger.root / "wavs").is_dir()
    assert (second.ledger.root / "wavs").is_dir()


def test_retention_rejects_corrupt_source_even_when_prior_retained_copy_is_valid(tmp_path: Path) -> None:
    """Catches a retained copy masking corruption in the full WAV tree about to be deleted."""
    first = _fixture(tmp_path, boundary_index=1, global_step=10)
    first.evaluator.run(first.model, first.vae, first.checkpoint, first.boundary)
    second = _fixture(tmp_path, boundary_index=2, global_step=20)
    _publish_without_retention(second)
    first.ledger.wav_path(0, rank=0).write_bytes(b"corrupt historical source")

    with pytest.raises(EvaluationIntegrityError, match="source audio"):
        second.evaluator._apply_retention()

    assert (first.ledger.root / "wavs").is_dir()
    assert (second.ledger.root / "wavs").is_dir()


def test_retention_stages_all_four_audio_files_before_atomic_publication_and_pruning(
    tmp_path: Path, monkeypatch
) -> None:
    """Catches a third-copy failure publishing a partial retained boundary or deleting full WAVs."""
    first = _fixture(tmp_path, boundary_index=1, global_step=10)
    first.evaluator.run(first.model, first.vae, first.checkpoint, first.boundary)
    second = _fixture(tmp_path, boundary_index=2, global_step=20)
    _publish_without_retention(second)
    copy = evaluation_module._atomic_copy
    copy_count = 0

    def fail_third_copy(source: Path, destination: Path) -> None:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 3:
            raise OSError("injected retained-copy failure")
        copy(source, destination)

    monkeypatch.setattr(evaluation_module, "_atomic_copy", fail_third_copy)

    with pytest.raises(OSError, match="retained-copy failure"):
        second.evaluator._apply_retention()

    assert not (tmp_path / "retained-audio" / second.ledger.root.name).exists()
    assert (first.ledger.root / "wavs").is_dir()
    assert (second.ledger.root / "wavs").is_dir()


def test_completed_boundary_rejects_changed_aggregate_hash_on_resume(tmp_path: Path) -> None:
    """Catches completion records trusting aggregate files whose durable content changed."""
    fixture = _fixture(tmp_path)
    fixture.evaluator.run(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)
    (fixture.ledger.root / "items.jsonl").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(EvaluationIntegrityError, match="items.jsonl"):
        fixture.evaluator.run(fixture.model, fixture.vae, fixture.checkpoint, fixture.boundary)


def test_resume_retention_failure_crosses_the_same_barrier_and_reaches_every_rank(tmp_path: Path, monkeypatch) -> None:
    """Catches rank zero raising before the resume barrier while a worker remains blocked or returns success."""
    completed = _fixture(tmp_path)
    completed.evaluator.run(completed.model, completed.vae, completed.checkpoint, completed.boundary)
    completed.runtime.world_size = 2
    worker_runtime = FakeRuntime(rank=1, world_size=2)
    worker = DistributedEvaluator(
        runtime=worker_runtime,
        rows=completed.rows,
        selection=completed.selection,
        ledger=completed.ledger,
        asr_factory=lambda _device_id: FakeASR(completed.events),
        run_manager=FakeRunManager(completed.events),
        expected_item_count=4,
        validation_root=tmp_path,
        payload_factory=_fake_payload,
        generation_settings={"cfg_value": 3.5, "inference_timesteps": 17, "max_length": 777},
    )

    def fail_retention() -> None:
        raise EvaluationIntegrityError("injected historical retention corruption")

    monkeypatch.setattr(completed.evaluator, "_apply_retention", fail_retention)

    with pytest.raises(EvaluationIntegrityError, match="historical retention corruption"):
        completed.evaluator.run(completed.model, completed.vae, completed.checkpoint, completed.boundary)
    with pytest.raises(EvaluationIntegrityError, match="historical retention corruption"):
        worker.run(FakeModel(), FakeVAE(), completed.checkpoint, completed.boundary)

    assert completed.runtime.barriers == 3  # two initial-run barriers plus one resume-status barrier
    assert worker_runtime.barriers == 1
    status = json.loads((completed.ledger.root / "resume-retention-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error"]["type"] == "EvaluationIntegrityError"


def test_resume_status_write_failure_is_gathered_before_any_rank_enters_barrier(tmp_path: Path, monkeypatch) -> None:
    """Catches rank zero leaving on status publication while a resume worker waits at the barrier."""
    completed = _fixture(tmp_path)
    completed.evaluator.run(completed.model, completed.vae, completed.checkpoint, completed.boundary)

    class CoordinatedFailureRuntime(FakeRuntime):
        def gather(self, value):
            self.gather_calls += 1
            return torch.tensor([0, 1], dtype=value.dtype, device=value.device)

    main_runtime = CoordinatedFailureRuntime(rank=0, world_size=2)
    worker_runtime = CoordinatedFailureRuntime(rank=1, world_size=2)
    completed.evaluator.runtime = main_runtime
    worker = DistributedEvaluator(
        runtime=worker_runtime,
        rows=completed.rows,
        selection=completed.selection,
        ledger=completed.ledger,
        asr_factory=lambda _device_id: FakeASR(completed.events),
        run_manager=FakeRunManager(completed.events),
        expected_item_count=4,
        validation_root=tmp_path,
        payload_factory=_fake_payload,
        generation_settings={"cfg_value": 3.5, "inference_timesteps": 17, "max_length": 777},
    )
    real_atomic_json = evaluation_module.atomic_json

    def fail_resume_status(path: Path, value: dict[str, object]) -> None:
        if Path(path).name == "resume-retention-status.json":
            raise OSError("injected resume status write failure")
        real_atomic_json(path, value)

    monkeypatch.setattr(evaluation_module, "atomic_json", fail_resume_status)

    for evaluator, model, vae in (
        (completed.evaluator, completed.model, completed.vae),
        (worker, FakeModel(), FakeVAE()),
    ):
        with pytest.raises(RuntimeError, match="resume retention status publication failed"):
            evaluator.run(model, vae, completed.checkpoint, completed.boundary)

    assert main_runtime.gather_calls == worker_runtime.gather_calls == 1
    assert main_runtime.barriers == worker_runtime.barriers == 0
    assert not (completed.ledger.root / "resume-retention-status.json").exists()
