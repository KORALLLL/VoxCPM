"""Durable, authorised per-item validation state for distributed evaluation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import wave

import pytest

from voxcpm.training.balalaika import artifacts as artifacts_module
from voxcpm.training.balalaika.artifacts import atomic_json, sha256_file
from voxcpm.training.balalaika.ledger import ValidationLedger, ValidationLedgerError


def _write_wav(path: Path, value: bytes = b"\x00\x00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(value * 40)


def _inputs(item_id: int) -> dict[str, object]:
    return {"prompt_id": item_id % 20, "stressed": f"число {item_id}"}


def _seed(item_id: int) -> int:
    return 10_000 + item_id


def _record(ledger: ValidationLedger, item_id: int) -> dict[str, object]:
    return json.loads(ledger.item_path(item_id).read_text(encoding="utf-8"))


def _claim(ledger: ValidationLedger, item_id: int, *, rank: int = 0, inputs=None, attempt_seed=None):
    claim = ledger.claim(
        item_id,
        rank=rank,
        inputs=_inputs(item_id) if inputs is None else inputs,
        attempt_seed=_seed(item_id) if attempt_seed is None else attempt_seed,
    )
    assert claim is not None
    return claim


def _complete(ledger: ValidationLedger, item_id: int, *, rank: int = 0, claim=None) -> Path:
    claim = _claim(ledger, item_id, rank=rank) if claim is None else claim
    wav_path = ledger.wav_path(item_id, rank=rank)
    _write_wav(wav_path)
    ledger.record_generation(item_id, claim=claim, wav_sha256=sha256_file(wav_path), path=wav_path)
    ledger.record_asr(item_id, claim=claim, hypothesis="текст")
    return wav_path


def test_ledger_reuses_only_hash_and_current_input_matching_complete_item(tmp_path: Path):
    """Catches stale generation, ASR, prompt, or seed inputs reusing a transcript."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    _complete(ledger, 7)

    assert ledger.is_complete(7, inputs=_inputs(7), attempt_seed=_seed(7))
    assert not ledger.is_complete(7, inputs={"prompt_id": 99}, attempt_seed=_seed(7))
    assert not ledger.is_complete(7, inputs=_inputs(7), attempt_seed=999)
    assert ledger.complete_ids({7: (_inputs(7), _seed(7))}) == {7}
    assert not ValidationLedger(tmp_path, "g2", "a1").is_complete(7, inputs=_inputs(7), attempt_seed=_seed(7))
    assert not ValidationLedger(tmp_path, "g1", "a2").is_complete(7, inputs=_inputs(7), attempt_seed=_seed(7))


def test_ledger_persists_atomic_per_item_inputs_attempts_hashes_timestamps_and_claim_epoch(tmp_path: Path):
    """Catches an interrupted item update exposing partial JSON or losing replay inputs."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    inputs = {"prompt": "p-1", "text": "число"}
    claim = _claim(ledger, 5, rank=1, inputs=inputs, attempt_seed=9005)
    wav_path = ledger.wav_path(5, rank=1)
    _write_wav(wav_path)
    wav_hash = sha256_file(wav_path)
    ledger.record_generation(5, claim=claim, wav_sha256=wav_hash, path=wav_path, elapsed_seconds=0.25)
    ledger.record_asr(5, claim=claim, hypothesis="пять", elapsed_seconds=0.5)

    record = _record(ledger, 5)
    assert ledger.item_path(5).name == "00005.json"
    assert record["fingerprints"] == {"asr": "a1", "generation": "g1"}
    assert record["inputs"] == inputs
    assert record["attempt_seed"] == 9005
    assert record["attempts"] == {"asr": 1, "generation": 1}
    assert record["claim_epoch"] == 1
    assert record["owner"] is None
    assert record["generation"]["wav_path"] == str(wav_path)
    assert record["generation"]["wav_sha256"] == wav_hash
    assert record["asr"]["hypothesis"] == "пять"
    assert len(record["asr"]["hypothesis_sha256"]) == 64
    assert record["created_at"].endswith("Z")
    assert record["updated_at"].endswith("Z")
    assert not list((tmp_path / "items").glob(".*.tmp"))


def test_ledger_rejects_a_mutation_without_a_live_claim(tmp_path: Path):
    """Catches a caller publishing generated audio without atomically owning the item."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    wav_path = ledger.wav_path(6)
    _write_wav(wav_path)

    with pytest.raises(ValidationLedgerError, match="claim"):
        ledger.record_generation(6, claim=None, wav_sha256=sha256_file(wav_path), path=wav_path)


def test_ledger_recovers_after_atomic_replace_failure(tmp_path: Path, monkeypatch):
    """Catches a disk/interruption error replacing a record from corrupting its last good state."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    claim = _claim(ledger, 3)
    wav_path = ledger.wav_path(3)
    _write_wav(wav_path)
    ledger.record_generation(3, claim=claim, wav_sha256=sha256_file(wav_path), path=wav_path)
    before = ledger.item_path(3).read_bytes()
    replace_file = artifacts_module.os.replace

    def fail_replace(source, destination):
        if Path(destination) == ledger.item_path(3):
            raise OSError("injected replacement failure")
        return replace_file(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        ledger.record_asr(3, claim=claim, hypothesis="три")

    assert ledger.item_path(3).read_bytes() == before
    assert not list((tmp_path / "items").glob(".*.tmp"))


def test_ledger_invalidates_missing_or_hash_mismatched_wav_before_reuse(tmp_path: Path):
    """Catches a transcript being reused after its waveform is deleted or replaced."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    wav_path = _complete(ledger, 9)
    wav_path.unlink()

    assert not ledger.is_complete(9, inputs=_inputs(9), attempt_seed=_seed(9))
    claim = _claim(ledger, 9)
    assert _record(ledger, 9)["generation"]["status"] == "pending"

    _complete(ledger, 9, claim=claim)
    wav_path.write_bytes(b"not the recorded waveform")
    assert not ledger.is_complete(9, inputs=_inputs(9), attempt_seed=_seed(9))


def test_ledger_current_input_check_rejects_changed_prompt_or_seed_without_claim(tmp_path: Path):
    """Catches a later evaluator querying stale completion before it calls claim()."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    _complete(ledger, 13)

    assert not ledger.is_complete(13, inputs={"prompt_id": 2}, attempt_seed=_seed(13))
    assert not ledger.is_complete(13, inputs=_inputs(13), attempt_seed=13)


def test_ledger_rejects_a_record_with_tampered_input_fingerprint(tmp_path: Path):
    """Catches hand-edited deterministic inputs bypassing a seemingly complete item record."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    _complete(ledger, 14)
    record = _record(ledger, 14)
    record["inputs"] = {"prompt_id": 999}
    atomic_json(ledger.item_path(14), record)

    assert not ledger.is_complete(14, inputs=_inputs(14), attempt_seed=_seed(14))


def test_ledger_caps_retries_and_keeps_the_last_exception_durable(tmp_path: Path):
    """Catches persistent failures being retried forever or replaced by a fabricated transcript."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", max_attempts=2)

    first = _claim(ledger, 21)
    ledger.record_failure(21, claim=first, stage="generation", exception=RuntimeError("first failure"))
    second = _claim(ledger, 21)
    ledger.record_failure(21, claim=second, stage="generation", exception=RuntimeError("second failure"))

    record = _record(ledger, 21)
    assert record["attempts"]["generation"] == 2
    assert record["generation"]["status"] == "failed"
    assert record["generation"]["exception"] == {"message": "second failure", "type": "RuntimeError"}
    assert record["asr"]["hypothesis"] is None
    assert ledger.claim(21, inputs=_inputs(21), attempt_seed=_seed(21)) is None
    with pytest.raises(ValidationLedgerError, match="claim"):
        ledger.record_failure(21, claim=second, stage="generation", exception=RuntimeError("third failure"))
    assert _record(ledger, 21)["attempts"]["generation"] == 2


def test_ledger_enforces_retry_cap_even_if_a_record_was_interrupted_while_pending(tmp_path: Path):
    """Catches a crash-shaped pending record exceeding its stage cap on a resumed mutation."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", max_attempts=1)
    claim = _claim(ledger, 211)
    record = _record(ledger, 211)
    record["attempts"]["generation"] = 1
    atomic_json(ledger.item_path(211), record)

    with pytest.raises(ValidationLedgerError, match="retry cap"):
        ledger.record_failure(211, claim=claim, stage="generation", exception=RuntimeError("over cap"))

    assert _record(ledger, 211)["attempts"]["generation"] == 1


def test_ledger_enforces_stage_order_and_immutable_successes(tmp_path: Path):
    """Catches ASR failures before generation and mutations that overwrite completed results."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    claim = _claim(ledger, 22)

    with pytest.raises(ValidationLedgerError, match="successful generation"):
        ledger.record_failure(22, claim=claim, stage="asr", exception=RuntimeError("out of order"))
    wav_path = ledger.wav_path(22)
    _write_wav(wav_path)
    ledger.record_generation(22, claim=claim, wav_sha256=sha256_file(wav_path), path=wav_path)
    with pytest.raises(ValidationLedgerError, match="generation state"):
        ledger.record_generation(22, claim=claim, wav_sha256=sha256_file(wav_path), path=wav_path)
    ledger.record_asr(22, claim=claim, hypothesis="двадцать два")
    with pytest.raises(ValidationLedgerError, match="claim"):
        ledger.record_failure(22, claim=claim, stage="asr", exception=RuntimeError("overwrite"))


def test_ledger_rejects_rejected_and_stale_claimants_from_writing(tmp_path: Path):
    """Catches a rank that lost ownership overwriting the current owner's durable state."""
    rank_zero = ValidationLedger(
        tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", claim_timeout_seconds=3600
    )
    rank_one = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", claim_timeout_seconds=3600)
    first = _claim(rank_zero, 31, rank=0)
    assert rank_one.claim(31, rank=1, inputs=_inputs(31), attempt_seed=_seed(31)) is None
    rejected = replace(first, rank=1)

    with pytest.raises(ValidationLedgerError, match="owner"):
        rank_one.record_failure(31, claim=rejected, stage="generation", exception=RuntimeError("rejected"))
    stale = _record(rank_zero, 31)
    stale["owner"]["claimed_at"] = "2000-01-01T00:00:00Z"
    atomic_json(rank_zero.item_path(31), stale)
    takeover = _claim(rank_one, 31, rank=1)

    with pytest.raises(ValidationLedgerError, match="claim"):
        rank_zero.record_failure(31, claim=first, stage="generation", exception=RuntimeError("stale"))
    rank_one.record_failure(31, claim=takeover, stage="generation", exception=RuntimeError("current"))
    assert _record(rank_one, 31)["generation"]["exception"]["message"] == "current"


def test_ledger_uses_ranked_wav_paths_and_allows_disjoint_ranks_to_publish(tmp_path: Path):
    """Catches ranks colliding on WAV names or per-rank records hiding global IDs."""
    rank_zero = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    rank_one = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    assert rank_zero.wav_path(40, rank=0) != rank_one.wav_path(40, rank=1)
    assert rank_zero.wav_path(40, rank=0).name == rank_one.wav_path(40, rank=1).name
    _complete(rank_zero, 40, rank=0)
    _complete(rank_one, 41, rank=1)

    assert rank_zero.complete_ids({40: (_inputs(40), _seed(40)), 41: (_inputs(41), _seed(41))}) == {40, 41}


def test_complete_ids_uses_current_input_callback_for_each_item(tmp_path: Path):
    """Catches bulk completion accepting all persisted inputs instead of evaluator-current assignments."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    _complete(ledger, 50)
    _complete(ledger, 51)

    def expected(item_id: int):
        if item_id == 50:
            return _inputs(item_id), _seed(item_id)
        return {"prompt_id": 999}, _seed(item_id)

    assert ledger.complete_ids(expected) == {50}
