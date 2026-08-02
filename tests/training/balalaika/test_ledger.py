"""Durable, per-item validation state for resumable distributed evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import wave

import pytest

from voxcpm.training.balalaika.artifacts import atomic_json, sha256_file
from voxcpm.training.balalaika import artifacts as artifacts_module
from voxcpm.training.balalaika.ledger import ValidationLedger


def _write_wav(path: Path, value: bytes = b"\x00\x00") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(value * 40)


def _record(ledger: ValidationLedger, item_id: int) -> dict[str, object]:
    return json.loads(ledger.item_path(item_id).read_text(encoding="utf-8"))


def _complete(ledger: ValidationLedger, item_id: int, *, rank: int = 0) -> Path:
    wav_path = ledger.wav_path(item_id, rank=rank)
    _write_wav(wav_path)
    ledger.record_generation(item_id, wav_sha256=sha256_file(wav_path), path=wav_path, attempt_seed=100 + item_id)
    ledger.record_asr(item_id, hypothesis="текст")
    return wav_path


def test_ledger_reuses_only_hash_matching_complete_item(tmp_path: Path):
    """Catches a changed generator/ASR model reusing a stale successful transcript."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    _complete(ledger, 7)

    assert ledger.is_complete(7)
    assert ledger.complete_ids() == {7}
    assert not ValidationLedger(tmp_path, "g2", "a1").is_complete(7)
    assert not ValidationLedger(tmp_path, "g1", "a2").is_complete(7)


def test_ledger_persists_atomic_per_item_inputs_attempts_hashes_and_timestamps(tmp_path: Path):
    """Catches an interrupted item update exposing partial JSON or losing replay inputs."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    assert ledger.claim(5, rank=1, inputs={"prompt": "p-1", "text": "число"}, attempt_seed=9005)
    wav_path = ledger.wav_path(5, rank=1)
    _write_wav(wav_path)
    wav_hash = sha256_file(wav_path)
    ledger.record_generation(5, wav_sha256=wav_hash, path=wav_path, attempt_seed=9005, elapsed_seconds=0.25)
    ledger.record_asr(5, hypothesis="пять", elapsed_seconds=0.5)

    record = _record(ledger, 5)
    assert ledger.item_path(5).name == "00005.json"
    assert record["fingerprints"] == {"asr": "a1", "generation": "g1"}
    assert record["inputs"] == {"prompt": "p-1", "text": "число"}
    assert record["attempt_seed"] == 9005
    assert record["attempts"] == {"asr": 1, "generation": 1}
    assert record["generation"]["wav_path"] == str(wav_path)
    assert record["generation"]["wav_sha256"] == wav_hash
    assert record["asr"]["hypothesis"] == "пять"
    assert len(record["asr"]["hypothesis_sha256"]) == 64
    assert record["created_at"].endswith("Z")
    assert record["updated_at"].endswith("Z")
    assert not list((tmp_path / "items").glob(".*.tmp"))


def test_ledger_records_a_generation_attempt_seed_without_a_prior_claim(tmp_path: Path):
    """Catches direct generation publication dropping the deterministic replay seed."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    wav_path = ledger.wav_path(6)
    _write_wav(wav_path)

    ledger.record_generation(6, wav_sha256=sha256_file(wav_path), path=wav_path, attempt_seed=906)

    record = _record(ledger, 6)
    assert record["attempt_seed"] == 906
    assert record["generation"]["attempt_seed"] == 906


def test_ledger_recovers_after_atomic_replace_failure(tmp_path: Path, monkeypatch):
    """Catches a disk/interruption error replacing a record from corrupting its last good state."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    wav_path = ledger.wav_path(3)
    _write_wav(wav_path)
    ledger.record_generation(3, wav_sha256=sha256_file(wav_path), path=wav_path)
    before = ledger.item_path(3).read_bytes()
    replace = artifacts_module.os.replace

    def fail_replace(source, destination):
        if Path(destination) == ledger.item_path(3):
            raise OSError("injected replacement failure")
        return replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        ledger.record_asr(3, hypothesis="три")

    assert ledger.item_path(3).read_bytes() == before
    assert not list((tmp_path / "items").glob(".*.tmp"))


def test_ledger_invalidates_missing_or_hash_mismatched_wav_before_reuse(tmp_path: Path):
    """Catches a transcript being reused after its waveform is deleted or replaced."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    wav_path = _complete(ledger, 9)
    wav_path.unlink()

    assert not ledger.is_complete(9)
    assert ledger.claim(9)
    assert _record(ledger, 9)["generation"]["status"] == "pending"

    _complete(ledger, 9)
    wav_path.write_bytes(b"not the recorded waveform")
    assert not ledger.is_complete(9)


def test_ledger_invalidates_changed_deterministic_inputs_before_reuse(tmp_path: Path):
    """Catches a changed prompt/seed reusing audio and ASR from a prior assignment."""
    original = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    assert original.claim(13, inputs={"prompt_id": 1}, attempt_seed=13)
    _complete(original, 13)
    resumed = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")

    assert resumed.claim(13, inputs={"prompt_id": 2}, attempt_seed=13)
    record = _record(resumed, 13)
    assert record["generation"]["status"] == "pending"
    assert record["asr"]["status"] == "pending"


def test_ledger_rejects_a_record_with_tampered_input_fingerprint(tmp_path: Path):
    """Catches hand-edited deterministic inputs bypassing a seemingly complete item record."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    assert ledger.claim(14, inputs={"prompt_id": 1}, attempt_seed=14)
    _complete(ledger, 14)
    record = _record(ledger, 14)
    record["inputs"] = {"prompt_id": 999}
    atomic_json(ledger.item_path(14), record)

    assert not ledger.is_complete(14)


def test_ledger_caps_retries_and_keeps_the_last_exception_durable(tmp_path: Path):
    """Catches persistent failures being retried forever or replaced by a fabricated transcript."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", max_attempts=2)

    assert ledger.claim(21)
    ledger.record_failure(21, stage="generation", exception=RuntimeError("first failure"))
    assert ledger.claim(21)
    ledger.record_failure(21, stage="generation", exception=RuntimeError("second failure"))

    record = _record(ledger, 21)
    assert record["attempts"]["generation"] == 2
    assert record["generation"]["status"] == "failed"
    assert record["generation"]["exception"] == {"message": "second failure", "type": "RuntimeError"}
    assert record["asr"]["hypothesis"] is None
    assert not ledger.claim(21)
    assert not ledger.is_complete(21)


def test_ledger_does_not_allow_a_capped_failure_to_be_overwritten_directly(tmp_path: Path):
    """Catches callers bypassing claim() and replacing a terminal exception with a success."""
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", max_attempts=1)
    ledger.record_failure(22, stage="generation", exception=RuntimeError("generation failed"))
    wav_path = ledger.wav_path(22)
    _write_wav(wav_path)

    with pytest.raises(Exception, match="retry cap"):
        ledger.record_generation(22, wav_sha256=sha256_file(wav_path), path=wav_path)

    assert _record(ledger, 22)["generation"]["status"] == "failed"


def test_ledger_uses_ranked_wav_paths_and_rejects_duplicate_or_stale_owners(tmp_path: Path):
    """Catches ranks colliding on a WAV or both processing one live item record."""
    rank_zero = ValidationLedger(
        tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", claim_timeout_seconds=3600
    )
    rank_one = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1", claim_timeout_seconds=3600)

    assert rank_zero.wav_path(31, rank=0) != rank_one.wav_path(31, rank=1)
    assert rank_zero.wav_path(31, rank=0).name == rank_one.wav_path(31, rank=1).name
    assert rank_zero.claim(31, rank=0)
    assert not rank_one.claim(31, rank=1)
    stale = _record(rank_zero, 31)
    stale["owner"]["claimed_at"] = "2000-01-01T00:00:00Z"
    atomic_json(rank_zero.item_path(31), stale)

    assert rank_one.claim(31, rank=1)
    assert _record(rank_one, 31)["owner"]["rank"] == 1


def test_ledger_allows_disjoint_ranks_to_publish_distinct_item_records(tmp_path: Path):
    """Catches per-rank ledger files that hide a duplicate/missing global benchmark ID."""
    rank_zero = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    rank_one = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    _complete(rank_zero, 40, rank=0)
    _complete(rank_one, 41, rank=1)

    assert rank_zero.complete_ids() == {40, 41}
    assert rank_zero.item_path(40) != rank_one.item_path(41)
