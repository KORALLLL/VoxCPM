"""Integration coverage for the strict, offset-only corpus index."""

from __future__ import annotations

from dataclasses import replace
import fcntl
import sqlite3

import pytest

from voxcpm.training.balalaika.index import IndexIntegrityError, build_index


def test_build_index_joins_by_identity_and_splits_exact_boundary(synthetic_corpus):
    """Catches a non-strict identity join or an incorrect 0.95 stage boundary."""
    audit = build_index(synthetic_corpus.config, synthetic_corpus.expectations)

    with sqlite3.connect(audit.index_path) as db:
        rows = db.execute(
            "SELECT source_relative_path, agreement, stage FROM samples ORDER BY source_relative_path"
        ).fetchall()

    assert rows == [
        ("000000/a.wav", 0.949999, 1),
        ("000000/b.wav", 0.95, 2),
        ("000001/c.wav", None, None),
    ]
    assert audit.excluded_null_agreement == 1


def test_build_index_rejects_missing_combined_text(synthetic_corpus):
    """Catches a source sample silently surviving without its training text."""
    synthetic_corpus.remove_combined_row("000000/b.wav")

    with pytest.raises(IndexIntegrityError, match="missing combined text"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)


def test_build_index_rejects_duplicate_identities(synthetic_corpus):
    """Catches a duplicate sidecar identity before it can fan out the SQL join."""
    synthetic_corpus.add_duplicate_combined_row("000000/a.wav")

    with pytest.raises(IndexIntegrityError, match="duplicate identity"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)

    assert not list(synthetic_corpus.config.index_dir.glob("*.sqlite3"))


def test_build_index_rejects_missing_audio_json_pair(synthetic_corpus):
    """Catches source metadata whose sibling audio member is missing."""
    synthetic_corpus.remove_source_audio("000000/a.wav")

    with pytest.raises(IndexIntegrityError, match="missing audio/JSON pair"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)


@pytest.mark.parametrize(
    ("expectation", "message"),
    [
        ("source_shard_count", "unexpected source shard count"),
        ("source_row_count", "unexpected source row count"),
        ("rover_row_count", "unexpected ROVER row count"),
        ("combined_row_count", "unexpected combined row count"),
    ],
)
def test_build_index_rejects_unexpected_shard_or_row_counts(synthetic_corpus, expectation, message):
    """Catches drift from the trusted corpus inventory before publication."""
    expected = getattr(synthetic_corpus.expectations, expectation)
    expectations = replace(synthetic_corpus.expectations, **{expectation: expected + 1})

    with pytest.raises(IndexIntegrityError, match=message):
        build_index(synthetic_corpus.config, expectations)


def test_build_index_rejects_unexpected_rover_shard_count(synthetic_corpus):
    """Catches an archive missing one of the required per-source-shard JSONL members."""
    synthetic_corpus.remove_rover_shard("000001")

    with pytest.raises(IndexIntegrityError, match="unexpected ROVER shard count"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)


def test_build_index_rejects_malformed_agreement(synthetic_corpus):
    """Catches a non-numeric ROVER agreement before stage assignment."""
    synthetic_corpus.replace_agreement("000000/a.wav", "not-a-score")

    with pytest.raises(IndexIntegrityError, match="malformed agreement"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)


def test_build_index_rejects_missing_agreement_key_but_accepts_explicit_null(synthetic_corpus):
    """Catches schema drift being mistaken for the intentional null-agreement exclusion."""
    audit = build_index(synthetic_corpus.config, synthetic_corpus.expectations)
    with sqlite3.connect(audit.index_path) as db:
        assert db.execute("SELECT stage FROM samples WHERE source_relative_path = '000001/c.wav'").fetchone() == (None,)

    synthetic_corpus.remove_agreement("000000/a.wav")
    with pytest.raises(IndexIntegrityError, match="missing asr_agreement_mean"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)


def test_build_index_rejects_concurrent_builder(synthetic_corpus):
    """Catches a second builder publishing a database/audit pair over an active build."""
    synthetic_corpus.config.index_dir.mkdir()
    lock_path = synthetic_corpus.config.index_dir / ".balalaika-index.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(IndexIntegrityError, match="another index build"):
            build_index(synthetic_corpus.config, synthetic_corpus.expectations)


def test_build_index_rejects_empty_combined_text(synthetic_corpus):
    """Catches whitespace-only selected training text."""
    synthetic_corpus.replace_text("000000/a.wav", " \t")

    with pytest.raises(IndexIntegrityError, match="empty combined text"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)


def test_build_index_rejects_incorrect_combined_sidecar_sha256(synthetic_corpus):
    """Catches a sidecar whose content no longer matches trusted provenance."""
    expectations = replace(synthetic_corpus.expectations, combined_sidecar_sha256="0" * 64)

    with pytest.raises(IndexIntegrityError, match="combined-sidecar SHA-256"):
        build_index(synthetic_corpus.config, expectations)


def test_build_index_rejects_incorrect_rover_archive_sha256(synthetic_corpus):
    """Catches a compressed ROVER archive substituted after metadata capture."""
    expectations = replace(synthetic_corpus.expectations, rover_archive_sha256="0" * 64)

    with pytest.raises(IndexIntegrityError, match="ROVER archive SHA-256"):
        build_index(synthetic_corpus.config, expectations)


def test_build_index_records_offsets_dense_ordinals_and_metadata(synthetic_corpus):
    """Catches text duplication or sparse per-stage random-access ordinals."""
    audit = build_index(synthetic_corpus.config, synthetic_corpus.expectations)

    with sqlite3.connect(audit.index_path) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(samples)")}
        ordinals = db.execute("""
            SELECT ordinals.stage, ordinals.ordinal, samples.source_relative_path
            FROM stage_ordinals AS ordinals
            JOIN samples USING (sample_id)
            ORDER BY ordinals.stage, ordinals.ordinal
            """).fetchall()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        metadata = dict(db.execute("SELECT key, value FROM metadata"))

    assert {"sidecar_offset", "sidecar_size", "audio_offset", "audio_size", "json_offset", "json_size"} <= columns
    assert "rover_punctuated_accented" not in columns
    assert ordinals == [(1, 0, "000000/a.wav"), (2, 0, "000000/b.wav")]
    assert integrity == "ok"
    assert metadata["fingerprint"] == audit.fingerprint
