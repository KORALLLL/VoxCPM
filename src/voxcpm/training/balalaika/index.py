"""Strict, streaming construction of the Balalaika corpus offset index."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sqlite3
import tarfile
from tempfile import mkstemp
from typing import BinaryIO, Mapping

import zstandard

from .artifacts import atomic_json, fingerprint, sha256_file
from .config import DataConfig

_ROVER_ARCHIVE = Path("punctuation_artifacts/20260729T135419Z/balalaika-rover-results-20260729T135419Z.tar.zst")
_COMBINED_SIDECAR = Path("combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl")
_SOURCE_GLOB = "train/shard_*.tar"
_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a"})
_BATCH_SIZE = 10_000


class IndexIntegrityError(RuntimeError):
    """Raised when an input corpus artifact fails its identity/provenance contract."""


@dataclass(frozen=True)
class BuildExpectations:
    """Trusted inventory and hashes captured before indexing a corpus snapshot."""

    source_shard_count: int
    source_row_count: int
    rover_row_count: int
    combined_row_count: int
    rover_archive_sha256: str
    combined_sidecar_sha256: str
    source_tar_sha256: Mapping[str, str]


@dataclass(frozen=True)
class IndexAudit:
    """Published locations and counters for one verified index build."""

    index_path: Path
    audit_path: Path
    fingerprint: str
    total_rows: int
    eligible_rows: int
    stage1_rows: int
    stage2_rows: int
    excluded_null_agreement: int


class _HashingReader:
    """A read-only stream wrapper that computes SHA-256 during a single pass."""

    def __init__(self, source: BinaryIO):
        self._source = source
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        value = self._source.read(size)
        self.digest.update(value)
        return value

    def readline(self, size: int = -1) -> bytes:
        value = self._source.readline(size)
        self.digest.update(value)
        return value

    def readable(self) -> bool:
        return True


def build_index(config: DataConfig, expectations: BuildExpectations) -> IndexAudit:
    """Build and atomically publish a strict offset-only SQLite corpus index."""
    corpus_root = Path(config.corpus_root)
    index_dir = Path(config.index_dir)
    rover_archive = corpus_root / _ROVER_ARCHIVE
    combined_sidecar = corpus_root / _COMBINED_SIDECAR
    source_tars = sorted(corpus_root.glob(_SOURCE_GLOB))
    _validate_source_inventory(corpus_root, source_tars, expectations)

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "balalaika-index.sqlite3"
    audit_path = index_dir / "balalaika-index-audit.json"
    temporary_index = _temporary_path(index_dir, index_path.name)
    database: sqlite3.Connection | None = None
    try:
        database = sqlite3.connect(temporary_index)
        _configure_database(database)
        _create_schema(database)
        source_count = _scan_source_tars(database, corpus_root, source_tars, expectations)
        rover_count = _scan_rover_archive(database, rover_archive, expectations)
        combined_count = _scan_combined_sidecar(database, combined_sidecar, expectations)
        _validate_counts(source_count, rover_count, combined_count, expectations)
        _join_staging_rows(database)
        audit = _create_audit(
            database=database,
            index_path=index_path,
            audit_path=audit_path,
            corpus_root=corpus_root,
            rover_archive=rover_archive,
            combined_sidecar=combined_sidecar,
            source_tars=source_tars,
            expectations=expectations,
        )
        _finish_database(database)
        database.close()
        database = None
        _fsync_file(temporary_index)
        index_sha256 = sha256_file(temporary_index)
        audit_payload = {
            "index_path": str(index_path),
            "fingerprint": audit.fingerprint,
            "index_sha256": index_sha256,
            "total_rows": audit.total_rows,
            "eligible_rows": audit.eligible_rows,
            "stage1_rows": audit.stage1_rows,
            "stage2_rows": audit.stage2_rows,
            "excluded_null_agreement": audit.excluded_null_agreement,
        }
        os.replace(temporary_index, index_path)
        _fsync_directory(index_dir)
        atomic_json(audit_path, audit_payload)
        _fsync_directory(index_dir)
        return audit
    except BaseException:
        if database is not None:
            database.close()
        _remove_sqlite_artifacts(temporary_index)
        raise


def _validate_source_inventory(corpus_root: Path, source_tars: list[Path], expectations: BuildExpectations) -> None:
    if len(source_tars) != expectations.source_shard_count:
        raise IndexIntegrityError(
            f"unexpected source shard count: expected {expectations.source_shard_count}, found {len(source_tars)}"
        )
    actual = {path.relative_to(corpus_root).as_posix() for path in source_tars}
    expected = set(expectations.source_tar_sha256)
    if actual != expected:
        raise IndexIntegrityError("unexpected source shard identities")


def _configure_database(database: sqlite3.Connection) -> None:
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA synchronous=FULL")
    database.execute("PRAGMA foreign_keys=ON")
    database.execute("PRAGMA cache_size=-65536")
    database.execute("PRAGMA temp_store=MEMORY")


def _create_schema(database: sqlite3.Connection) -> None:
    database.executescript("""
        CREATE TABLE source_stage (
            source_relative_path TEXT PRIMARY KEY,
            source_tar_path TEXT NOT NULL,
            audio_offset INTEGER NOT NULL CHECK (audio_offset >= 0),
            audio_size INTEGER NOT NULL CHECK (audio_size >= 0),
            json_offset INTEGER NOT NULL CHECK (json_offset >= 0),
            json_size INTEGER NOT NULL CHECK (json_size >= 0),
            duration REAL
        ) WITHOUT ROWID;

        CREATE TABLE rover_stage (
            source_relative_path TEXT PRIMARY KEY,
            agreement REAL
        ) WITHOUT ROWID;

        CREATE TABLE combined_stage (
            source_relative_path TEXT PRIMARY KEY,
            sidecar_offset INTEGER NOT NULL CHECK (sidecar_offset >= 0),
            sidecar_size INTEGER NOT NULL CHECK (sidecar_size > 0)
        ) WITHOUT ROWID;

        CREATE TABLE samples (
            sample_id INTEGER PRIMARY KEY,
            source_relative_path TEXT NOT NULL UNIQUE,
            source_tar_path TEXT NOT NULL,
            audio_offset INTEGER NOT NULL CHECK (audio_offset >= 0),
            audio_size INTEGER NOT NULL CHECK (audio_size >= 0),
            json_offset INTEGER NOT NULL CHECK (json_offset >= 0),
            json_size INTEGER NOT NULL CHECK (json_size >= 0),
            sidecar_offset INTEGER NOT NULL CHECK (sidecar_offset >= 0),
            sidecar_size INTEGER NOT NULL CHECK (sidecar_size > 0),
            duration REAL,
            agreement REAL,
            stage INTEGER CHECK (stage IN (1, 2) OR stage IS NULL)
        );

        CREATE TABLE stage_ordinals (
            stage INTEGER NOT NULL CHECK (stage IN (1, 2)),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            sample_id INTEGER NOT NULL UNIQUE REFERENCES samples(sample_id),
            PRIMARY KEY (stage, ordinal)
        ) WITHOUT ROWID;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE INDEX samples_stage_path ON samples(stage, source_relative_path);
        """)


def _scan_source_tars(
    database: sqlite3.Connection,
    corpus_root: Path,
    source_tars: list[Path],
    expectations: BuildExpectations,
) -> int:
    count = 0
    for source_tar in source_tars:
        members = _scan_source_tar(source_tar)
        relative_path = source_tar.relative_to(corpus_root).as_posix()
        actual_hash = members.pop("sha256")
        expected_hash = expectations.source_tar_sha256[relative_path]
        if actual_hash != expected_hash:
            raise IndexIntegrityError(f"source tar SHA-256 mismatch: {relative_path}")
        rows = members.pop("rows")
        _insert_rows(
            database,
            """
            INSERT INTO source_stage (
                source_relative_path, source_tar_path, audio_offset, audio_size, json_offset, json_size, duration
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
            "source tar",
        )
        count += len(rows)
    return count


def _scan_source_tar(path: Path) -> dict[str, object]:
    audio_members: dict[str, tuple[str, int, int]] = {}
    json_members: dict[str, tuple[str, int, int, float | None]] = {}
    with path.open("rb") as source:
        reader = _HashingReader(source)
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_path = _normalize_identity(member.name, "source tar member")
                suffix = PurePosixPath(member_path).suffix.lower()
                stem = str(PurePosixPath(member_path).with_suffix(""))
                if suffix in _AUDIO_SUFFIXES:
                    if stem in audio_members:
                        raise IndexIntegrityError(f"duplicate identity in source tar: {member_path}")
                    audio_members[stem] = (member_path, member.offset_data, member.size)
                elif suffix == ".json":
                    if stem in json_members:
                        raise IndexIntegrityError(f"duplicate identity in source tar: {member_path}")
                    member_file = archive.extractfile(member)
                    if member_file is None:
                        raise IndexIntegrityError(f"cannot read source JSON member: {member_path}")
                    payload = _load_json(member_file.read(), f"source JSON member {member_path}")
                    identity = _identity_from_row(payload, f"source JSON member {member_path}")
                    json_members[stem] = (identity, member.offset_data, member.size, _duration_from_row(payload))
        _drain(reader)
        digest = reader.digest.hexdigest()

    missing = sorted(set(audio_members).symmetric_difference(json_members))
    if missing:
        raise IndexIntegrityError(f"missing audio/JSON pair in {path.name}: {missing[0]}")
    rows: list[tuple[object, ...]] = []
    for stem in sorted(audio_members):
        audio_identity, audio_offset, audio_size = audio_members[stem]
        json_identity, json_offset, json_size, duration = json_members[stem]
        if audio_identity != json_identity:
            raise IndexIntegrityError(
                f"source JSON identity does not match audio member: {json_identity!r} != {audio_identity!r}"
            )
        rows.append((audio_identity, str(path.resolve()), audio_offset, audio_size, json_offset, json_size, duration))
    return {"sha256": digest, "rows": rows}


def _scan_rover_archive(database: sqlite3.Connection, path: Path, expectations: BuildExpectations) -> int:
    if not path.is_file():
        raise IndexIntegrityError(f"missing ROVER archive: {path}")
    rows: list[tuple[str, float | None]] = []
    count = 0
    shard_count = 0
    with path.open("rb") as source:
        hashed = _HashingReader(source)
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(hashed) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".jsonl"):
                        continue
                    shard_count += 1
                    member_file = archive.extractfile(member)
                    if member_file is None:
                        raise IndexIntegrityError(f"cannot read ROVER member: {member.name}")
                    for line_number, line in enumerate(member_file, start=1):
                        if not line.strip():
                            raise IndexIntegrityError(f"malformed ROVER JSONL row: {member.name}:{line_number}")
                        payload = _load_json(line, f"ROVER row {member.name}:{line_number}")
                        rows.append(
                            (
                                _identity_from_row(payload, f"ROVER row {member.name}:{line_number}"),
                                _agreement_from_row(payload, f"ROVER row {member.name}:{line_number}"),
                            )
                        )
                        count += 1
                        if len(rows) == _BATCH_SIZE:
                            _insert_rows(
                                database,
                                "INSERT INTO rover_stage (source_relative_path, agreement) VALUES (?, ?)",
                                rows,
                                "ROVER archive",
                            )
                            rows = []
            _drain(decompressed)
        _drain(hashed)
        actual_hash = hashed.digest.hexdigest()
    _insert_rows(
        database,
        "INSERT INTO rover_stage (source_relative_path, agreement) VALUES (?, ?)",
        rows,
        "ROVER archive",
    )
    if actual_hash != expectations.rover_archive_sha256:
        raise IndexIntegrityError("ROVER archive SHA-256 mismatch")
    if shard_count != expectations.source_shard_count:
        raise IndexIntegrityError(
            f"unexpected ROVER shard count: expected {expectations.source_shard_count}, found {shard_count}"
        )
    return count


def _scan_combined_sidecar(database: sqlite3.Connection, path: Path, expectations: BuildExpectations) -> int:
    if not path.is_file():
        raise IndexIntegrityError(f"missing combined sidecar: {path}")
    rows: list[tuple[str, int, int]] = []
    count = 0
    offset = 0
    with path.open("rb") as source:
        reader = _HashingReader(source)
        line_number = 0
        while line := reader.readline():
            line_number += 1
            size = len(line)
            if not line.strip():
                raise IndexIntegrityError(f"malformed combined JSONL row: {line_number}")
            payload = _load_json(line, f"combined sidecar row {line_number}")
            identity = _identity_from_row(payload, f"combined sidecar row {line_number}")
            text = payload.get("rover_punctuated_accented")
            if not isinstance(text, str) or not text.strip():
                raise IndexIntegrityError(f"empty combined text for identity {identity!r}")
            rows.append((identity, offset, size))
            count += 1
            offset += size
            if len(rows) == _BATCH_SIZE:
                _insert_rows(
                    database,
                    "INSERT INTO combined_stage (source_relative_path, sidecar_offset, sidecar_size) VALUES (?, ?, ?)",
                    rows,
                    "combined sidecar",
                )
                rows = []
        actual_hash = reader.digest.hexdigest()
    _insert_rows(
        database,
        "INSERT INTO combined_stage (source_relative_path, sidecar_offset, sidecar_size) VALUES (?, ?, ?)",
        rows,
        "combined sidecar",
    )
    if actual_hash != expectations.combined_sidecar_sha256:
        raise IndexIntegrityError("combined-sidecar SHA-256 mismatch")
    return count


def _insert_rows(
    database: sqlite3.Connection, statement: str, rows: list[tuple[object, ...]], source_name: str
) -> None:
    if not rows:
        return
    try:
        with database:
            database.executemany(statement, rows)
    except sqlite3.IntegrityError as error:
        if "UNIQUE constraint failed" in str(error):
            raise IndexIntegrityError(f"duplicate identity in {source_name}") from error
        raise IndexIntegrityError(f"invalid {source_name} row") from error


def _validate_counts(source_count: int, rover_count: int, combined_count: int, expectations: BuildExpectations) -> None:
    for actual, expected, label in (
        (source_count, expectations.source_row_count, "source"),
        (rover_count, expectations.rover_row_count, "ROVER"),
        (combined_count, expectations.combined_row_count, "combined"),
    ):
        if actual != expected:
            raise IndexIntegrityError(f"unexpected {label} row count: expected {expected}, found {actual}")


def _join_staging_rows(database: sqlite3.Connection) -> None:
    _raise_missing_identity(
        database,
        """
        SELECT source.source_relative_path
        FROM source_stage AS source
        LEFT JOIN rover_stage AS rover USING (source_relative_path)
        WHERE rover.source_relative_path IS NULL
        LIMIT 1
        """,
        "missing ROVER agreement",
    )
    _raise_missing_identity(
        database,
        """
        SELECT source.source_relative_path
        FROM source_stage AS source
        LEFT JOIN combined_stage AS combined USING (source_relative_path)
        WHERE combined.source_relative_path IS NULL
        LIMIT 1
        """,
        "missing combined text",
    )
    _raise_missing_identity(
        database,
        """
        SELECT rover.source_relative_path
        FROM rover_stage AS rover
        LEFT JOIN source_stage AS source USING (source_relative_path)
        WHERE source.source_relative_path IS NULL
        LIMIT 1
        """,
        "unexpected ROVER identity",
    )
    _raise_missing_identity(
        database,
        """
        SELECT combined.source_relative_path
        FROM combined_stage AS combined
        LEFT JOIN source_stage AS source USING (source_relative_path)
        WHERE source.source_relative_path IS NULL
        LIMIT 1
        """,
        "unexpected combined identity",
    )
    with database:
        database.execute("""
            INSERT INTO samples (
                source_relative_path, source_tar_path, audio_offset, audio_size, json_offset, json_size,
                sidecar_offset, sidecar_size, duration, agreement, stage
            )
            SELECT
                source.source_relative_path, source.source_tar_path, source.audio_offset, source.audio_size,
                source.json_offset, source.json_size, combined.sidecar_offset, combined.sidecar_size,
                source.duration, rover.agreement,
                CASE
                    WHEN rover.agreement IS NULL THEN NULL
                    WHEN rover.agreement < 0.95 THEN 1
                    ELSE 2
                END
            FROM source_stage AS source
            JOIN rover_stage AS rover USING (source_relative_path)
            JOIN combined_stage AS combined USING (source_relative_path)
            """)
        source_count = database.execute("SELECT COUNT(*) FROM source_stage").fetchone()[0]
        joined_count = database.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        if source_count != joined_count:
            raise IndexIntegrityError("strict SQL join did not preserve one-to-one source rows")
        database.execute("""
            INSERT INTO stage_ordinals (stage, ordinal, sample_id)
            SELECT
                stage,
                ROW_NUMBER() OVER (PARTITION BY stage ORDER BY source_relative_path) - 1,
                sample_id
            FROM samples
            WHERE stage IS NOT NULL
            """)
        database.executescript("DROP TABLE source_stage; DROP TABLE rover_stage; DROP TABLE combined_stage;")


def _raise_missing_identity(database: sqlite3.Connection, query: str, problem: str) -> None:
    row = database.execute(query).fetchone()
    if row is not None:
        raise IndexIntegrityError(f"{problem} for identity {row[0]!r}")


def _create_audit(
    *,
    database: sqlite3.Connection,
    index_path: Path,
    audit_path: Path,
    corpus_root: Path,
    rover_archive: Path,
    combined_sidecar: Path,
    source_tars: list[Path],
    expectations: BuildExpectations,
) -> IndexAudit:
    total_rows = database.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    stage1_rows = database.execute("SELECT COUNT(*) FROM samples WHERE stage = 1").fetchone()[0]
    stage2_rows = database.execute("SELECT COUNT(*) FROM samples WHERE stage = 2").fetchone()[0]
    excluded = database.execute("SELECT COUNT(*) FROM samples WHERE agreement IS NULL").fetchone()[0]
    provenance = {
        "schema_version": 1,
        "corpus_root": str(corpus_root),
        "rover_archive": str(rover_archive),
        "combined_sidecar": str(combined_sidecar),
        "source_tars": [str(path) for path in source_tars],
        "source_tar_sha256": dict(sorted(expectations.source_tar_sha256.items())),
        "rover_archive_sha256": expectations.rover_archive_sha256,
        "combined_sidecar_sha256": expectations.combined_sidecar_sha256,
        "source_shard_count": expectations.source_shard_count,
        "source_row_count": expectations.source_row_count,
        "rover_row_count": expectations.rover_row_count,
        "combined_row_count": expectations.combined_row_count,
        "stage_rule": {"stage1": "agreement < 0.95", "stage2": "agreement >= 0.95", "excluded": "agreement is null"},
    }
    build_fingerprint = fingerprint(provenance)
    with database:
        database.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            (
                ("fingerprint", build_fingerprint),
                ("provenance", json.dumps(provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                ("sidecar_path", str(combined_sidecar)),
            ),
        )
    return IndexAudit(
        index_path=index_path,
        audit_path=audit_path,
        fingerprint=build_fingerprint,
        total_rows=total_rows,
        eligible_rows=stage1_rows + stage2_rows,
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        excluded_null_agreement=excluded,
    )


def _finish_database(database: sqlite3.Connection) -> None:
    with database:
        integrity = database.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise IndexIntegrityError(f"SQLite integrity_check failed: {integrity}")
    database.execute("VACUUM")
    database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    database.execute("PRAGMA journal_mode=DELETE")


def _identity_from_row(payload: object, context: str) -> str:
    if not isinstance(payload, dict):
        raise IndexIntegrityError(f"malformed JSON object in {context}")
    return _normalize_identity(payload.get("source_relative_path"), context)


def _agreement_from_row(payload: object, context: str) -> float | None:
    if not isinstance(payload, dict):
        raise IndexIntegrityError(f"malformed JSON object in {context}")
    value = payload.get("asr_agreement_mean")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexIntegrityError(f"malformed agreement in {context}")
    agreement = float(value)
    if not math.isfinite(agreement) or not 0.0 <= agreement <= 1.0:
        raise IndexIntegrityError(f"malformed agreement in {context}")
    return agreement


def _duration_from_row(payload: object) -> float | None:
    if not isinstance(payload, dict) or "duration" not in payload or payload["duration"] is None:
        return None
    value = payload["duration"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexIntegrityError("malformed duration in source JSON")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise IndexIntegrityError("malformed duration in source JSON")
    return duration


def _normalize_identity(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise IndexIntegrityError(f"missing source_relative_path in {context}")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise IndexIntegrityError(f"invalid source_relative_path in {context}")
    return path.as_posix()


def _load_json(payload: bytes, context: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise IndexIntegrityError(f"malformed JSON in {context}") from error


def _drain(reader: object) -> None:
    read = getattr(reader, "read")
    while read(1024 * 1024):
        pass


def _temporary_path(directory: Path, basename: str) -> Path:
    descriptor, temporary = mkstemp(prefix=f".{basename}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    temporary_path = Path(temporary)
    temporary_path.unlink()
    return temporary_path


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_sqlite_artifacts(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)
