"""Seeded, streaming selection manifests for Balalaika memorization and validation."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
from tempfile import NamedTemporaryFile, mkdtemp
from typing import Iterator, Sequence, TypeVar

from pydantic import BaseModel, ConfigDict, Field
import torchaudio

from .artifacts import atomic_json, fingerprint, sha256_file
from .index import IndexIntegrityError, normalize_identity


_MEMORIZATION_COUNT = 4
_PROMPT_COUNT = 20
_BENCHMARK_COUNT = 2_000
_AUDIO_LOG_COUNT = 4
_SCHEMA_VERSION = 1


class SelectionError(RuntimeError):
    """Raised when the pinned inputs cannot produce the fixed selection protocol."""


class SelectedSample(BaseModel):
    """A selected corpus item and its extracted, decoded WAV artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_relative_path: str
    agreement: float
    stage: int
    text: str
    wav_path: Path
    wav_sha256: str


class PromptSample(SelectedSample):
    """A validation prompt with a stable manifest identifier."""

    prompt_id: str


class SelectionBundle(BaseModel):
    """All fixed artifacts used by memorization and every benchmark validation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fingerprint: str
    seed: int
    memorization: list[SelectedSample]
    prompts: list[PromptSample]
    benchmark_prompt_by_id: dict[int, str]
    audio_log_ids: list[int] = Field(min_length=_AUDIO_LOG_COUNT, max_length=_AUDIO_LOG_COUNT)


@dataclass(frozen=True)
class _IndexedSample:
    source_relative_path: str
    source_tar_path: Path
    audio_offset: int
    audio_size: int
    sidecar_offset: int
    sidecar_size: int
    agreement: float
    stage: int


def create_selection_manifests(
    index_path: str | Path,
    benchmark_path: str | Path,
    output_dir: str | Path,
    seed: int,
) -> SelectionBundle:
    """Create and atomically publish the deterministic selection artifacts.

    Reservoir sampling keeps corpus-memory usage bounded while preserving an
    equal chance for every eligible row.  The two corpus selections use
    independent RNG streams; benchmark assignment deliberately continues the
    validation stream so the complete validation manifest is seed-derived.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SelectionError("seed must be an integer")
    index_path = Path(index_path).resolve()
    benchmark_path = Path(benchmark_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    if not benchmark_path.is_file():
        raise FileNotFoundError(benchmark_path)

    with _SelectionBuildLock(output_dir):
        database = _open_index(index_path)
        try:
            sidecar_path = _sidecar_path(database)
            index_fingerprint = _metadata_value(database, "fingerprint")
            memorization_rows = _reservoir(
                _iter_samples(database, stage=2), _MEMORIZATION_COUNT, random.Random(seed), "stage-2 memorization"
            )
            validation_rng = random.Random(_derived_seed(seed, "validation-prompts"))
            prompt_rows = _reservoir(
                _iter_samples(database, stage=None), _PROMPT_COUNT, validation_rng, "usable prompt"
            )
        finally:
            database.close()

        staging_dir = Path(mkdtemp(prefix=f".{output_dir.name}.build-", dir=output_dir.parent))
        try:
            memorization = [
                _materialize(
                    row,
                    sidecar_path,
                    output_dir / "audio" / "memorization" / f"item-{number:02d}.wav",
                    staging_dir / "audio" / "memorization" / f"item-{number:02d}.wav",
                )
                for number, row in enumerate(memorization_rows)
            ]
            prompts = [
                PromptSample(
                    prompt_id=f"prompt-{number:02d}",
                    **_materialize(
                        row,
                        sidecar_path,
                        output_dir / "audio" / "prompts" / f"prompt-{number:02d}.wav",
                        staging_dir / "audio" / "prompts" / f"prompt-{number:02d}.wav",
                    ).model_dump(),
                )
                for number, row in enumerate(prompt_rows)
            ]
            benchmark_ids = _load_benchmark_ids(benchmark_path)
            prompt_ids = [prompt.prompt_id for prompt in prompts]
            assignments = {benchmark_id: validation_rng.choice(prompt_ids) for benchmark_id in benchmark_ids}
            audio_log_ids = sorted(validation_rng.sample(benchmark_ids, _AUDIO_LOG_COUNT))

            selection_fingerprint = fingerprint(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "seed": seed,
                    "index_fingerprint": index_fingerprint,
                    "memorization": [_sample_fingerprint_value(item) for item in memorization],
                    "prompts": [_sample_fingerprint_value(item) for item in prompts],
                    "benchmark_prompt_by_id": assignments,
                    "audio_log_ids": audio_log_ids,
                }
            )
            bundle = SelectionBundle(
                fingerprint=selection_fingerprint,
                seed=seed,
                memorization=memorization,
                prompts=prompts,
                benchmark_prompt_by_id=assignments,
                audio_log_ids=audio_log_ids,
            )
            _write_staged_manifests(staging_dir, bundle)
            _validate_staged_bundle(staging_dir, output_dir, bundle)
            _publish_staged_bundle(staging_dir, output_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
        else:
            shutil.rmtree(staging_dir)
            return bundle


def _open_index(path: Path) -> sqlite3.Connection:
    try:
        database = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise SelectionError(f"cannot open selection index: {path}") from error
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA query_only=ON")
    return database


def _metadata_value(database: sqlite3.Connection, key: str) -> str:
    try:
        row = database.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error as error:
        raise SelectionError("selection index is missing metadata") from error
    if row is None or not isinstance(row["value"], str) or not row["value"]:
        raise SelectionError(f"selection index is missing {key} metadata")
    return row["value"]


def _sidecar_path(database: sqlite3.Connection) -> Path:
    path = Path(_metadata_value(database, "sidecar_path"))
    if not path.is_file():
        raise SelectionError(f"indexed sidecar is unavailable: {path}")
    return path


def _iter_samples(database: sqlite3.Connection, *, stage: int | None) -> Iterator[_IndexedSample]:
    where = "stage = 2 AND agreement >= 0.95" if stage == 2 else "stage IN (1, 2)"
    try:
        cursor = database.execute(
            f"""
            SELECT source_relative_path, source_tar_path, audio_offset, audio_size,
                   sidecar_offset, sidecar_size, agreement, stage
            FROM samples
            WHERE {where} AND audio_size > 0 AND sidecar_size > 0
            ORDER BY source_relative_path
            """
        )
        for row in cursor:
            agreement = row["agreement"]
            selected_stage = row["stage"]
            if (
                isinstance(agreement, bool)
                or not isinstance(agreement, (int, float))
                or isinstance(selected_stage, bool)
                or selected_stage not in (1, 2)
            ):
                raise SelectionError("selection index has an invalid stage or agreement")
            yield _IndexedSample(
                source_relative_path=str(row["source_relative_path"]),
                source_tar_path=Path(str(row["source_tar_path"])),
                audio_offset=int(row["audio_offset"]),
                audio_size=int(row["audio_size"]),
                sidecar_offset=int(row["sidecar_offset"]),
                sidecar_size=int(row["sidecar_size"]),
                agreement=float(agreement),
                stage=int(selected_stage),
            )
    except sqlite3.Error as error:
        raise SelectionError("selection index cannot stream joined samples") from error


_Sample = TypeVar("_Sample")


def _reservoir(rows: Iterator[_Sample], size: int, rng: random.Random, label: str) -> list[_Sample]:
    reservoir: list[_Sample] = []
    seen = 0
    for row in rows:
        seen += 1
        if len(reservoir) < size:
            reservoir.append(row)
            continue
        replacement = rng.randrange(seen)
        if replacement < size:
            reservoir[replacement] = row
    if seen < size:
        raise SelectionError(f"need at least {size} {label} rows; found {seen}")
    return sorted(reservoir, key=lambda row: getattr(row, "source_relative_path"))


def _materialize(row: _IndexedSample, sidecar_path: Path, wav_path: Path, staging_wav_path: Path) -> SelectedSample:
    text = _read_indexed_text(sidecar_path, row)
    audio = _read_range(row.source_tar_path, row.audio_offset, row.audio_size, "audio")
    _write_decoded_wav(audio, staging_wav_path)
    return SelectedSample(
        source_relative_path=row.source_relative_path,
        agreement=row.agreement,
        stage=row.stage,
        text=text,
        wav_path=wav_path,
        wav_sha256=sha256_file(staging_wav_path),
    )


def _read_indexed_text(sidecar_path: Path, row: _IndexedSample) -> str:
    payload = _read_range(sidecar_path, row.sidecar_offset, row.sidecar_size, "sidecar")
    try:
        value = json.loads(payload.decode("utf-8"))
        identity = normalize_identity(value.get("source_relative_path"), "selection sidecar")
    except (AttributeError, UnicodeDecodeError, json.JSONDecodeError, IndexIntegrityError) as error:
        raise SelectionError("selected sidecar row is malformed") from error
    if identity != row.source_relative_path:
        raise SelectionError("selected sidecar row does not match indexed sample")
    text = value.get("rover_punctuated_accented")
    if not isinstance(text, str) or not text.strip():
        raise SelectionError("selected sidecar text is missing or empty")
    return text


def _read_range(path: Path, offset: int, size: int, label: str) -> bytes:
    if (
        isinstance(offset, bool)
        or isinstance(size, bool)
        or not isinstance(offset, int)
        or not isinstance(size, int)
        or offset < 0
        or size <= 0
    ):
        raise SelectionError(f"invalid selected {label} byte range")
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as error:
        raise SelectionError(f"cannot open selected {label} file: {path}") from error
    try:
        file_size = os.fstat(descriptor).st_size
        if offset > file_size or size > file_size - offset:
            raise SelectionError(f"invalid selected {label} byte range")
        payload = os.pread(descriptor, size, offset)
    except (MemoryError, OSError, OverflowError) as error:
        raise SelectionError(f"cannot read selected {label} byte range") from error
    finally:
        os.close(descriptor)
    if len(payload) != size:
        raise SelectionError(f"truncated selected {label} byte range")
    return payload


def _write_decoded_wav(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        waveform, sample_rate = torchaudio.load(io.BytesIO(payload))
        with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.stem}.", suffix=".wav", delete=False) as temp:
            temporary_path = Path(temp.name)
        torchaudio.save(temporary_path, waveform, sample_rate, format="wav")
        os.replace(temporary_path, destination)
    except Exception as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise SelectionError("cannot decode selected audio to WAV") from error


def _load_benchmark_ids(path: Path) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise SelectionError(f"benchmark has an empty row at line {line_number}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SelectionError(f"benchmark has malformed JSON at line {line_number}") from error
                identifier = row.get("id") if isinstance(row, dict) else None
                if isinstance(identifier, bool) or not isinstance(identifier, int):
                    raise SelectionError(f"benchmark row {line_number} has an invalid id")
                if identifier in seen:
                    raise SelectionError(f"benchmark has duplicate id {identifier}")
                seen.add(identifier)
                ids.append(identifier)
    except OSError as error:
        raise SelectionError(f"cannot read benchmark: {path}") from error
    if len(ids) != _BENCHMARK_COUNT:
        raise SelectionError(f"benchmark must contain exactly {_BENCHMARK_COUNT} rows; found {len(ids)}")
    return sorted(ids)


def _derived_seed(seed: int, namespace: str) -> int:
    material = f"balalaika-selection-v{_SCHEMA_VERSION}:{namespace}:{seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _sample_fingerprint_value(sample: SelectedSample) -> dict[str, object]:
    value = sample.model_dump(mode="json")
    value.pop("wav_path")
    return value


def _write_staged_manifests(staging_dir: Path, bundle: SelectionBundle) -> None:
    common = {"schema_version": _SCHEMA_VERSION, "fingerprint": bundle.fingerprint, "seed": bundle.seed}
    atomic_json(
        staging_dir / "memorization.json",
        {**common, "memorization": [item.model_dump(mode="json") for item in bundle.memorization]},
    )
    atomic_json(staging_dir / "prompts.json", {**common, "prompts": [item.model_dump(mode="json") for item in bundle.prompts]})
    atomic_json(
        staging_dir / "benchmark-prompts.json",
        {**common, "benchmark_prompt_by_id": bundle.benchmark_prompt_by_id},
    )
    atomic_json(staging_dir / "audio-log-ids.json", {**common, "audio_log_ids": bundle.audio_log_ids})


_PUBLISHED_PATHS = (
    Path("audio"),
    Path("memorization.json"),
    Path("prompts.json"),
    Path("benchmark-prompts.json"),
    Path("audio-log-ids.json"),
)


def _validate_staged_bundle(staging_dir: Path, output_dir: Path, bundle: SelectionBundle) -> None:
    manifest_names = {path.name for path in staging_dir.glob("*.json")}
    expected_names = {path.name for path in _PUBLISHED_PATHS if path.suffix == ".json"}
    if manifest_names != expected_names:
        raise SelectionError("staged selection manifests are incomplete")
    manifests: dict[str, dict[str, object]] = {}
    for name in sorted(manifest_names):
        try:
            value = json.loads((staging_dir / name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SelectionError("staged selection manifest is malformed") from error
        if not isinstance(value, dict):
            raise SelectionError("staged selection manifest is malformed")
        manifests[name] = value
    if any(
        value.get("schema_version") != _SCHEMA_VERSION
        or value.get("fingerprint") != bundle.fingerprint
        or value.get("seed") != bundle.seed
        for value in manifests.values()
    ):
        raise SelectionError("staged selection manifests do not share a fingerprint")
    if manifests["memorization.json"].get("memorization") != [item.model_dump(mode="json") for item in bundle.memorization]:
        raise SelectionError("staged memorization manifest does not match selected samples")
    if manifests["prompts.json"].get("prompts") != [item.model_dump(mode="json") for item in bundle.prompts]:
        raise SelectionError("staged prompts manifest does not match selected samples")
    expected_assignments = {str(key): value for key, value in bundle.benchmark_prompt_by_id.items()}
    assignments = manifests["benchmark-prompts.json"].get("benchmark_prompt_by_id")
    if assignments != expected_assignments or len(assignments) != _BENCHMARK_COUNT:
        raise SelectionError("staged benchmark assignments are incomplete")
    prompt_ids = {prompt.prompt_id for prompt in bundle.prompts}
    if set(assignments.values()) - prompt_ids:
        raise SelectionError("staged benchmark assignments reference an unknown prompt")
    audio_log_ids = manifests["audio-log-ids.json"].get("audio_log_ids")
    if audio_log_ids != bundle.audio_log_ids or len(set(audio_log_ids)) != _AUDIO_LOG_COUNT:
        raise SelectionError("staged W&B audio log IDs are invalid")
    if not set(audio_log_ids) <= set(bundle.benchmark_prompt_by_id):
        raise SelectionError("staged W&B audio log IDs are not assigned benchmarks")
    for sample in [*bundle.memorization, *bundle.prompts]:
        try:
            staged_wav_path = staging_dir / sample.wav_path.relative_to(output_dir)
        except ValueError as error:
            raise SelectionError("staged WAV path escapes the selection output directory") from error
        if not staged_wav_path.is_file() or sha256_file(staged_wav_path) != sample.wav_sha256:
            raise SelectionError("staged WAV hash does not match the selection manifest")


def _publish_staged_bundle(staging_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = staging_dir / "backups"
    backup_dir.mkdir()
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for relative_path in _PUBLISHED_PATHS:
            published_path = output_dir / relative_path
            if published_path.exists() or published_path.is_symlink():
                backup_path = backup_dir / relative_path
                os.replace(published_path, backup_path)
                backups[relative_path] = backup_path
        for relative_path in _PUBLISHED_PATHS:
            os.replace(staging_dir / relative_path, output_dir / relative_path)
            installed.append(relative_path)
    except BaseException:
        for relative_path in reversed(installed):
            _remove_path(output_dir / relative_path)
        for relative_path, backup_path in backups.items():
            os.replace(backup_path, output_dir / relative_path)
        raise


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


class _SelectionBuildLock:
    """Nonblocking lock that serializes construction of one output bundle."""

    def __init__(self, output_dir: Path) -> None:
        self._path = output_dir.parent / f".{output_dir.name}.selection.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> "_SelectionBuildLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._descriptor = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._descriptor)
            self._descriptor = None
            raise SelectionError("another selection build is already active for this output directory") from error
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None
