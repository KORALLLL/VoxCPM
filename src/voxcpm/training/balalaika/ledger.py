"""Atomic per-item state for resumable distributed validation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

from .artifacts import atomic_json, fingerprint, sha256_file

_STAGES = frozenset({"generation", "asr"})


class ValidationLedgerError(RuntimeError):
    """Raised when a ledger transition is invalid for the durable item state."""


class ValidationLedger:
    """One atomically replaced JSON record per globally unique benchmark item ID.

    Each operation takes an advisory lock only for one item file.  That makes
    writes from ranks with disjoint partitions independent while still keeping a
    duplicate assignment from publishing two competing records for the same ID.
    """

    def __init__(
        self,
        root: Path,
        generation_fingerprint: str,
        asr_fingerprint: str,
        *,
        max_attempts: int = 3,
        claim_timeout_seconds: float = 900.0,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one.")
        if claim_timeout_seconds < 0:
            raise ValueError("claim_timeout_seconds cannot be negative.")
        self.root = Path(root)
        self.generation_fingerprint = str(generation_fingerprint)
        self.asr_fingerprint = str(asr_fingerprint)
        self.max_attempts = int(max_attempts)
        self.claim_timeout_seconds = float(claim_timeout_seconds)
        self.items_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    @property
    def items_dir(self) -> Path:
        return self.root / "items"

    @property
    def locks_dir(self) -> Path:
        return self.root / "locks"

    def item_path(self, item_id: int) -> Path:
        return self.items_dir / f"{_item_id(item_id):05d}.json"

    def wav_path(self, item_id: int, *, rank: int = 0) -> Path:
        """Return a rank-namespaced, deterministic WAV destination for an item."""
        if rank < 0:
            raise ValueError("rank cannot be negative.")
        return self.root / "wavs" / f"rank-{rank:02d}" / f"{_item_id(item_id):05d}.wav"

    def claim(
        self,
        item_id: int,
        *,
        rank: int = 0,
        inputs: Mapping[str, Any] | None = None,
        attempt_seed: int | None = None,
    ) -> bool:
        """Atomically claim an incomplete item unless another live rank owns it."""
        if rank < 0:
            raise ValueError("rank cannot be negative.")
        with self._locked_record(item_id) as record:
            changed = self._prepare_for_reuse(record, inputs=inputs, attempt_seed=attempt_seed)
            if self._is_complete_record(record) or self._exhausted(record):
                if changed:
                    self._write_record(item_id, record)
                return False
            owner = record.get("owner")
            if isinstance(owner, dict) and not self._owner_is_stale(owner):
                if changed:
                    self._write_record(item_id, record)
                return False
            record["owner"] = {"rank": rank, "claimed_at": _timestamp()}
            self._touch(record)
            self._write_record(item_id, record)
            return True

    def record_generation(
        self,
        item_id: int,
        *,
        wav_sha256: str,
        path: str | Path,
        attempt_seed: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Record a generated WAV only after its supplied digest verifies on disk."""
        actual_path = self._resolve_wav_path(path)
        if not actual_path.is_file():
            raise FileNotFoundError(f"Generated WAV is missing: {actual_path}")
        actual_hash = sha256_file(actual_path)
        if actual_hash != wav_sha256:
            raise ValueError(f"Generated WAV hash does not match for {actual_path}.")
        with self._locked_record(item_id) as record:
            self._prepare_for_reuse(record, inputs=None, attempt_seed=attempt_seed)
            if self._stage_exhausted(record, "generation"):
                raise ValidationLedgerError(f"Generation retry cap reached for item {_item_id(item_id)}.")
            record["attempts"]["generation"] += 1
            record["generation"] = {
                "status": "success",
                "wav_path": str(actual_path),
                "wav_sha256": actual_hash,
                "output_sha256": actual_hash,
                "attempt_seed": record["attempt_seed"],
                "elapsed_seconds": elapsed_seconds,
                "completed_at": _timestamp(),
                "exception": None,
            }
            self._reset_asr(record, reset_attempts=True)
            record["owner"] = None
            self._touch(record)
            self._write_record(item_id, record)

    def record_asr(self, item_id: int, *, hypothesis: str, elapsed_seconds: float | None = None) -> None:
        """Persist an ASR success, including a valid empty-string hypothesis."""
        if not isinstance(hypothesis, str):
            raise TypeError("ASR hypothesis must be a string.")
        with self._locked_record(item_id) as record:
            self._prepare_for_reuse(record, inputs=None, attempt_seed=None)
            if self._stage_exhausted(record, "asr"):
                raise ValidationLedgerError(f"ASR retry cap reached for item {_item_id(item_id)}.")
            if record["generation"]["status"] != "success" or not self._wav_is_valid(record):
                raise ValidationLedgerError(
                    f"Cannot record ASR for item {_item_id(item_id)} without a verified generated WAV."
                )
            record["attempts"]["asr"] += 1
            record["asr"] = {
                "status": "success",
                "hypothesis": hypothesis,
                "hypothesis_sha256": _text_hash(hypothesis),
                "input_wav_sha256": record["generation"]["wav_sha256"],
                "elapsed_seconds": elapsed_seconds,
                "completed_at": _timestamp(),
                "exception": None,
            }
            record["owner"] = None
            self._touch(record)
            self._write_record(item_id, record)

    def record_failure(self, item_id: int, *, stage: str, exception: BaseException | str) -> None:
        """Persist an exception as incomplete state for bounded deterministic retry."""
        if stage not in _STAGES:
            raise ValueError(f"Unknown validation stage: {stage!r}")
        with self._locked_record(item_id) as record:
            self._prepare_for_reuse(record, inputs=None, attempt_seed=None)
            record["attempts"][stage] += 1
            error = _exception_record(exception)
            if stage == "generation":
                record["generation"] = {
                    "status": "failed",
                    "wav_path": None,
                    "wav_sha256": None,
                    "output_sha256": None,
                    "attempt_seed": record["attempt_seed"],
                    "elapsed_seconds": None,
                    "completed_at": _timestamp(),
                    "exception": error,
                }
                self._reset_asr(record, reset_attempts=True)
            else:
                record["asr"] = {
                    "status": "failed",
                    "hypothesis": None,
                    "hypothesis_sha256": None,
                    "input_wav_sha256": record["generation"]["wav_sha256"],
                    "elapsed_seconds": None,
                    "completed_at": _timestamp(),
                    "exception": error,
                }
            record["owner"] = None
            self._touch(record)
            self._write_record(item_id, record)

    def is_complete(self, item_id: int) -> bool:
        """Return whether an item remains valid for reuse under these fingerprints."""
        path = self.item_path(item_id)
        if not path.is_file():
            return False
        with self._locked_record(item_id, create=False) as record:
            return self._is_complete_record(record)

    def complete_ids(self) -> set[int]:
        """Return all globally complete IDs, irrespective of which rank wrote them."""
        complete: set[int] = set()
        for path in sorted(self.items_dir.glob("*.json")):
            try:
                item_id = int(path.stem)
            except ValueError:
                continue
            if self.is_complete(item_id):
                complete.add(item_id)
        return complete

    @contextmanager
    def _locked_record(self, item_id: int, *, create: bool = True) -> Iterator[dict[str, Any]]:
        normalized_id = _item_id(item_id)
        lock_path = self.locks_dir / f"{normalized_id:05d}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                path = self.item_path(normalized_id)
                if path.is_file():
                    try:
                        record = json.loads(path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError as error:
                        raise ValidationLedgerError(f"Ledger item record is corrupt: {path}") from error
                    if not isinstance(record, dict):
                        raise ValidationLedgerError(f"Ledger item record is not an object: {path}")
                elif create:
                    record = self._new_record(normalized_id)
                else:
                    record = self._new_record(normalized_id)
                yield record
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _new_record(self, item_id: int) -> dict[str, Any]:
        now = _timestamp()
        record: dict[str, Any] = {
            "version": 1,
            "id": item_id,
            "fingerprints": {"generation": self.generation_fingerprint, "asr": self.asr_fingerprint},
            "inputs": {},
            "attempt_seed": None,
            "input_fingerprint": fingerprint({"inputs": {}, "attempt_seed": None}),
            "attempts": {"generation": 0, "asr": 0},
            "generation": {},
            "asr": {},
            "owner": None,
            "created_at": now,
            "updated_at": now,
        }
        self._reset_generation(record, reset_attempts=False)
        return record

    def _prepare_for_reuse(
        self,
        record: dict[str, Any],
        *,
        inputs: Mapping[str, Any] | None,
        attempt_seed: int | None,
    ) -> bool:
        changed = False
        fingerprints = record.get("fingerprints")
        if not isinstance(fingerprints, dict):
            fingerprints = {}
            record["fingerprints"] = fingerprints
            changed = True
        generation_changed = fingerprints.get("generation") != self.generation_fingerprint
        asr_changed = fingerprints.get("asr") != self.asr_fingerprint
        if generation_changed:
            fingerprints["generation"] = self.generation_fingerprint
            fingerprints["asr"] = self.asr_fingerprint
            self._reset_generation(record, reset_attempts=True)
            record["owner"] = None
            changed = True
        elif asr_changed:
            fingerprints["asr"] = self.asr_fingerprint
            self._reset_asr(record, reset_attempts=True)
            record["owner"] = None
            changed = True

        if inputs is not None:
            normalized_inputs = _normalize_inputs(inputs)
            if record.get("inputs") != normalized_inputs or record.get("attempt_seed") != attempt_seed:
                record["inputs"] = normalized_inputs
                record["attempt_seed"] = attempt_seed
                record["input_fingerprint"] = fingerprint({"inputs": normalized_inputs, "attempt_seed": attempt_seed})
                self._reset_generation(record, reset_attempts=True)
                record["owner"] = None
                changed = True
        elif attempt_seed is not None and record.get("attempt_seed") != attempt_seed:
            record["attempt_seed"] = attempt_seed
            record["input_fingerprint"] = fingerprint(
                {"inputs": record.get("inputs", {}), "attempt_seed": attempt_seed}
            )
            self._reset_generation(record, reset_attempts=True)
            record["owner"] = None
            changed = True
        elif record.get("input_fingerprint") != fingerprint(
            {"inputs": record.get("inputs", {}), "attempt_seed": record.get("attempt_seed")}
        ):
            self._reset_generation(record, reset_attempts=True)
            record["input_fingerprint"] = fingerprint(
                {"inputs": record.get("inputs", {}), "attempt_seed": record.get("attempt_seed")}
            )
            record["owner"] = None
            changed = True

        if record["generation"].get("status") == "success" and not self._wav_is_valid(record):
            self._reset_generation(record, reset_attempts=True)
            record["owner"] = None
            changed = True
        elif record["asr"].get("status") == "success" and not self._asr_is_valid(record):
            self._reset_asr(record, reset_attempts=True)
            record["owner"] = None
            changed = True
        if changed:
            self._touch(record)
        return changed

    def _is_complete_record(self, record: Mapping[str, Any]) -> bool:
        fingerprints = record.get("fingerprints")
        return bool(
            isinstance(fingerprints, Mapping)
            and fingerprints.get("generation") == self.generation_fingerprint
            and fingerprints.get("asr") == self.asr_fingerprint
            and record.get("generation", {}).get("status") == "success"
            and record.get("asr", {}).get("status") == "success"
            and self._inputs_are_valid(record)
            and self._wav_is_valid(record)
            and self._asr_is_valid(record)
        )

    def _inputs_are_valid(self, record: Mapping[str, Any]) -> bool:
        inputs = record.get("inputs")
        if not isinstance(inputs, Mapping):
            return False
        try:
            expected = fingerprint({"inputs": dict(inputs), "attempt_seed": record.get("attempt_seed")})
        except (TypeError, ValueError):
            return False
        return record.get("input_fingerprint") == expected

    def _wav_is_valid(self, record: Mapping[str, Any]) -> bool:
        generation = record.get("generation")
        if not isinstance(generation, Mapping):
            return False
        stored_path = generation.get("wav_path")
        stored_hash = generation.get("wav_sha256")
        if not isinstance(stored_path, str) or not isinstance(stored_hash, str):
            return False
        path = self._resolve_wav_path(stored_path)
        try:
            return path.is_file() and sha256_file(path) == stored_hash == generation.get("output_sha256")
        except OSError:
            return False

    def _asr_is_valid(self, record: Mapping[str, Any]) -> bool:
        asr = record.get("asr")
        generation = record.get("generation")
        return bool(
            isinstance(asr, Mapping)
            and isinstance(generation, Mapping)
            and isinstance(asr.get("hypothesis"), str)
            and asr.get("hypothesis_sha256") == _text_hash(asr["hypothesis"])
            and asr.get("input_wav_sha256") == generation.get("wav_sha256")
        )

    def _exhausted(self, record: Mapping[str, Any]) -> bool:
        return self._stage_exhausted(record, "generation") or self._stage_exhausted(record, "asr")

    def _stage_exhausted(self, record: Mapping[str, Any], stage: str) -> bool:
        attempts = record.get("attempts", {})
        stage_record = record.get(stage, {})
        return bool(stage_record.get("status") == "failed" and attempts.get(stage, 0) >= self.max_attempts)

    def _owner_is_stale(self, owner: Mapping[str, Any]) -> bool:
        claimed_at = owner.get("claimed_at")
        if not isinstance(claimed_at, str):
            return True
        try:
            claimed = datetime.fromisoformat(claimed_at.removesuffix("Z") + "+00:00")
        except ValueError:
            return True
        return time.time() - claimed.timestamp() >= self.claim_timeout_seconds

    def _resolve_wav_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _reset_generation(self, record: dict[str, Any], *, reset_attempts: bool) -> None:
        if reset_attempts:
            record["attempts"]["generation"] = 0
        record["generation"] = {
            "status": "pending",
            "wav_path": None,
            "wav_sha256": None,
            "output_sha256": None,
            "attempt_seed": record.get("attempt_seed"),
            "elapsed_seconds": None,
            "completed_at": None,
            "exception": None,
        }
        self._reset_asr(record, reset_attempts=reset_attempts)

    def _reset_asr(self, record: dict[str, Any], *, reset_attempts: bool) -> None:
        if reset_attempts:
            record["attempts"]["asr"] = 0
        record["asr"] = {
            "status": "pending",
            "hypothesis": None,
            "hypothesis_sha256": None,
            "input_wav_sha256": None,
            "elapsed_seconds": None,
            "completed_at": None,
            "exception": None,
        }

    def _touch(self, record: dict[str, Any]) -> None:
        record["updated_at"] = _timestamp()

    def _write_record(self, item_id: int, record: Mapping[str, Any]) -> None:
        atomic_json(self.item_path(item_id), record)


def _item_id(item_id: int) -> int:
    if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 0:
        raise ValueError(f"item ID must be a non-negative integer, got {item_id!r}")
    return item_id


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _normalize_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(dict(inputs), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Validation inputs must be JSON-serializable.") from error
    return json.loads(encoded)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exception_record(exception: BaseException | str) -> dict[str, str]:
    if isinstance(exception, BaseException):
        return {"type": type(exception).__name__, "message": str(exception)}
    return {"type": "Error", "message": str(exception)}
