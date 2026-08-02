"""Atomic, authorised per-item state for resumable distributed validation."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import fcntl
import hashlib
import json
from pathlib import Path
import secrets
import time
from typing import Any

from .artifacts import atomic_json, fingerprint, sha256_file

_STAGES = frozenset({"generation", "asr"})
_ExpectedItem = tuple[Mapping[str, Any], int | None]


class ValidationLedgerError(RuntimeError):
    """Raised when a ledger transition is invalid for the durable item state."""


@dataclass(frozen=True)
class ValidationClaim:
    """An opaque, single-owner lease authorising one item's state transitions."""

    item_id: int
    rank: int
    token: str
    epoch: int


class ValidationLedger:
    """One atomically replaced JSON record per globally unique benchmark item ID.

    Each record has a leased :class:`ValidationClaim`.  Every mutation checks
    that claim while holding the item lock, so a rank that was rejected or whose
    lease became stale cannot overwrite a newer owner.  Locks are per item,
    letting disjoint distributed partitions publish independently.
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
        inputs: Mapping[str, Any],
        attempt_seed: int | None,
    ) -> ValidationClaim | None:
        """Lease an incomplete item or return ``None`` when it is unavailable.

        ``inputs`` and ``attempt_seed`` are the caller-current deterministic
        assignment.  They are compared before reusing any stored WAV/ASR output.
        """
        if rank < 0:
            raise ValueError("rank cannot be negative.")
        normalized_inputs = _normalize_inputs(inputs)
        with self._locked_record(item_id) as record:
            owner = record.get("owner")
            if isinstance(owner, Mapping) and not self._owner_is_stale(owner):
                return None
            if isinstance(owner, Mapping):
                record["owner"] = None

            changed = self._prepare_for_current_inputs(record, normalized_inputs, attempt_seed)
            if self._is_complete_record(record, normalized_inputs, attempt_seed) or self._exhausted(record):
                if changed:
                    self._write_record(item_id, record)
                return None

            epoch = int(record.get("claim_epoch", 0)) + 1
            token = secrets.token_urlsafe(32)
            record["claim_epoch"] = epoch
            record["owner"] = {
                "rank": rank,
                "token": token,
                "epoch": epoch,
                "claimed_at": _timestamp(),
            }
            self._touch(record)
            self._write_record(item_id, record)
            return ValidationClaim(item_id=_item_id(item_id), rank=rank, token=token, epoch=epoch)

    def record_generation(
        self,
        item_id: int,
        *,
        claim: ValidationClaim | None,
        wav_sha256: str,
        path: str | Path,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Record a generated WAV only for the live owner and legal source state."""
        actual_path = self._resolve_wav_path(path)
        if not actual_path.is_file():
            raise FileNotFoundError(f"Generated WAV is missing: {actual_path}")
        actual_hash = sha256_file(actual_path)
        if actual_hash != wav_sha256:
            raise ValueError(f"Generated WAV hash does not match for {actual_path}.")
        with self._locked_record(item_id) as record:
            self._authorize(record, item_id, claim)
            if record["generation"].get("status") not in {"pending", "failed"}:
                raise ValidationLedgerError(f"Invalid generation state for item {_item_id(item_id)}.")
            if record["asr"].get("status") != "pending":
                raise ValidationLedgerError(f"Invalid ASR state before generation for item {_item_id(item_id)}.")
            if self._attempt_cap_reached(record, "generation"):
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
            self._touch(record)
            self._write_record(item_id, record)

    def record_asr(
        self,
        item_id: int,
        *,
        claim: ValidationClaim | None,
        hypothesis: str,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Persist an ASR success, including a valid empty-string hypothesis."""
        if not isinstance(hypothesis, str):
            raise TypeError("ASR hypothesis must be a string.")
        with self._locked_record(item_id) as record:
            self._authorize(record, item_id, claim)
            if record["generation"].get("status") != "success" or not self._wav_is_valid(record):
                raise ValidationLedgerError(
                    f"Cannot record ASR for item {_item_id(item_id)} without a successful generation."
                )
            if record["asr"].get("status") not in {"pending", "failed"}:
                raise ValidationLedgerError(f"Invalid ASR state for item {_item_id(item_id)}.")
            if self._attempt_cap_reached(record, "asr"):
                raise ValidationLedgerError(f"ASR retry cap reached for item {_item_id(item_id)}.")
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

    def record_failure(
        self,
        item_id: int,
        *,
        claim: ValidationClaim | None,
        stage: str,
        exception: BaseException | str,
    ) -> None:
        """Persist a legal stage failure, releasing ownership for its bounded retry."""
        if stage not in _STAGES:
            raise ValueError(f"Unknown validation stage: {stage!r}")
        with self._locked_record(item_id) as record:
            self._authorize(record, item_id, claim)
            if self._attempt_cap_reached(record, stage):
                raise ValidationLedgerError(f"{stage.title()} retry cap reached for item {_item_id(item_id)}.")
            error = _exception_record(exception)
            if stage == "generation":
                if record["generation"].get("status") not in {"pending", "failed"}:
                    raise ValidationLedgerError(f"Invalid generation state for item {_item_id(item_id)}.")
                if record["asr"].get("status") != "pending":
                    raise ValidationLedgerError(f"Invalid ASR state before generation for item {_item_id(item_id)}.")
                record["attempts"]["generation"] += 1
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
                if record["generation"].get("status") != "success" or not self._wav_is_valid(record):
                    raise ValidationLedgerError(
                        f"Cannot record ASR failure for item {_item_id(item_id)} without a successful generation."
                    )
                if record["asr"].get("status") not in {"pending", "failed"}:
                    raise ValidationLedgerError(f"Invalid ASR state for item {_item_id(item_id)}.")
                record["attempts"]["asr"] += 1
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

    def is_complete(self, item_id: int, *, inputs: Mapping[str, Any], attempt_seed: int | None) -> bool:
        """Return whether this caller's current deterministic assignment is complete."""
        normalized_inputs = _normalize_inputs(inputs)
        path = self.item_path(item_id)
        if not path.is_file():
            return False
        with self._locked_record(item_id, create=False) as record:
            return self._is_complete_record(record, normalized_inputs, attempt_seed)

    def complete_ids(
        self,
        expected_items: Mapping[int, _ExpectedItem] | Callable[[int], _ExpectedItem | None],
    ) -> set[int]:
        """Return only items valid against evaluator-current inputs for every ID.

        ``expected_items`` is either an ID-to-``(inputs, seed)`` mapping or a
        callback returning that tuple (or ``None`` to omit an ID).
        """
        complete: set[int] = set()
        for path in sorted(self.items_dir.glob("*.json")):
            try:
                item_id = int(path.stem)
            except ValueError:
                continue
            expected = expected_items(item_id) if callable(expected_items) else expected_items.get(item_id)
            if expected is None:
                continue
            inputs, attempt_seed = _expected_item(expected)
            if self.is_complete(item_id, inputs=inputs, attempt_seed=attempt_seed):
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
                    raise ValidationLedgerError(f"Ledger item record does not exist: {path}")
                yield record
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _new_record(self, item_id: int) -> dict[str, Any]:
        now = _timestamp()
        record: dict[str, Any] = {
            "version": 2,
            "id": item_id,
            "fingerprints": {"generation": self.generation_fingerprint, "asr": self.asr_fingerprint},
            "inputs": {},
            "attempt_seed": None,
            "input_fingerprint": fingerprint({"inputs": {}, "attempt_seed": None}),
            "attempts": {"generation": 0, "asr": 0},
            "generation": {},
            "asr": {},
            "owner": None,
            "claim_epoch": 0,
            "created_at": now,
            "updated_at": now,
        }
        self._reset_generation(record, reset_attempts=False)
        return record

    def _prepare_for_current_inputs(
        self,
        record: dict[str, Any],
        normalized_inputs: Mapping[str, Any],
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
            changed = True
        elif asr_changed:
            fingerprints["asr"] = self.asr_fingerprint
            self._reset_asr(record, reset_attempts=True)
            changed = True

        current_input_fingerprint = fingerprint({"inputs": normalized_inputs, "attempt_seed": attempt_seed})
        if (
            record.get("inputs") != normalized_inputs
            or record.get("attempt_seed") != attempt_seed
            or record.get("input_fingerprint") != current_input_fingerprint
        ):
            record["inputs"] = dict(normalized_inputs)
            record["attempt_seed"] = attempt_seed
            record["input_fingerprint"] = current_input_fingerprint
            self._reset_generation(record, reset_attempts=True)
            changed = True
        elif record["generation"].get("status") == "success" and not self._wav_is_valid(record):
            self._reset_generation(record, reset_attempts=True)
            changed = True
        elif record["asr"].get("status") == "success" and not self._asr_is_valid(record):
            self._reset_asr(record, reset_attempts=True)
            changed = True
        if changed:
            self._touch(record)
        return changed

    def _authorize(self, record: Mapping[str, Any], item_id: int, claim: ValidationClaim | None) -> None:
        if not isinstance(claim, ValidationClaim):
            raise ValidationLedgerError(f"A live validation claim is required for item {_item_id(item_id)}.")
        if claim.item_id != _item_id(item_id):
            raise ValidationLedgerError(f"Validation claim item does not match item {_item_id(item_id)}.")
        owner = record.get("owner")
        if not isinstance(owner, Mapping) or self._owner_is_stale(owner):
            raise ValidationLedgerError(f"Validation claim is no longer live for item {_item_id(item_id)}.")
        if owner.get("rank") != claim.rank:
            raise ValidationLedgerError(
                f"Validation claim owner rank does not match current owner for item {_item_id(item_id)}."
            )
        if owner.get("epoch") != claim.epoch or not isinstance(owner.get("token"), str):
            raise ValidationLedgerError(f"Validation claim does not match current owner for item {_item_id(item_id)}.")
        if not secrets.compare_digest(owner["token"], claim.token):
            raise ValidationLedgerError(f"Validation claim does not match current owner for item {_item_id(item_id)}.")

    def _is_complete_record(
        self,
        record: Mapping[str, Any],
        expected_inputs: Mapping[str, Any],
        expected_seed: int | None,
    ) -> bool:
        fingerprints = record.get("fingerprints")
        return bool(
            isinstance(fingerprints, Mapping)
            and fingerprints.get("generation") == self.generation_fingerprint
            and fingerprints.get("asr") == self.asr_fingerprint
            and self._inputs_are_current(record, expected_inputs, expected_seed)
            and record.get("generation", {}).get("status") == "success"
            and record.get("asr", {}).get("status") == "success"
            and self._wav_is_valid(record)
            and self._asr_is_valid(record)
        )

    def _inputs_are_current(
        self,
        record: Mapping[str, Any],
        expected_inputs: Mapping[str, Any],
        expected_seed: int | None,
    ) -> bool:
        if record.get("inputs") != expected_inputs or record.get("attempt_seed") != expected_seed:
            return False
        try:
            expected_fingerprint = fingerprint({"inputs": dict(expected_inputs), "attempt_seed": expected_seed})
        except (TypeError, ValueError):
            return False
        return record.get("input_fingerprint") == expected_fingerprint

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
        generation = record.get("generation", {})
        asr = record.get("asr", {})
        return bool(
            (generation.get("status") in {"pending", "failed"} and self._attempt_cap_reached(record, "generation"))
            or (
                generation.get("status") == "success"
                and asr.get("status") in {"pending", "failed"}
                and self._attempt_cap_reached(record, "asr")
            )
        )

    def _attempt_cap_reached(self, record: Mapping[str, Any], stage: str) -> bool:
        attempts = record.get("attempts", {})
        return attempts.get(stage, 0) >= self.max_attempts

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


def _expected_item(value: _ExpectedItem) -> _ExpectedItem:
    if not isinstance(value, tuple) or len(value) != 2 or not isinstance(value[0], Mapping):
        raise ValueError("Expected validation item must be an (inputs, attempt_seed) tuple.")
    return value


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exception_record(exception: BaseException | str) -> dict[str, str]:
    if isinstance(exception, BaseException):
        return {"type": type(exception).__name__, "message": str(exception)}
    return {"type": "Error", "message": str(exception)}
