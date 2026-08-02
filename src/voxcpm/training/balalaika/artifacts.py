"""Small, durable primitives for training artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace *path* with a JSON representation of *value*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot fingerprint {type(value).__name__}")


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for a JSON-compatible value."""
    canonical = json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
