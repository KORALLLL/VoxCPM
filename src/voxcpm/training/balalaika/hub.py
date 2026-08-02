"""Immutable, authenticated Hugging Face Hub input pinning."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from .artifacts import atomic_json, sha256_file
from .config import HubConfig


class CommandRunner(Protocol):
    """Run a command as an argv sequence, allowing the CLI to use its own auth."""

    def run(self, argv: Sequence[str]) -> CompletedProcess[str]: ...


class HubPin(BaseModel):
    """A fully resolved Hub repository and the local files that implement it."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["model", "dataset"]
    repo_id: str
    revision: str
    local_dir: Path
    files: dict[str, str]


def pin_hub_inputs(config: HubConfig, runner: CommandRunner) -> dict[str, HubPin]:
    """Resolve, download, hash, and atomically record each configured Hub input."""
    manifest_path = config.local_dir / "hub-pins.json"
    existing = _load_manifest(manifest_path)
    inputs = [
        ("model", "model", config.model_repo_id, config.local_dir / "model"),
        ("benchmark", "dataset", config.benchmark_repo_id, config.local_dir / "benchmark"),
    ]
    if config.gigaam_repo_id is not None:
        inputs.append(("gigaam", "model", config.gigaam_repo_id, config.local_dir / "gigaam"))

    pins = {
        name: _pin_repository(
            name=name,
            kind=kind,
            repo_id=repo_id,
            local_dir=local_dir,
            existing=existing.get(name),
            runner=runner,
        )
        for name, kind, repo_id, local_dir in inputs
    }
    atomic_json(manifest_path, {name: pin.model_dump(mode="json") for name, pin in pins.items()})
    return pins


def _pin_repository(
    *,
    name: str,
    kind: Literal["model", "dataset"],
    repo_id: str,
    local_dir: Path,
    existing: HubPin | None,
    runner: CommandRunner,
) -> HubPin:
    revision = _resolve_revision(kind, repo_id, runner)
    current_files = _hash_download(local_dir)
    if _matches(existing, kind=kind, repo_id=repo_id, revision=revision, local_dir=local_dir, files=current_files):
        return existing

    argv = ["hf", "download", repo_id]
    if kind == "dataset":
        argv.extend(("--repo-type", "dataset"))
    argv.extend(("--revision", revision, "--local-dir", str(local_dir)))
    _run(runner, argv)
    return HubPin(kind=kind, repo_id=repo_id, revision=revision, local_dir=local_dir, files=_hash_download(local_dir))


def _resolve_revision(kind: Literal["model", "dataset"], repo_id: str, runner: CommandRunner) -> str:
    collection = "models" if kind == "model" else "datasets"
    result = _run(runner, ("hf", collection, "info", repo_id))
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Hub {kind} {repo_id!r} did not return JSON metadata.") from error
    revision = payload.get("sha") if isinstance(payload, dict) else None
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Hub {kind} {repo_id!r} did not return an immutable sha.")
    return revision


def _run(runner: CommandRunner, argv: Sequence[str]) -> CompletedProcess[str]:
    result = runner.run(argv)
    if result.returncode != 0:
        raise CalledProcessError(result.returncode, list(argv), result.stdout, result.stderr)
    return result


def _load_manifest(path: Path) -> dict[str, HubPin]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    pins: dict[str, HubPin] = {}
    for name, pin in value.items():
        try:
            pins[name] = HubPin.model_validate(pin)
        except (TypeError, ValueError):
            continue
    return pins


def _hash_download(local_dir: Path) -> dict[str, str]:
    if not local_dir.is_dir():
        return {}
    return {
        path.relative_to(local_dir).as_posix(): sha256_file(path)
        for path in sorted(local_dir.rglob("*"))
        if path.is_file()
    }


def _matches(
    pin: HubPin | None,
    *,
    kind: Literal["model", "dataset"],
    repo_id: str,
    revision: str,
    local_dir: Path,
    files: dict[str, str],
) -> bool:
    return pin is not None and (
        pin.kind,
        pin.repo_id,
        pin.revision,
        pin.local_dir,
        pin.files,
    ) == (kind, repo_id, revision, local_dir, files)
