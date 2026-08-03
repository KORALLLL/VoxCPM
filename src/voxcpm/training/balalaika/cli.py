"""Operator-facing command routing for Balalaika LoRA training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol

from .config import BalalaikaConfig


@dataclass(frozen=True)
class CommandResult:
    """One JSON-serializable command outcome."""

    command: str
    values: Mapping[str, Any]


class BalalaikaCommands(Protocol):
    """Typed orchestration boundary kept separate from argument parsing."""

    def pin(self, config: BalalaikaConfig) -> CommandResult: ...

    def prepare(self, config: BalalaikaConfig) -> CommandResult: ...

    def memorize(self, config: BalalaikaConfig, *, smoke: bool) -> CommandResult: ...

    def approve(self, config: BalalaikaConfig, *, wandb_run_id: str) -> CommandResult: ...

    def train(
        self,
        config: BalalaikaConfig,
        *,
        stage: int,
        smoke: bool,
        resume: Path | None,
        stage1_checkpoint: Path | None,
    ) -> CommandResult: ...

    def validate(self, config: BalalaikaConfig, *, checkpoint: Path, smoke: bool) -> CommandResult: ...

    def audit(self, config: BalalaikaConfig) -> CommandResult: ...


class CliSafetyError(RuntimeError):
    """A requested operation is unsafe or lacks a required artifact."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="voxcpm-balalaika")
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("pin")
    subparsers.add_parser("prepare")

    memorize = subparsers.add_parser("memorize")
    memorize.add_argument("--smoke", action="store_true")

    approve = subparsers.add_parser("approve")
    approve.add_argument("--wandb-run-id", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--stage", type=int, choices=(1, 2), required=True)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--resume", type=Path)
    train.add_argument("--stage1-checkpoint", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--checkpoint", type=Path)
    validate.add_argument("--smoke", action="store_true")

    subparsers.add_parser("audit")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    commands: BalalaikaCommands | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Parse one command, enforce launch guards, and print its JSON result."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        config = BalalaikaConfig.load(args.config)
        command_set = commands or _production_commands()
        environment = os.environ if environ is None else environ
        result = _dispatch(args, config, command_set, environment)
    except (CliSafetyError, OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"command": result.command, **dict(result.values)}, sort_keys=True, default=str))
    return 0


def _dispatch(
    args: argparse.Namespace,
    config: BalalaikaConfig,
    commands: BalalaikaCommands,
    environ: Mapping[str, str],
) -> CommandResult:
    command = args.command
    if command == "pin":
        return commands.pin(config)
    if command == "prepare":
        return commands.prepare(config)
    if command == "memorize":
        if args.smoke:
            raise CliSafetyError(
                "memorize --smoke is synthetic-only; run scripts/smoke_balalaika_accelerate.py directly"
            )
        _require_world_size(environ, smoke=args.smoke)
        return commands.memorize(config, smoke=args.smoke)
    if command == "approve":
        return commands.approve(config, wandb_run_id=args.wandb_run_id)
    if command == "train":
        _require_world_size(environ, smoke=args.smoke)
        resume = _existing_directory(args.resume, "resume checkpoint") if args.resume is not None else None
        if args.stage == 1:
            approval = config.output_dir / "memorization" / "memorization-approval.json"
            if not approval.is_file():
                raise CliSafetyError(f"stage 1 requires a matching memorization approval: {approval}")
            stage1_checkpoint = None
        else:
            candidate = args.stage1_checkpoint
            if candidate is None and resume is None:
                candidate = _latest_stage1_checkpoint(config)
            stage1_checkpoint = _completed_stage1(candidate, config.stage1.epochs) if candidate is not None else None
        return commands.train(
            config,
            stage=args.stage,
            smoke=args.smoke,
            resume=resume,
            stage1_checkpoint=stage1_checkpoint,
        )
    if command == "validate":
        _require_world_size(environ, smoke=args.smoke)
        if args.checkpoint is None:
            raise CliSafetyError("validate requires --checkpoint")
        return commands.validate(
            config,
            checkpoint=_existing_directory(args.checkpoint, "validation checkpoint"),
            smoke=args.smoke,
        )
    if command == "audit":
        return commands.audit(config)
    raise AssertionError(f"unhandled command: {command}")


def _require_world_size(environ: Mapping[str, str], *, smoke: bool) -> None:
    if smoke:
        return
    raw = environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise CliSafetyError(f"WORLD_SIZE must be an integer, found {raw!r}") from error
    if world_size != 8:
        raise CliSafetyError(f"large Balalaika commands require exactly 8 processes; found {world_size}")


def _existing_directory(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise CliSafetyError(f"{label} does not exist: {resolved}")
    return resolved


def _completed_stage1(path: Path, stage1_epochs: int) -> Path:
    resolved = _existing_directory(path, "stage-1 checkpoint")
    adapter = resolved / "adapter_model.safetensors"
    metadata_path = resolved / "metadata.json"
    if not adapter.is_file() or not metadata_path.is_file():
        raise CliSafetyError(f"stage 2 requires a completed stage-1 boundary adapter: {resolved}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CliSafetyError(f"stage-1 checkpoint metadata is unreadable: {metadata_path}") from error
    expected = {
        "stage": "stage1",
        "epoch": stage1_epochs - 1,
        "boundary": 8,
        "checkpoint_kind": "boundary",
    }
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in expected.items()):
        raise CliSafetyError(f"stage 2 requires a completed stage-1 boundary adapter: {resolved}")
    return resolved


def _latest_stage1_checkpoint(config: BalalaikaConfig) -> Path:
    root = config.output_dir / "stage1" / "checkpoints"
    pointer_path = root / "latest.json"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CliSafetyError("stage 2 requires a completed stage-1 adapter checkpoint") from error
    name = pointer.get("checkpoint") if isinstance(pointer, dict) else None
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise CliSafetyError(f"stage-1 latest pointer is malformed: {pointer_path}")
    return root / name


def _production_commands() -> BalalaikaCommands:
    from .workflow import ProductionCommands

    return ProductionCommands()


if __name__ == "__main__":
    raise SystemExit(main())
