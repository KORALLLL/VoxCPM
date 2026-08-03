"""Typed operator workflows behind the thin Balalaika CLI."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
from subprocess import CompletedProcess
from tempfile import mkdtemp
from typing import Any

from .artifacts import sha256_file
from .artifacts import fingerprint
from .asr import GigaAMRNNT
from .checkpoint import CheckpointManager
from .cli import CommandResult
from .config import BalalaikaConfig, DataConfig
from .hub import CommandRunner, HubPin, pin_hub_inputs
from .index import BuildExpectations, IndexAudit, build_index
from .evaluation import DistributedEvaluator
from .ledger import ValidationLedger
from .memorization import approve_memorization, run_memorization, verify_approval
from .metrics import BenchmarkRow
from .runtime import AccelerateRuntime
from .selection import PromptSample, SelectedSample, SelectionBundle, create_selection_manifests
from .tracking import create_run_manager
from .trainer import BalalaikaTrainer, EvaluationBoundary

_JSON = Mapping[str, Any]


class SubprocessRunner(CommandRunner):
    """Execute an argv sequence without a shell, preserving the operator's auth."""

    def run(self, argv: Sequence[str]) -> CompletedProcess[str]:
        return subprocess.run(list(argv), check=False, capture_output=True, text=True)


def load_build_expectations(config: DataConfig) -> BuildExpectations:
    """Derive immutable index expectations from the trusted corpus manifests."""
    if config.verification_manifest is None or config.combined_metadata is None:
        raise ValueError("data verification_manifest and combined_metadata are required for preparation")
    verification = _read_json(config.verification_manifest, "corpus verification manifest")
    combined = _read_json(config.combined_metadata, "combined-sidecar metadata")
    if verification.get("status") != "ok":
        raise ValueError("corpus verification manifest is not successful")
    stats = _mapping(verification.get("stats"), "corpus verification stats")
    shards = verification.get("shards")
    if not isinstance(shards, list) or len(shards) != config.expected_shards:
        raise ValueError(
            f"corpus verification must contain {config.expected_shards} shards; found "
            f"{len(shards) if isinstance(shards, list) else 'malformed'}"
        )
    source_hashes: dict[str, str] = {}
    source_rows = 0
    for item in shards:
        row = _mapping(item, "corpus shard verification")
        name = row.get("shard")
        digest = row.get("sha256")
        samples = row.get("samples")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("shard_")
            or not name.endswith(".tar")
        ):
            raise ValueError("corpus verification contains an invalid shard name")
        _require_sha256(digest, f"source shard {name}")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise ValueError(f"source shard {name} has an invalid sample count")
        manifest_path = Path(config.corpus_root) / "manifests" / Path(name).with_suffix(".json").name
        manifest = _read_json(manifest_path, f"augmented shard manifest {name}")
        shard_id = name.removeprefix("shard_").removesuffix(".tar")
        if manifest.get("shard_id") != shard_id or manifest.get("samples") != samples:
            raise ValueError(f"augmented shard manifest does not match verified shard {name}")
        if manifest.get("source_sha256") != digest:
            raise ValueError(f"augmented shard manifest is not bound to verified source {name}")
        augmented_digest = manifest.get("sha256")
        _require_sha256(augmented_digest, f"augmented shard {name}")
        relative = f"train/{name}"
        if relative in source_hashes:
            raise ValueError(f"corpus verification contains duplicate shard {name}")
        source_hashes[relative] = augmented_digest
        source_rows += samples
    declared_rows = stats.get("samples")
    if declared_rows != config.expected_rows or source_rows != config.expected_rows:
        raise ValueError(
            f"corpus row expectation mismatch: expected {config.expected_rows}, "
            f"manifest={declared_rows}, shards={source_rows}"
        )
    if combined.get("complete") is not True or combined.get("rows") != config.expected_rows:
        raise ValueError("combined-sidecar metadata is incomplete or has the wrong row count")
    provenance = _mapping(
        _mapping(combined.get("inputs"), "combined-sidecar inputs").get("provenance_binding"),
        "ROVER provenance binding",
    )
    if provenance.get("trusted_release") is not True:
        raise ValueError("combined-sidecar metadata does not bind a trusted ROVER release")
    rover_hash = provenance.get("rover_archive_sha256")
    combined_hash = _mapping(combined.get("output"), "combined-sidecar output").get("sha256")
    _require_sha256(rover_hash, "ROVER archive")
    _require_sha256(combined_hash, "combined sidecar")
    return BuildExpectations(
        source_shard_count=config.expected_shards,
        source_row_count=config.expected_rows,
        rover_row_count=config.expected_rows,
        combined_row_count=config.expected_rows,
        rover_archive_sha256=rover_hash,
        combined_sidecar_sha256=combined_hash,
        source_tar_sha256=source_hashes,
    )


@dataclass
class ProductionCommands:
    """Production orchestration using the pipeline's typed module interfaces."""

    runner: CommandRunner = field(default_factory=SubprocessRunner)
    hub_pinner: Callable[[Any, CommandRunner], dict[str, HubPin]] = pin_hub_inputs
    expectation_loader: Callable[[DataConfig], BuildExpectations] = load_build_expectations
    index_builder: Callable[[DataConfig, BuildExpectations], IndexAudit] = build_index
    selection_builder: Callable[[str | Path, str | Path, str | Path, int], SelectionBundle] = create_selection_manifests
    generation_validator: Callable[[BalalaikaConfig, IndexAudit, SelectionBundle], None] | None = None
    runtime_factory: Callable[[Any], Any] = AccelerateRuntime.create
    memorization_runner: Callable[[Any, Any], Any] = run_memorization
    approval_writer: Callable[[str | Path, str, str], Path] = approve_memorization
    training_runner: Callable[..., Path] | None = None
    validation_runner: Callable[..., Mapping[str, Any]] | None = None

    def pin(self, config: BalalaikaConfig) -> CommandResult:
        pins = self.hub_pinner(config.hub, self.runner)
        return CommandResult(
            "pin",
            {
                "status": "complete",
                "pins": {name: pin.model_dump(mode="json") for name, pin in pins.items()},
            },
        )

    def prepare(self, config: BalalaikaConfig) -> CommandResult:
        _verified_pins(config)
        expectations = self.expectation_loader(config.data)
        benchmark_path = config.hub.local_dir / "benchmark" / config.hub.benchmark_file
        with _PreparationTransaction(config) as transaction:
            audit = self.index_builder(config.data, expectations)
            selection = self.selection_builder(
                audit.index_path,
                benchmark_path,
                config.selection_dir,
                config.runtime.seed,
            )
            validator = self.generation_validator or _validate_prepared_generation
            validator(config, audit, selection)
            transaction.commit()
        return CommandResult(
            "prepare",
            {
                "status": "complete",
                "index_path": audit.index_path,
                "audit_path": audit.audit_path,
                "fingerprint": audit.fingerprint,
                "joined_rows": audit.total_rows,
                "eligible_rows": audit.eligible_rows,
                "stage1_rows": audit.stage1_rows,
                "stage2_rows": audit.stage2_rows,
                "excluded_null_agreement": audit.excluded_null_agreement,
                "selection_fingerprint": selection.fingerprint,
                "memorization_samples": len(selection.memorization),
                "validation_prompts": len(selection.prompts),
                "benchmark_assignments": len(selection.benchmark_prompt_by_id),
            },
        )

    def memorize(self, config: BalalaikaConfig, *, smoke: bool) -> CommandResult:
        del smoke
        runtime = self.runtime_factory(config.memorization)
        failure: BaseException | None = None
        try:
            result = self.memorization_runner(config, runtime)
            return CommandResult(
                "memorize",
                {
                    "status": result.status,
                    "result_path": result.result_path,
                    "checkpoint": result.checkpoint,
                    "wandb_run_id": result.wandb_run_id,
                    "large_training_started": result.large_training_started,
                },
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            _close_runtime(runtime, failure)

    def approve(self, config: BalalaikaConfig, *, wandb_run_id: str) -> CommandResult:
        approver = os.environ.get("USER") or os.environ.get("LOGNAME")
        if not approver:
            raise ValueError("approval requires USER or LOGNAME to identify the operator")
        path = self.approval_writer(config.output_dir / "memorization", wandb_run_id, approver)
        return CommandResult("approve", {"status": "approved", "approval_path": path})

    def train(
        self,
        config: BalalaikaConfig,
        *,
        stage: int,
        smoke: bool,
        resume: Path | None,
        stage1_checkpoint: Path | None,
    ) -> CommandResult:
        runner = self.training_runner or _run_configured_training
        checkpoint = runner(
            config,
            stage=stage,
            smoke=smoke,
            resume=resume,
            stage1_checkpoint=stage1_checkpoint,
        )
        return CommandResult("train", {"status": "complete", "stage": stage, "checkpoint": checkpoint})

    def validate(self, config: BalalaikaConfig, *, checkpoint: Path, smoke: bool) -> CommandResult:
        runner = self.validation_runner or _run_configured_validation
        return CommandResult("validate", runner(config, checkpoint=checkpoint, smoke=smoke))

    def audit(self, config: BalalaikaConfig) -> CommandResult:
        audit_path = config.data.index_dir / "balalaika-index-audit.json"
        index_path = config.data.index_dir / "balalaika-index.sqlite3"
        _verified_pins(config)
        expectations = self.expectation_loader(config.data)
        index_values = _deep_audit_index(config, index_path, audit_path, expectations)
        selections = _deep_audit_selection(config, index_path, index_values["fingerprint"])
        return CommandResult(
            "audit",
            {
                "status": "complete",
                "source_shards": config.data.expected_shards,
                "joined_rows": index_values["total_rows"],
                "eligible_rows": index_values["eligible_rows"],
                "stage1_rows": index_values["stage1_rows"],
                "stage2_rows": index_values["stage2_rows"],
                "excluded_null_agreement": index_values["excluded_null_agreement"],
                "index_bytes": index_path.stat().st_size,
                "fingerprint": index_values["fingerprint"],
                "audit_identity": index_values["audit_identity"],
                **selections,
            },
        )


class _PreparationTransaction:
    """Publish index and selection as one rollback-capable filesystem generation."""

    def __init__(self, config: BalalaikaConfig):
        self._roots = (config.data.index_dir.resolve(), config.selection_dir.resolve())
        if self._roots[0] == self._roots[1]:
            raise ValueError("index_dir and selection_dir must be distinct generation roots")
        self._lock_path = config.output_dir.resolve() / ".prepare.lock"
        self._lock_descriptor: int | None = None
        self._backups: dict[Path, Path] = {}
        self._committed = False

    def __enter__(self) -> "_PreparationTransaction":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._lock_descriptor)
            self._lock_descriptor = None
            raise RuntimeError("another Balalaika preparation is already active") from error
        try:
            for root in self._roots:
                root.parent.mkdir(parents=True, exist_ok=True)
                if root.exists():
                    backup = Path(mkdtemp(prefix=f".{root.name}.previous-", dir=root.parent))
                    backup.rmdir()
                    os.replace(root, backup)
                    self._backups[root] = backup
        except BaseException:
            self._restore_previous()
            self._unlock()
            raise
        return self

    def commit(self) -> None:
        self._committed = True

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        try:
            if self._committed and _type is None:
                for backup in self._backups.values():
                    shutil.rmtree(backup)
            else:
                self._restore_previous()
        finally:
            self._unlock()

    def _restore_previous(self) -> None:
        for root in reversed(self._roots):
            if root.exists():
                shutil.rmtree(root)
            backup = self._backups.get(root)
            if backup is not None and backup.exists():
                os.replace(backup, root)

    def _unlock(self) -> None:
        if self._lock_descriptor is not None:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None


def _validate_prepared_generation(
    config: BalalaikaConfig,
    audit: IndexAudit,
    selection: SelectionBundle,
) -> None:
    index_path = config.data.index_dir / "balalaika-index.sqlite3"
    audit_path = config.data.index_dir / "balalaika-index-audit.json"
    if audit.index_path.resolve() != index_path.resolve() or not index_path.is_file():
        raise ValueError("prepared generation index is missing or published at the wrong path")
    if audit.audit_path.resolve() != audit_path.resolve() or not audit_path.is_file():
        raise ValueError("prepared generation index audit is missing or published at the wrong path")
    declared_audit = _read_json(audit_path, "prepared generation index audit")
    if declared_audit.get("index_sha256") != sha256_file(index_path):
        raise ValueError("prepared generation index hash does not match its audit")
    index_fingerprint = _index_fingerprint(config)
    if declared_audit.get("fingerprint") != index_fingerprint or audit.fingerprint != index_fingerprint:
        raise ValueError("prepared generation index identity is inconsistent")
    selection_summary = _deep_audit_selection(config, index_path, index_fingerprint)
    if selection_summary["selection_fingerprint"] != selection.fingerprint:
        raise ValueError("prepared generation selection identity is inconsistent")


def _run_configured_training(
    config: BalalaikaConfig,
    *,
    stage: int,
    smoke: bool,
    resume: Path | None,
    stage1_checkpoint: Path | None,
) -> Path:
    if smoke:
        raise ValueError("production --smoke uses scripts/smoke_balalaika_accelerate.py, not corpus training")
    pins = _verified_pins(config)
    data_fingerprint = _index_fingerprint(config)
    selection = _load_selection(config.selection_dir)
    lora_fingerprint = fingerprint(config.lora.model_dump(mode="json"))
    approval_path: Path | None = None
    approval_expected: Mapping[str, Any] | None = None
    if stage == 1:
        approval_path = config.output_dir / "memorization" / "memorization-approval.json"
        approval_expected = _approval_expectations(
            config,
            pins=pins,
            data_fingerprint=data_fingerprint,
            selection_fingerprint=selection.fingerprint,
            lora_fingerprint=lora_fingerprint,
        )
        # The operator gate is verified before runtime, tracking, model, or dataset setup.
        verify_approval(approval_path, approval_expected)

    runtime = AccelerateRuntime.create(config.runtime)
    failure: BaseException | None = None
    try:
        return _run_training_with_runtime(
            config,
            stage=stage,
            resume=resume,
            stage1_checkpoint=stage1_checkpoint,
            pins=pins,
            data_fingerprint=data_fingerprint,
            selection=selection,
            lora_fingerprint=lora_fingerprint,
            approval_path=approval_path,
            approval_expected=approval_expected,
            runtime=runtime,
        )
    except BaseException as error:
        failure = error
        raise
    finally:
        _close_runtime(runtime, failure)


def _run_training_with_runtime(
    config: BalalaikaConfig,
    *,
    stage: int,
    resume: Path | None,
    stage1_checkpoint: Path | None,
    pins: Mapping[str, Mapping[str, Any]],
    data_fingerprint: str,
    selection: SelectionBundle,
    lora_fingerprint: str,
    approval_path: Path | None,
    approval_expected: Mapping[str, Any] | None,
    runtime: Any,
) -> Path:
    stage_name = f"stage{stage}"
    stage_root = config.output_dir / stage_name
    run_state_path = stage_root / "wandb-run.json"
    run_manager = create_run_manager(
        is_main_process=runtime.rank == 0,
        config=config,
        job_type=stage_name,
        run_state_path=run_state_path,
    )
    runtime.barrier()
    run_state = _read_json(run_state_path, f"{stage_name} W&B run state")
    run_id = run_state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"{stage_name} W&B run state has no run ID")
    group = config.wandb.group
    if not isinstance(group, str) or not group:
        raise ValueError("production training requires an explicit W&B group")
    stage_config = config.stage1 if stage == 1 else config.stage2
    optimization_fingerprint = fingerprint(
        {
            "stage": stage_config.model_dump(mode="json"),
            "runtime": config.runtime.model_dump(mode="json"),
            "optimization": config.optimization.model_dump(mode="json"),
        }
    )
    identity = {
        "base_revision": pins["model"]["revision"],
        "evaluator_revision": pins["gigaam"]["revision"],
        "data_fingerprint": data_fingerprint,
        "selection_fingerprint": selection.fingerprint,
        "lora_fingerprint": lora_fingerprint,
        "optimization_fingerprint": optimization_fingerprint,
        "wandb_run_id": run_id,
        "wandb_group": group,
    }
    rows = _load_benchmark_rows(config.hub.local_dir / "benchmark" / config.hub.benchmark_file)

    def evaluator_factory(boundary: EvaluationBoundary, checkpoint: Path) -> DistributedEvaluator:
        boundary_root = (
            config.output_dir
            / "validation"
            / boundary.stage
            / f"epoch-{boundary.epoch + 1:02d}-boundary-{boundary.boundary:02d}-{checkpoint.name}"
        )
        ledger = ValidationLedger(
            boundary_root,
            generation_fingerprint=fingerprint(config.generation.model_dump(mode="json")),
            asr_fingerprint=fingerprint(pins["gigaam"]),
            max_attempts=config.validation.retries,
        )
        gigaam_dir = Path(str(pins["gigaam"]["local_dir"]))
        return DistributedEvaluator(
            runtime=runtime,
            rows=rows,
            selection=selection,
            ledger=ledger,
            asr_factory=lambda device_id: GigaAMRNNT(gigaam_dir, device_id, fingerprint(pins["gigaam"])),
            run_manager=run_manager,
            expected_item_count=config.validation.benchmark_size,
            validation_root=config.output_dir / "validation",
        )

    trainer = BalalaikaTrainer(
        config,
        runtime,
        checkpoint_manager=CheckpointManager(stage_root / "checkpoints"),
        evaluator_factory=evaluator_factory,
        identity=identity,
        approval_verifier=verify_approval if stage == 1 else None,
        approval_path=approval_path,
        approval_expected=approval_expected,
        accumulation=config.runtime.accumulation,
        workers=config.runtime.workers,
        sampler_seed=config.runtime.seed,
        warmup_fraction=config.optimization.warmup_fraction,
        weight_decay=config.optimization.weight_decay,
        max_grad_norm=config.optimization.max_grad_norm,
        loss_weights=config.optimization.loss_weights,
        resume_checkpoint=resume,
        stage1_checkpoint=stage1_checkpoint,
    )
    failure: BaseException | None = None
    try:
        return trainer.run_stage(stage)
    except BaseException as error:
        failure = error
        raise
    finally:
        _finish_run_manager(run_manager, failure)


def _finish_run_manager(run_manager: Any, active_failure: BaseException | None) -> None:
    try:
        run_manager.finish()
    except BaseException:
        if active_failure is None:
            raise


def _close_runtime(runtime: Any, active_failure: BaseException | None) -> None:
    try:
        runtime.close()
    except BaseException:
        if active_failure is None:
            raise


def _run_configured_validation(
    config: BalalaikaConfig,
    *,
    checkpoint: Path,
    smoke: bool,
) -> Mapping[str, Any]:
    del smoke
    pins = _verified_pins(config)
    selection = _load_selection(config.selection_dir)
    expected = {
        "base_revision": pins["model"]["revision"],
        "evaluator_revision": pins["gigaam"]["revision"],
        "data_fingerprint": _index_fingerprint(config),
        "selection_fingerprint": selection.fingerprint,
        "lora_fingerprint": fingerprint(config.lora.model_dump(mode="json")),
    }
    metadata = CheckpointManager(checkpoint.parent).verify(checkpoint, expected)
    return {
        "status": "complete",
        "checkpoint": checkpoint,
        "checkpoint_fingerprint": metadata["checkpoint_fingerprint"],
        "stage": metadata["stage"],
        "epoch": metadata["epoch"],
        "boundary": metadata["boundary"],
        "note": "checkpoint integrity and current input identities verified; boundary evaluation resumes in train",
    }


def _verified_pins(config: BalalaikaConfig) -> dict[str, dict[str, Any]]:
    manifest = _read_json(config.hub.local_dir / "hub-pins.json", "Hub pin manifest")
    expected_repos = {
        "model": ("model", config.hub.model_repo_id, config.hub.local_dir / "model"),
        "benchmark": ("dataset", config.hub.benchmark_repo_id, config.hub.local_dir / "benchmark"),
        "gigaam": ("model", config.hub.gigaam_repo_id, config.hub.local_dir / "gigaam"),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (kind, repo_id, local_dir) in expected_repos.items():
        pin = _mapping(manifest.get(name), f"{name} Hub pin")
        if pin.get("kind") != kind or pin.get("repo_id") != repo_id:
            raise ValueError(f"{name} Hub pin does not match configured repository")
        revision = pin.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError(f"{name} Hub pin has no immutable revision")
        if Path(str(pin.get("local_dir", ""))).resolve() != local_dir.resolve():
            raise ValueError(f"{name} Hub pin local directory does not match configuration")
        recorded_files = pin.get("files")
        if not isinstance(recorded_files, Mapping) or not recorded_files:
            raise ValueError(f"{name} Hub pin has no file hashes")
        actual_files = {
            path.relative_to(local_dir).as_posix(): sha256_file(path)
            for path in sorted(local_dir.rglob("*"))
            if path.is_file()
        }
        if dict(recorded_files) != actual_files:
            raise ValueError(f"{name} pinned files changed after download")
        result[name] = dict(pin)
    return result


def _deep_audit_index(
    config: BalalaikaConfig,
    index_path: Path,
    audit_path: Path,
    expectations: BuildExpectations,
) -> dict[str, Any]:
    if not index_path.is_file():
        raise ValueError(f"prepared index is absent: {index_path}")
    audit = _read_json(audit_path, "index audit")
    corpus_root = config.data.corpus_root.resolve()
    source_tars = sorted(corpus_root.glob("train/shard_*.tar"))
    actual_source_identities = {path.relative_to(corpus_root).as_posix() for path in source_tars}
    if actual_source_identities != set(expectations.source_tar_sha256):
        raise ValueError("current source tar inventory does not match immutable expectations")
    for path in source_tars:
        relative = path.relative_to(corpus_root).as_posix()
        if sha256_file(path) != expectations.source_tar_sha256[relative]:
            raise ValueError(f"source tar SHA-256 mismatch: {relative}")

    try:
        with sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True) as database:
            database.row_factory = sqlite3.Row
            database.execute("PRAGMA query_only=ON")
            integrity = database.execute("PRAGMA integrity_check").fetchall()
            if [tuple(row) for row in integrity] != [("ok",)]:
                raise ValueError(f"SQLite integrity_check failed: {[tuple(row) for row in integrity]}")
            metadata = {row["key"]: row["value"] for row in database.execute("SELECT key, value FROM metadata")}
            try:
                provenance = _mapping(json.loads(metadata["provenance"]), "index provenance")
            except (KeyError, json.JSONDecodeError) as error:
                raise ValueError("prepared index has malformed provenance metadata") from error
            stored_fingerprint = metadata.get("fingerprint")
            recomputed_fingerprint = fingerprint(provenance)
            if stored_fingerprint != recomputed_fingerprint:
                raise ValueError("prepared index fingerprint does not match canonical provenance")
            counts = database.execute("""
                SELECT
                    COUNT(*) AS total_rows,
                    SUM(CASE WHEN stage = 1 THEN 1 ELSE 0 END) AS stage1_rows,
                    SUM(CASE WHEN stage = 2 THEN 1 ELSE 0 END) AS stage2_rows,
                    SUM(CASE WHEN agreement IS NULL THEN 1 ELSE 0 END) AS excluded
                FROM samples
                """).fetchone()
            invalid_stage = database.execute("""
                SELECT source_relative_path FROM samples
                WHERE (agreement IS NULL AND stage IS NOT NULL)
                   OR (agreement IS NOT NULL AND agreement < 0.95 AND stage IS NOT 1)
                   OR (agreement IS NOT NULL AND agreement >= 0.95 AND stage IS NOT 2)
                LIMIT 1
                """).fetchone()
            if invalid_stage is not None:
                raise ValueError(f"prepared index stage rule mismatch: {invalid_stage[0]}")
            ordinal_count = database.execute("SELECT COUNT(*) FROM stage_ordinals").fetchone()[0]
            ordinal_mismatch = database.execute("""
                SELECT ordinal.stage, ordinal.ordinal
                FROM stage_ordinals AS ordinal
                JOIN samples AS sample USING (sample_id)
                WHERE ordinal.stage != sample.stage
                LIMIT 1
                """).fetchone()
            ordinal_ranges = {
                row["stage"]: (row["rows"], row["minimum"], row["maximum"]) for row in database.execute("""
                    SELECT stage, COUNT(*) AS rows, MIN(ordinal) AS minimum, MAX(ordinal) AS maximum
                    FROM stage_ordinals GROUP BY stage
                    """)
            }
    except sqlite3.Error as error:
        raise ValueError(f"cannot independently audit prepared index: {index_path}") from error

    total_rows = int(counts["total_rows"])
    stage1_rows = int(counts["stage1_rows"])
    stage2_rows = int(counts["stage2_rows"])
    excluded = int(counts["excluded"])
    expected_counts = {
        "total row count": (total_rows, config.data.expected_rows),
        "stage-1 row count": (stage1_rows, config.data.expected_stage1_rows),
        "stage-2 row count": (stage2_rows, config.data.expected_stage2_rows),
        "null-agreement row count": (excluded, config.data.expected_null_agreement),
    }
    for label, (actual, expected) in expected_counts.items():
        if actual != expected:
            raise ValueError(f"prepared index {label} mismatch: expected {expected}, found {actual}")
    eligible_rows = stage1_rows + stage2_rows
    if total_rows != eligible_rows + excluded:
        raise ValueError("prepared index eligible/null counts do not partition all rows")
    if ordinal_count != eligible_rows or ordinal_mismatch is not None:
        raise ValueError("prepared index stage ordinals do not bind every eligible row exactly once")
    for stage, expected_rows in ((1, stage1_rows), (2, stage2_rows)):
        expected_range = (expected_rows, 0, expected_rows - 1) if expected_rows else None
        if ordinal_ranges.get(stage) != expected_range:
            raise ValueError(f"prepared index stage-{stage} ordinals are not contiguous")

    expected_provenance = {
        "schema_version": 1,
        "corpus_root": str(corpus_root),
        "rover_archive": provenance.get("rover_archive"),
        "combined_sidecar": provenance.get("combined_sidecar"),
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
    if dict(provenance) != expected_provenance:
        raise ValueError("prepared index provenance does not match current immutable expectations")
    rover_path = _corpus_provenance_path(provenance.get("rover_archive"), corpus_root, "ROVER archive")
    combined_path = _corpus_provenance_path(provenance.get("combined_sidecar"), corpus_root, "combined sidecar")
    if sha256_file(rover_path) != expectations.rover_archive_sha256:
        raise ValueError("ROVER archive SHA-256 mismatch")
    if sha256_file(combined_path) != expectations.combined_sidecar_sha256:
        raise ValueError("combined sidecar SHA-256 mismatch")
    if metadata.get("sidecar_path") != str(combined_path):
        raise ValueError("prepared index sidecar path does not match canonical provenance")

    index_sha256 = sha256_file(index_path)
    recomputed_audit = {
        "index_path": str(index_path),
        "fingerprint": recomputed_fingerprint,
        "index_sha256": index_sha256,
        "total_rows": total_rows,
        "eligible_rows": eligible_rows,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "excluded_null_agreement": excluded,
    }
    for name, expected in recomputed_audit.items():
        if audit.get(name) != expected:
            raise ValueError(f"index audit {name} mismatch: expected {expected}, found {audit.get(name)}")
    return {**recomputed_audit, "audit_identity": fingerprint(recomputed_audit)}


def _corpus_provenance_path(value: object, corpus_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"prepared index has no {label} path")
    path = Path(value).resolve()
    if not path.is_relative_to(corpus_root) or not path.is_file():
        raise ValueError(f"prepared index {label} path is outside the immutable corpus")
    return path


def _deep_audit_selection(config: BalalaikaConfig, index_path: Path, index_fingerprint: str) -> dict[str, Any]:
    root = config.selection_dir.resolve()
    raw_manifests = {
        "memorization": _read_json(root / "memorization.json", "memorization selection"),
        "prompts": _read_json(root / "prompts.json", "prompt selection"),
        "assignments": _read_json(root / "benchmark-prompts.json", "benchmark assignments"),
        "audio_ids": _read_json(root / "audio-log-ids.json", "audio-log selection"),
    }
    common = {
        (value.get("schema_version"), value.get("fingerprint"), value.get("seed")) for value in raw_manifests.values()
    }
    if len(common) != 1:
        raise ValueError("selection manifests do not share one schema, fingerprint, and seed")
    schema_version, declared_fingerprint, seed = next(iter(common))
    if schema_version != 1 or seed != config.runtime.seed or not isinstance(declared_fingerprint, str):
        raise ValueError("selection manifests do not match the configured schema and seed")
    bundle = _load_selection(root)
    if bundle.seed != config.runtime.seed or bundle.fingerprint != declared_fingerprint:
        raise ValueError("selection bundle identity does not match its manifests")

    benchmark_rows = _load_benchmark_rows(config.hub.local_dir / "benchmark" / config.hub.benchmark_file)
    benchmark_ids = {row.id for row in benchmark_rows}
    prompt_ids = {item.prompt_id for item in bundle.prompts}
    expected_prompt_ids = {f"prompt-{number:02d}" for number in range(20)}
    if len(bundle.memorization) != 4 or len({item.source_relative_path for item in bundle.memorization}) != 4:
        raise ValueError("memorization selection must contain four unique source identities")
    if len(bundle.prompts) != 20 or len({item.source_relative_path for item in bundle.prompts}) != 20:
        raise ValueError("prompt selection must contain 20 unique source identities")
    if prompt_ids != expected_prompt_ids:
        raise ValueError("prompt selection IDs are not the exact canonical prompt set")
    if set(bundle.benchmark_prompt_by_id) != benchmark_ids:
        raise ValueError("benchmark assignment IDs do not exactly match the pinned benchmark")
    if not set(bundle.benchmark_prompt_by_id.values()).issubset(prompt_ids):
        raise ValueError("benchmark assignment has an unknown prompt reference")
    if (
        len(bundle.audio_log_ids) != 4
        or len(set(bundle.audio_log_ids)) != 4
        or not set(bundle.audio_log_ids).issubset(benchmark_ids)
    ):
        raise ValueError("audio-log IDs must be four unique pinned benchmark IDs")

    selected = [*bundle.memorization, *bundle.prompts]
    selected_rows = _selected_index_rows(index_path, [item.source_relative_path for item in selected])
    combined_path = _selected_sidecar_path(index_path)
    for item in bundle.memorization:
        row = selected_rows.get(item.source_relative_path)
        if (
            row is None
            or row["stage"] != 2
            or row["agreement"] is None
            or row["agreement"] < 0.95
            or item.stage != 2
            or item.agreement < 0.95
        ):
            raise ValueError("memorization selection contains a row outside stage-2 high-agreement eligibility")
    for item in bundle.prompts:
        row = selected_rows.get(item.source_relative_path)
        if row is None or row["stage"] not in (1, 2) or row["agreement"] is None:
            raise ValueError("prompt selection contains an ineligible row")

    recomputed_fingerprint = fingerprint(
        {
            "schema_version": 1,
            "seed": bundle.seed,
            "index_fingerprint": index_fingerprint,
            "memorization": [_selection_sample_fingerprint(item) for item in bundle.memorization],
            "prompts": [_selection_sample_fingerprint(item) for item in bundle.prompts],
            "benchmark_prompt_by_id": bundle.benchmark_prompt_by_id,
            "audio_log_ids": bundle.audio_log_ids,
        }
    )
    if recomputed_fingerprint != declared_fingerprint:
        raise ValueError("selection fingerprint does not match actual manifest content and current inputs")
    for number, item in enumerate(bundle.memorization):
        _verify_selected_sample(
            item,
            selected_rows[item.source_relative_path],
            combined_path,
            root / "audio" / "memorization" / f"item-{number:02d}.wav",
        )
    for number, item in enumerate(bundle.prompts):
        _verify_selected_sample(
            item,
            selected_rows[item.source_relative_path],
            combined_path,
            root / "audio" / "prompts" / f"prompt-{number:02d}.wav",
        )
    return {
        "selection_fingerprint": recomputed_fingerprint,
        "memorization_samples": len(bundle.memorization),
        "validation_prompts": len(bundle.prompts),
        "benchmark_assignments": len(bundle.benchmark_prompt_by_id),
        "audio_log_ids": len(bundle.audio_log_ids),
    }


def _selected_index_rows(index_path: Path, identities: list[str]) -> dict[str, sqlite3.Row]:
    unique_identities = sorted(set(identities))
    placeholders = ",".join("?" for _ in unique_identities)
    try:
        with sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True) as database:
            database.row_factory = sqlite3.Row
            return {
                row["source_relative_path"]: row
                for row in database.execute(
                    f"""
                    SELECT source_relative_path, agreement, stage, sidecar_offset, sidecar_size
                    FROM samples WHERE source_relative_path IN ({placeholders})
                    """,
                    unique_identities,
                )
            }
    except sqlite3.Error as error:
        raise ValueError("cannot validate selection identities against the prepared index") from error


def _selected_sidecar_path(index_path: Path) -> Path:
    try:
        with sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True) as database:
            row = database.execute("SELECT value FROM metadata WHERE key = 'sidecar_path'").fetchone()
    except sqlite3.Error as error:
        raise ValueError("cannot read selected sidecar identity from the prepared index") from error
    if row is None or not isinstance(row[0], str) or not Path(row[0]).is_file():
        raise ValueError("prepared index selected sidecar is unavailable")
    return Path(row[0])


def _verify_selected_sample(item: SelectedSample, row: sqlite3.Row, sidecar_path: Path, expected_wav: Path) -> None:
    if item.stage != row["stage"] or item.agreement != row["agreement"]:
        raise ValueError("selection stage/agreement does not match the prepared index")
    if item.wav_path.resolve() != expected_wav.resolve() or not expected_wav.is_file():
        raise ValueError("selection WAV path does not match the canonical bundle layout")
    if sha256_file(expected_wav) != item.wav_sha256:
        raise ValueError("selection WAV SHA-256 does not match the extracted artifact")
    with sidecar_path.open("rb") as source:
        source.seek(row["sidecar_offset"])
        payload = source.read(row["sidecar_size"])
    try:
        sidecar = _mapping(json.loads(payload), "selected sidecar row")
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("selected sidecar row is malformed") from error
    if sidecar.get("source_relative_path") != item.source_relative_path:
        raise ValueError("selected sidecar row identity does not match the manifest")
    if sidecar.get("rover_punctuated_accented") != item.text:
        raise ValueError("selected manifest text does not match the immutable sidecar")


def _selection_sample_fingerprint(item: SelectedSample) -> dict[str, Any]:
    value = item.model_dump(mode="json")
    value.pop("wav_path")
    return value


def _index_fingerprint(config: BalalaikaConfig) -> str:
    path = config.data.index_dir / "balalaika-index.sqlite3"
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as database:
            row = database.execute("SELECT value FROM metadata WHERE key = 'fingerprint'").fetchone()
    except sqlite3.Error as error:
        raise ValueError(f"cannot read prepared index identity: {path}") from error
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise ValueError("prepared index has no fingerprint")
    return row[0]


def _load_selection(root: Path) -> SelectionBundle:
    memorization = _read_json(root / "memorization.json", "memorization selection")
    prompts = _read_json(root / "prompts.json", "prompt selection")
    assignments = _read_json(root / "benchmark-prompts.json", "benchmark assignments")
    audio_ids = _read_json(root / "audio-log-ids.json", "audio-log selection")
    fingerprints = {value.get("fingerprint") for value in (memorization, prompts, assignments, audio_ids)}
    seeds = {value.get("seed") for value in (memorization, prompts, assignments, audio_ids)}
    if len(fingerprints) != 1 or None in fingerprints or len(seeds) != 1:
        raise ValueError("selection manifests do not share one fingerprint and seed")
    try:
        return SelectionBundle(
            fingerprint=next(iter(fingerprints)),
            seed=next(iter(seeds)),
            memorization=[SelectedSample.model_validate(item) for item in memorization["memorization"]],
            prompts=[PromptSample.model_validate(item) for item in prompts["prompts"]],
            benchmark_prompt_by_id=assignments["benchmark_prompt_by_id"],
            audio_log_ids=audio_ids["audio_log_ids"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("selection manifests are malformed") from error


def _load_benchmark_rows(path: Path) -> tuple[BenchmarkRow, ...]:
    rows: list[BenchmarkRow] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    value = json.loads(line)
                    rows.append(BenchmarkRow(**_mapping(value, f"benchmark row {line_number}")))
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(f"benchmark row {line_number} is malformed") from error
    except OSError as error:
        raise ValueError(f"cannot read pinned benchmark {path}: {error}") from error
    if len(rows) != 2_000 or len({row.id for row in rows}) != 2_000:
        raise ValueError(f"benchmark must contain 2,000 unique rows; found {len(rows)}")
    return tuple(rows)


def _approval_expectations(
    config: BalalaikaConfig,
    *,
    pins: Mapping[str, Mapping[str, Any]],
    data_fingerprint: str,
    selection_fingerprint: str,
    lora_fingerprint: str,
) -> dict[str, str]:
    result_path = config.output_dir / "memorization" / "memorization-result.json"
    result = _read_json(result_path, "memorization result")
    stored = _mapping(result.get("fingerprints"), "memorization result fingerprints")
    current = {
        "base_revision": pins["model"]["revision"],
        "data_fingerprint": data_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "lora_fingerprint": lora_fingerprint,
    }
    for name, value in current.items():
        if stored.get(name) != value:
            raise ValueError(f"memorization approval {name} does not match current inputs")
    expected = {
        **current,
        "checkpoint": stored.get("checkpoint"),
        "wandb_run_id": stored.get("wandb_run_id"),
        "wandb_completion": stored.get("wandb_completion"),
        "result": sha256_file(result_path),
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise ValueError("memorization result has incomplete approval fingerprints")
    return expected


def _load_selection_summary(root: Path) -> dict[str, Any]:
    memorization = _read_json(root / "memorization.json", "memorization selection")
    prompts = _read_json(root / "prompts.json", "prompt selection")
    assignments = _read_json(root / "benchmark-prompts.json", "benchmark assignments")
    audio_ids = _read_json(root / "audio-log-ids.json", "audio-log selection")
    fingerprints = {value.get("fingerprint") for value in (memorization, prompts, assignments, audio_ids)}
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError("selection manifests do not share one fingerprint")
    memorization_rows = memorization.get("memorization")
    prompt_rows = prompts.get("prompts")
    assignment_rows = assignments.get("benchmark_prompt_by_id")
    fixed_audio = audio_ids.get("audio_log_ids")
    counts = (
        len(memorization_rows) if isinstance(memorization_rows, list) else -1,
        len(prompt_rows) if isinstance(prompt_rows, list) else -1,
        len(assignment_rows) if isinstance(assignment_rows, dict) else -1,
        len(fixed_audio) if isinstance(fixed_audio, list) else -1,
    )
    if counts != (4, 20, 2_000, 4):
        raise ValueError(f"selection counts are invalid: {counts}")
    return {
        "selection_fingerprint": next(iter(fingerprints)),
        "memorization_samples": counts[0],
        "validation_prompts": counts[1],
        "benchmark_assignments": counts[2],
        "audio_log_ids": counts[3],
    }


def _read_json(path: Path, label: str) -> _JSON:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is malformed: {path}") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> _JSON:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must have a lowercase SHA-256 digest")
