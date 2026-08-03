# Balalaika Task 14 Fix Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five Task 14 review findings without allowing any real smoke training, accepting self-declared prepared artifacts, exposing mixed preparation generations, or leaking Accelerate resources.

**Architecture:** Keep the CLI thin, but reject memorization smoke before invoking the command service. Make `ProductionCommands` verify immutable Hub inputs before preparation, coordinate index rollback with the selection publisher, and delegate audit to independent SQLite/source/selection verification. Runtime owners close Accelerate exactly once in outer `finally` blocks after tracking cleanup.

**Tech Stack:** Python 3.14, argparse, Pydantic, SQLite, pytest, Accelerate, atomic filesystem rename/publication.

## Global Constraints

- Strict RED/GREEN TDD for each finding.
- Never run real memorization, approval, stage training, or publication.
- Do not rerun production preparation; use only focused synthetic transaction tests.
- Run the real deep audit against the existing production artifacts after implementation.
- Do not implement the ledgered checkpoint-argument minor in this round.
- Prefix every executable runbook command with `rtk` and use `rtk uv run` for project tools.

---

### Task 1: Reject memorization smoke before side effects

**Files:**
- Modify: `src/voxcpm/training/balalaika/cli.py`
- Test: `tests/training/balalaika/test_cli.py`

**Interfaces:**
- Consumes: parsed `memorize --smoke`.
- Produces: `CliSafetyError` directing the operator to `scripts/smoke_balalaika_accelerate.py` before command-service dispatch.

- [ ] Add a CLI regression whose command double raises if invoked and whose filesystem sentinel remains absent.
- [ ] Run the exact test and observe it fail because `commands.memorize(..., smoke=True)` is called.
- [ ] Reject the route in `_dispatch` before the world-size/service boundary.
- [ ] Run the exact test and CLI routing tests to GREEN.

### Task 2: Independently deep-audit prepared artifacts

**Files:**
- Modify: `src/voxcpm/training/balalaika/config.py`
- Modify: `src/voxcpm/training/balalaika/workflow.py`
- Modify: `conf/voxcpm_v2/balalaika_lora.yaml`
- Test: `tests/training/balalaika/test_cli.py`

**Interfaces:**
- Consumes: current index/audit, immutable corpus manifests and files, Hub pins, benchmark, four selection manifests, and 24 WAVs.
- Produces: recomputed counts, identities, hashes, and fingerprints; raises on any mismatch.

- [ ] Add production-shaped SQLite/selection fixture coverage for integrity/count/stage/ordinal validation, selection fingerprint tampering with preserved declarations, eligibility/reference tampering, and WAV tampering.
- [ ] Run the tests and observe acceptance/failure at the old shallow audit boundary.
- [ ] Add configured exact stage counts and implement read-only SQLite/source/provenance/audit/selection recomputation.
- [ ] Run the audit regressions and relevant index/selection/hub tests to GREEN.

### Task 3: Verify pins and rollback index publication when selection fails

**Files:**
- Modify: `src/voxcpm/training/balalaika/workflow.py`
- Test: `tests/training/balalaika/test_cli.py`

**Interfaces:**
- Consumes: verified `hub-pins.json`, current index directory, atomic selection publisher.
- Produces: a complete matching index+selection generation, or the previous complete generation restored by same-filesystem directory renames without copying the SQLite file.

- [ ] Add an ordering test proving pin verification precedes expectation/index/benchmark consumption.
- [ ] Add failure injection proving a previous index directory remains byte-identical after selection failure and no partial new index survives.
- [ ] Run both tests and observe the old mixed-generation behavior.
- [ ] Implement a preparation lock/rollback transaction around index build and selection publication.
- [ ] Run transaction and preparation tests to GREEN.

### Task 4: Close real runtime owners exactly once

**Files:**
- Modify: `src/voxcpm/training/balalaika/workflow.py`
- Test: `tests/training/balalaika/test_cli.py`

**Interfaces:**
- Consumes: runtime returned by `runtime_factory`/`AccelerateRuntime.create`.
- Produces: one `runtime.close()` per created runtime on success and failure; training preserves `run_manager.finish()` before runtime teardown.

- [ ] Add memorization success/failure lifecycle regressions with a fake runtime and runner.
- [ ] Add configured-training lifecycle regressions using monkeypatched runtime/tracking/trainer seams, including setup and run failures.
- [ ] Run and observe missing close calls.
- [ ] Add outer `try/finally` ownership blocks that preserve the original exception and cleanup ordering.
- [ ] Run lifecycle tests and affected runtime/trainer tests to GREEN.

### Task 5: Correct executable runbook environments

**Files:**
- Modify: `docs/balalaika_training.md`

**Interfaces:**
- Produces: executable examples using `rtk uv run hf ...` and `rtk uv run wandb ...` consistently with the verified project launcher.

- [ ] Replace the three unprefixed authentication commands and manually inspect every fenced executable command.
- [ ] Exercise the relevant installed `--help` routes to verify the documented executables resolve in the project environment.

### Task 6: Evidence, verification, and local commit

**Files:**
- Modify: `implementation-notes.md`
- Modify: `.superpowers/sdd/2026-08-02-balalaika-two-stage-lora-training/task-14-report.md`
- Create: `.superpowers/sdd/2026-08-02-balalaika-two-stage-lora-training/task-14-fix-round-1-evidence.md`

**Interfaces:**
- Produces: exact RED/GREEN, production audit, GPU, corpus immutability, and self-review evidence with no credentials.

- [ ] Run focused CLI/workflow/index/selection/hub/runtime tests.
- [ ] Run full `tests/`, Black, Flake8, compileall, and `git diff --check`.
- [ ] Rerun affected one-rank/eight-rank synthetic smokes and the real read-only deep audit.
- [ ] Append raw commands, exit codes, concise outputs, artifact queries, and before/after corpus metadata.
- [ ] Review the diff, commit separately, do not push, and report only status/commit/test/audit/smoke summary/concerns.
