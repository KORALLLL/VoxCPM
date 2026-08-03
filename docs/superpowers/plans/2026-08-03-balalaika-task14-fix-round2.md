# Balalaika Task 14 Fix Round 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make production audit reject any self-consistent selection bundle that was not produced by the configured seed and canonical selection algorithm.

**Architecture:** Extract the seed-derived reservoir choices, prompt assignments, audio-log IDs, and fingerprint inputs into one pure canonical selection-plan interface. Publication materializes that plan; audit independently streams authoritative SQLite rows and pinned benchmark IDs into the same pure interface, then compares the exact ordered regenerated plan and independently derived selected content against the published bundle before retaining all existing WAV/hash checks.

**Tech Stack:** Python 3.14, SQLite read-only queries, `random.Random`, Pydantic, pytest, SHA-256 artifact fingerprints, Accelerate synthetic smokes, modern Hugging Face `hf` CLI.

## Global Constraints

- Strict RED/GREEN TDD: no production change before the self-consistent replacement test visibly fails by audit accepting the replacement.
- Preserve uniform reservoir sampling, within-set uniqueness, allowed memorization/prompt overlap, and the current independent memorization and validation RNG streams.
- Memorization candidates are exactly stage 2 with agreement at least 0.95; prompt candidates are all eligible stage-1/stage-2 rows.
- Benchmark IDs are exactly the 2,000 authoritative pinned IDs; prompt assignments and four audio-log IDs continue the validation RNG stream.
- Audit must not mutate production artifacts or trust published selection values as generator inputs.
- Preserve current canonical manifest layout, exact selected sidecar content checks, and all 24 WAV/hash checks.
- Do not run real memorization, approval, stage training, W&B training, or publication.
- Prefix every executable command with `rtk`; do not print authentication tokens.
- Preserve the two pushed recovery-note sections in `implementation-notes.md` verbatim and append Fix Round 2 evidence after them.

---

### Task 1: Reproduce self-consistent non-deterministic replacement acceptance

**Files:**
- Modify: `tests/training/balalaika/test_workflow.py`

**Interfaces:**
- Consumes: the production-shaped `prepared_deep_audit` index, selection manifests, selected WAV layout, benchmark rows, and configured seed.
- Produces: a regression that replaces the four memorization identities and twenty prompt identities with different eligible authoritative rows, rebuilds matching WAVs/content/fingerprint/assignments/audio IDs, and expects `ProductionCommands.audit` to reject the non-canonical generation.

- [ ] Add enough eligible fixture rows that an alternate disjoint-within-set selection exists while preserving allowed cross-set overlap.
- [ ] Build the alternate manifest entirely from authoritative eligible rows and literal deterministic alternate choices; do not call the production canonical selector to derive the expected rejection.
- [ ] Update all four manifests, selected WAVs, benchmark assignments, audio-log IDs, and the declared fingerprint so the replacement is internally consistent.
- [ ] Run the exact regression with `rtk proxy uv run pytest tests/training/balalaika/test_workflow.py::<test-name> -q` and record RED as an assertion failure because audit returns `status=complete`.

### Task 2: Share a pure canonical selection plan between publisher and verifier

**Files:**
- Modify: `src/voxcpm/training/balalaika/selection.py`
- Modify: `src/voxcpm/training/balalaika/workflow.py`
- Modify: `tests/training/balalaika/test_selection.py`
- Modify: `tests/training/balalaika/test_workflow.py`

**Interfaces:**
- Consumes: ordered authoritative `_IndexedSample` streams, exact benchmark IDs, configured integer seed, index fingerprint, and a resolver that derives immutable text/WAV identity after rows are chosen.
- Produces: a frozen canonical plan containing ordered memorization rows, ordered prompt rows with canonical `prompt-00` through `prompt-19`, exact benchmark-to-prompt assignment insertion order, four sorted audio-log IDs, and fingerprint material independent of published manifests.

- [ ] Add a selection-level test proving the pure plan produces the same ordered identities, assignments, IDs, and fingerprint material used by publication while permitting memorization/prompt overlap.
- [ ] Run it and observe RED because no pure interface exists.
- [ ] Extract only the deterministic choice/assignment logic from `create_selection_manifests`; keep reservoir implementation and RNG order unchanged.
- [ ] Make publication use the pure plan, materialize selected rows, compute hashes/content, and publish the same schema and fingerprints as before.
- [ ] Make `_deep_audit_selection` open the authoritative index read-only, regenerate the canonical plan from `_iter_samples`, load exact pinned benchmark IDs independently, resolve authoritative text/content, and compare exact ordered identities/content/assignments/audio IDs plus fingerprint against the published bundle.
- [ ] Run the exact workflow replacement regression and selection tests to GREEN.
- [ ] Mutate one canonical comparison mentally at a time: selected identity, order, prompt ID, assignment, audio-log ID, content, or fingerprint must make a regression fail.

### Task 3: Production, GPU, Hub, and repository qualification

**Files:**
- Modify: `implementation-notes.md`
- Modify: `.superpowers/sdd/2026-08-02-balalaika-two-stage-lora-training/task-14-report.md`
- Modify: `.superpowers/sdd/2026-08-02-balalaika-two-stage-lora-training/task-14-fix-round-1-evidence.md`

**Interfaces:**
- Produces: exact Fix Round 2 RED/GREEN, production deep-audit, three synthetic GPU-smoke, modern `hf` revision-info, full-suite/static-check, warning, and safety evidence.

- [ ] Run focused selection/workflow/CLI tests and record exact counts/warnings.
- [ ] Run the real read-only production deep audit against `/workspace/balalaika_lora_training` and record exact JSON output and exit code.
- [ ] Run the prescribed one-rank train/checkpoint, eight-rank train/checkpoint, and eight-rank 32-item validation synthetic smokes with project-environment Accelerate; record commands, exit codes, concise contract outputs, and teardown warnings.
- [ ] Run `rtk uv run hf models info` for VoxCPM2 and GigaAM and `rtk uv run hf datasets info` for the hard-number benchmark at the configured immutable revisions; record only public revision metadata and never tokens.
- [ ] Run the complete `tests/` suite, fix-round Black/Flake8, compileall, and `git diff --check`; classify every remaining warning and residual concern.
- [ ] Append evidence without changing either pushed recovery-note section, inspect the final diff, create one separate Fix Round 2 commit, do not push, and return the commit hash plus audit/test/smoke/concern summary.
