# Balalaika final fix wave report

Date: 2026-08-03

Base: `1c1cbac`

Scope: one Critical and seven Important findings from `final-review-report.md`

## Outcome

All eight findings are implemented with focused regression coverage. The production audit remained read-only, all protected corpus artifacts retained their exact baseline sizes and mtimes, and no prohibited production workflow was launched.

## Finding-to-fix trace

1. **Critical destructive roots** — `_PreparationTransaction` resolves and validates corpus/index/selection/output roots before any write; rejects equal, ancestor, descendant, symlinked, and mutually nested generation roots; and requires a schema/owner/role/canonical-root/status marker before an existing directory may be replaced or deleted. Tests cover each required path relation and refusal of an unmarked operator directory.
2. **Preparation availability and crash recovery** — builds occur in marked same-parent sibling generations while the old index and selection remain visible. A durable `.prepare-transaction.json` is written before staging begins. Publication is journal-backed; startup removes an interrupted build or completes a mixed interrupted publication. Consumers refuse an active journal. Tests cover live-old-generation visibility, interrupted build cleanup, interrupted two-root publication, and journal-before-stage ordering.
3. **Pre-commit independent validation** — preparation runs the deep index and selection audits against staged physical roots and final declared roots before publication. Exact total/stage-1/stage-2/null counts, SQLite integrity, stage rules, ordinal coverage/contiguity, provenance, source hashes, canonical deterministic selection, selected content, and WAV hashes are checked. A wrong configured stage count fails before commit.
4. **Shared prepared-generation verifier** — `memorize`, `train`, and `validate` run the same deep verifier before constructing runtime/model state. SQLite SHA-256 is compared with the audit in addition to the internal fingerprint, and the selection is independently regenerated/verified. SQLite-byte and self-consistent manifest-text tamper regressions fail before runtime setup.
5. **Production microbatch probe** — the workflow always supplies the trainer a production selector. Fixed microbatch uses the explicit bypass. Otherwise, the selector queries the longest eligible stage ordinals, materializes them through the production dataset and `VoxCPMCollator`, and runs the real processor/forward/weighted-loss/backward/optimizer probe collectively before `runtime.prepare`. Tests cover wiring, lifecycle ordering, representative-row selection, real step execution, and fixed bypass.
6. **Generation settings** — evaluation and memorization pass configured `cfg_value`, `inference_timesteps`, and `max_len` to VoxCPM2. Validation requires settings whose fingerprint matches its ledger and includes them in input/item identity. Memorization result and manual approval fingerprints bind the identical settings. Tests assert exact keyword names/values and the extended evidence.
7. **Collective-safe evaluation status** — local execution and status-write outcomes are retained in memory and all ranks enter outcome collectives from `finally` before errors or diagnostics are published. A rank-local `atomic_json` status failure regression proves the collective is entered instead of stranding peers.
8. **Final-boundary signal transition** — after optimizer completion and boundary 8, a stop signal returns the final durable boundary checkpoint; it does not create a recovery checkpoint that stage 2 would reject. The regression verifies stage-2 transition acceptance.

## TDD evidence

- Initial focused RED: 17 failures covering destructive paths, interrupted publication, exact counts, prepared-artifact tampering, production microbatch wiring/probe order, generation arguments/identity, status collective behavior, and final-boundary signal routing.
- Intermediate GREEN: 13 passed / 4 failed, after which the remaining test seams were corrected.
- Affected GREEN: `148 passed, 21 warnings in 11.33s`.
- Final journal-order RED: `1 failed` with `stage_exists` observed as `True` before durable journaling.
- Final journal-order/recovery GREEN: `3 passed, 2 warnings in 0.43s`.

## Verification evidence

- Full suite: `rtk proxy uv run pytest tests -q` → `370 passed, 17 warnings in 89.95s` (final post-change run).
- Static checks: Black check → 10 files unchanged; Flake8 `--max-line-length=120` → exit 0; compileall → exit 0; `git diff --check` → exit 0.
- Note: an earlier repo-root `pytest -q` exited during collection because standalone `scripts/test_pick_runtime_dtype.py` calls `sys.exit(0)`. The repository's established test boundary is `tests/`, used for the successful full run.

### Read-only production audit

Command: `rtk proxy uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml audit`

- Status: complete, exit 0
- Source shards: 519
- Joined / eligible / excluded: 4,075,032 / 4,074,723 / 309
- Stage 1 / stage 2: 2,486,821 / 1,587,902
- Memorization / prompts / assignments / audio IDs: 4 / 20 / 2,000 / 4
- Index bytes: 1,159,962,624
- Index fingerprint: `b276bdaf32e1ef2ed0275b919625215959e91494ad354caca8f272273f210cf2`
- Selection fingerprint: `d7ab905d69f33a64c4c747dbf665eef6ed5c313cad6ac12534d9e3036991b629`
- Audit identity: `7cc1d8d787045937796895e4e824e32fc24e98a40c0e311b3bcf01d6d4d23abf`

Protected files before and after audit:

| Artifact | Size | mtime (UTC) |
| --- | ---: | --- |
| `verification.json` | 80,269 | `2026-07-29 22:14:42.000000000 +0000` |
| `rover-punctuation-stress.jsonl` | 1,366,964,556 | `2026-07-31 22:56:02.000000000 +0000` |
| `balalaika-rover-results-20260729T135419Z.tar.zst` | 375,137,790 | `2026-07-29 21:58:57.000000000 +0000` |

### Synthetic GPU smokes

- 1 rank, train/checkpoint: PASS; restored checkpoint, optimizer step 1, probe microbatch 2, LoRA gradients present.
- 8 ranks, train/checkpoint: PASS; all ranks restored, identical synchronized LoRA gradients, exact disjoint coverage of samples 0–31, next sample position 32.
- 8 ranks, validation, 32 items: PASS; exact disjoint coverage, ASR released on every rank, rank-0-only tracking, exactly one aggregate and completion artifact.

## Residual concerns

- POSIX provides no single atomic rename for two independent directories. The pair is logically atomic: the durable journal blocks every guarded consumer during sequential renames, and startup deterministically completes publication before new preparation. Uncoordinated external readers that ignore the journal could observe a mixed pair.
- Current production generations predate the new ownership marker. Read-only audit succeeds, but a future `prepare` intentionally refuses to replace an unmarked legacy directory until an operator makes an explicit migration/ownership decision.
- The shared verifier rehashes all immutable source shards before every protected production entrypoint. This is intentionally high assurance and can be slow on shared storage.
- Existing non-blocking minor triage from the final review remains deferred: stricter curriculum operator/threshold validation and bounded duration/agreement/ASR/per-shard audit summaries.

## Prohibited-action attestation

This fix wave did **not** run real memorization, create approval, start stage 1 or stage 2 training, publish W&B/model artifacts, run real pin/prepare, expose credentials, or mutate the proprietary corpus/current production generation. Only read-only production audit and synthetic bounded GPU smokes were executed.

## Final residual correction — 2026-08-03

Base: `fa8210a534ce65dcedefb13c503d22ff7fa6bd52`

The scoped re-review's three verified residuals were corrected without addressing optional or minor scope:

1. **NULL ordinal substitution** — the independent ordinal join now rejects `samples.stage IS NULL` explicitly in addition to unequal eligible stages. A self-consistent regression replaces a stage-1 ordinal's sample with the excluded row while preserving ordinal count/range and the refreshed audit hash.
2. **Resume-status publication** — completed-validation resume retains the rank-zero write error, gathers write outcomes on every rank, and raises collectively before any rank enters the barrier or reads the missing status. The regression executes rank zero and a worker against a shared simulated `[failure, success]` outcome and proves one gather plus zero barriers on both.
3. **Probe materialization** — representative-ordinal lookup, dataset indexing/decode, and collation now occur inside the protected per-candidate `sample_factory`. The regression injects a local decode failure and proves it becomes a gathered terminal `ProbeError` rather than escaping before the collective.

### Residual TDD evidence

- RED command: focused three-test run → `3 failed` for the exact defects: ordinal substitution did not raise, resume surfaced raw `OSError`, and decode surfaced raw `ValueError` before a gather.
- GREEN command: identical focused run → `3 passed, 5 warnings in 4.96s`.
- Final post-format full suite: `rtk proxy uv run pytest tests -q` → `373 passed, 18 warnings in 103.84s`.
- Changed-file Black check: 4 files unchanged; Flake8 `--max-line-length=120`, compileall, and `git diff --check`: exit 0.

### Residual production and GPU evidence

The required read-only audit exited 0 and reproduced:

- 519 source shards; 4,075,032 joined; 4,074,723 eligible; 309 excluded
- Stage 1 = 2,486,821; stage 2 = 1,587,902
- Index bytes = 1,159,962,624
- Index fingerprint = `b276bdaf32e1ef2ed0275b919625215959e91494ad354caca8f272273f210cf2`
- Selection fingerprint = `d7ab905d69f33a64c4c747dbf665eef6ed5c313cad6ac12534d9e3036991b629`
- Audit identity = `7cc1d8d787045937796895e4e824e32fc24e98a40c0e311b3bcf01d6d4d23abf`

Fresh bounded smokes all exited 0: one-rank train/checkpoint, eight-rank train/checkpoint with synchronized LoRA gradients and exact 0–31 sample coverage, and eight-rank/32-item validation with ASR release on every rank plus one aggregate/completion/tracking publication.

Protected files retained identical before/after sizes and nanosecond mtimes: `verification.json` = 80,269 bytes at `2026-07-29 22:14:42.000000000 +0000`; combined sidecar = 1,366,964,556 bytes at `2026-07-31 22:56:02.000000000 +0000`; ROVER archive = 375,137,790 bytes at `2026-07-29 21:58:57.000000000 +0000`.

No real memorization, approval, preparation, stage training, W&B/model publication, credential access, or production/corpus mutation occurred. Previously documented non-blocking concerns remain unchanged; this correction introduces no new known load-bearing residual.
