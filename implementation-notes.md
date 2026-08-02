## 2026-08-02 - Balalaika training foundation

- Decision: Use Accelerate as the primary distributed-training interface, with a native DDP fallback retained for environments where Accelerate is unavailable or unsuitable.
- Scope: This foundation validates the two-stage curriculum contract and keeps only the configuration fields needed by the initial pipeline boundary.
- Validation: `rtk uv sync --extra dev --extra balalaika`, the focused Balalaika tests, formatting checks, and the full `tests/` suite completed successfully.

## 2026-08-02 - GigaAM adapter and validation ledger

- Decision: Load only the named `gigaam-v3-rnnt` model from its already pinned local directory, and verify CUDA availability before opening a session; the loader is never given a remote repository identifier.
- Decision: Store each benchmark ID as an independently locked, atomically replaced JSON record so distributed ranks can safely own disjoint IDs and a crash cannot publish a partial record.
- Assumption: Generation and ASR fingerprints identify all deterministic inputs supplied by the later evaluator; WAV content hashes are recomputed before reuse.
- Validation: RED collection failed while the modules were absent; focused adapter/ledger tests are green (16 passed), `py_compile`, Black, and `git diff --check` pass; the full `tests/` suite passes (134 tests).

## 2026-08-02 - GigaAM and ledger Fix Round 1

- Decision: A claim is now an opaque token plus monotonic per-item epoch and rank. Every generation/ASR/failure mutation requires the live claim under that item's advisory lock; generation retains the claim through ASR, while terminal stage results release it.
- Decision: Completion queries require the evaluator's current deterministic inputs and seed. Bulk completion receives an ID-to-`(inputs, seed)` mapping or callback, so stored inputs alone can never establish reuse.
- Decision: After `onnx_asr.load_model`, inspect the local package's documented adapter-to-ASR ownership (`adapter.asr._encoder`, `_decoder`, `_joiner`) and require every discovered model session to report CUDA as primary on the requested device. CPU-only pre/post-processing is not inspected as a model session.
- Validation: Targeted RED tests covered missing mutation claims, token/rank/lease takeover, current-input completion, active CPU fallback, unlisted `.onnx`, directory fsync, and interrupted pending attempt caps. GREEN: 24 focused tests and 139 full-suite tests passed; Black, compile, and diff checks passed.

## 2026-08-02 - Rank-zero W&B validation tracking

- Decision: Worker ranks receive a `NullRunManager`, so only the main process dynamically imports, initializes, logs, or finishes W&B. Memorization, stage 1, and stage 2 persist independent UUIDs before `wandb.init(..., resume="allow")`, while sharing the configured experiment group.
- Decision: A validation handoff is immutable and rejects any boundary other than 2,000 unique scored IDs and four existing local WAVs. Each upload builds a fresh W&B table and retains the selection-owned audio ID order in captions.
- Decision: Write an atomic pending boundary manifest before `run.log`; fsync the W&B run directory and atomically mark it complete only after `run.log` returns. Completed matching manifests suppress duplicate uploads on resume; failed or pending boundaries remain incomplete and retryable.

## 2026-08-02 - W&B validation tracking Fix Round 1

- Decision: A completed or pending boundary is identified by a canonical, atomically written and read-back full snapshot rather than a short metadata manifest. The snapshot holds every logged table row, metrics/counts/categories, progress, fingerprints, timing/failure values, and ordered local audio paths/hashes/captions; the manifest stores its fingerprint.
- Decision: Validation recomputes aggregates from every `ItemScore`, and binds the Task-5 selection fingerprint plus ordered audio IDs to the payload. This prevents aggregate drift, wrong example ordering, changed WAVs, and score/benchmark normalized-gold divergence from reaching W&B.
- Decision: Existing W&B state can resume only with the persisted config fingerprint. Local recovery is durable, but W&B backend acknowledgement is not made transactional; a crash after `run.log` remains safely retryable from the verified snapshot.

## 2026-08-02 - Exact fractional scheduling and LoRA-only checkpoints

- Decision: Epoch geometry floors the row count to complete global microbatches, then floors again to complete accumulation groups. The combined row loss is recorded as dropped samples, and geometries with fewer than eight optimizer steps are rejected because they cannot represent eight unique boundaries.
- Decision: Validation boundaries are stage-relative completed optimizer steps computed as integer ceilings for eighths. `TrainingProgress` persists the emitted-boundary cursor separately from optimizer progress, preventing step-zero, resume, and epoch-end duplicate emissions.
- Decision: Same-stage saves use temporary sibling directories and Accelerate save/load pre-hooks. The save hook writes only `lora_` tensors and clears Accelerate's model-weight list; optimizer, scheduler, RNG, and registered progress remain owned by `save_state`/`load_state`. Every file is hashed before atomic publication and the latest pointer carries the immutable metadata fingerprint.
- Decision: Stage transition is a separate adapter-only path. It accepts only the final stage-1 boundary, validates target-compatible identities, never receives an Accelerator, and optionally resets stage-relative progress while preserving the cross-stage global step.
- Validation: Focused RED failed at collection while both modules were absent. Focused GREEN covers 27 schedule/checkpoint cases; the full `tests/` suite passes 194 tests. Black check, `py_compile`, and `git diff --check` also complete cleanly.

## 2026-08-02 - LoRA checkpoint hardening Fix Round 1

- Changed: Checkpoint schema 2 binds exact Accelerate optimizer, scheduler, registered-progress, and per-rank RNG filenames/counts. RNG file hashes are embedded in checkpoint metadata and also covered by the complete file manifest.
- Decision: Only Accelerate `NO`, `MULTI_CPU`, and `MULTI_GPU` backends are supported. DDP-prefixed hook state is normalized against the unwrapped model's complete LoRA key set; FSDP, DeepSpeed, incomplete adapters, and partial expected identities are rejected before state mutation.
- Decision: Boundary metadata now includes optimizer-step epoch geometry, state-object counts, and stage-start global progress. Checkpoints must occur on an exact eighth boundary with aligned microsteps, epoch/sampler state, and stage/global counters. Stage 2 imports the verified stage-1 global step before resetting all stage-relative counters.
- Decision: Same-stage resume safely preflights registered progress against metadata. Any later Accelerate load failure poisons the manager and requires a process restart rather than permitting use of potentially partial model/optimizer/RNG state.
- Validation: Four focused RED groups reproduced DDP/backend/adapter, transition/identity, exact-state/progress, and late-restore failures (5, 3, 7, and 2 failures respectively). GREEN passes 48 focused tests and 215 full-suite tests. A real single-rank CPU Gloo `DistributedDataParallel` roundtrip confirms `module.*` normalization; Black, `py_compile`, and diff checks pass.
