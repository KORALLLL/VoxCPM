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

## 2026-08-02 - Accelerate runtime and synchronized microbatch probe

- Decision: Accelerate is the only implementation under the initial spike. Native DDP will be added only if the same Accelerate model-behavior, synchronization, or restore failure remains after at most two minimal runtime/probe fixes; transient shared-GPU resource contention is not a fallback trigger.
- Scope: The real one- and eight-process checks use only a tiny synthetic LoRA module and synthetic sample IDs. They do not load VoxCPM, memorization data, or a production training path.
- Changed: The first eight-process restore exposed that Accelerate 1.14.0 writes shared model/optimizer state on the main process but `save_state` does not end with a barrier. Non-main ranks returned early and ranks 2, 4, and 7 observed a temporarily absent `optimizer.bin`. The first and only runtime/probe fix adds `wait_for_everyone()` after the public `save_state`; its targeted RED observed `[save]`, GREEN observes `[save, barrier]`, and the identical eight-process retry passed.
- Attempts: Two initial `/usr/local/bin/accelerate` launches failed before runtime construction because the global Python first lacked the editable `voxcpm` package and then `argbind`. A temporary diagnostic import bootstrap was removed. All authoritative commands use the project environment: `rtk .venv/bin/accelerate launch --num_processes 1 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint` and the same command with `--num_processes 8`. The eight-process launcher found port 29500 occupied and safely selected an open standalone port; no external process was altered.
- Validation: RED initially failed collection because `runtime.py`/`probe.py` were absent. Focused GREEN passes 10 tests. The pre-change Balalaika baseline passes 152 tests and the final full repository test directory passes 218 tests. Bare `pytest -q` is not a valid repository command because `scripts/test_pick_runtime_dtype.py` exits during collection; `pytest tests/ -q` is the established full suite. Black, `py_compile`, and diff checks pass.
- Validation: Canonical one GPU passed with IDs `[0,1,2,3]`, accumulation `[false,true]`, optimizer step 1, restored next position 4, and 16.267 MiB peak. Canonical eight GPU passed with disjoint partitions `[[0,1,16,17],[2,3,18,19],[4,5,20,21],[6,7,22,23],[8,9,24,25],[10,11,26,27],[12,13,28,29],[14,15,30,31]]`; every rank reported accumulation `[false,true]`, LoRA gradient `[-0.0039180964,-0.0026081726]`, optimizer step 1, restored next position 32, and 16.269 MiB peak. Probe microbatch 2, reference update, and checkpoint restore passed on every rank. Logs and checkpoints are in `.superpowers/sdd/2026-08-02-balalaika-two-stage-lora-training/task-10-artifacts/`.
- Environment: Python 3.14.0, Accelerate 1.14.0, PyTorch 2.10.0+cu128, CUDA runtime 12.8, cuDNN 9.10.2, NCCL 2.27.5, NVIDIA driver 570.153.02, eight NVIDIA GeForce RTX 5090 GPUs with 32607 MiB each. Before the spike, shared usage ranged from 15119 to 27231 MiB; the synthetic spike allocated only about 16.3 MiB per process.
- Decision: Accelerate primary path passed; native DDP not implemented.

## 2026-08-02 - Accelerate probe Fix Round 1

- Changed: The original probe step received the runtime and prepared model, so a rank-local OOM could branch to the status gather while peer ranks entered DDP backward synchronization. It also used `runtime.accumulate`, advancing Accelerate's accumulation cursor during a supposedly reversible probe.
- Decision: The second and final permitted Accelerate fix makes probing a strictly local pre-`prepare` operation. The step receives only the verified unwrapped model and sample, uses ordinary local autograd/optimizer operations, and performs exactly one cross-rank status collective per candidate. Passing a wrapped model is rejected before work begins.
- Validation: Unit RED produced six expected failures because the old probe required `step_fn.model` and passed the runtime into the candidate. Focused GREEN passes 12 runtime/probe tests, including wrapped-model rejection, coherent local/remote non-OOM termination, one status gather with no barrier, an OOM candidate followed by a later successful candidate, complete rollback, and an unchanged fake accumulation cursor.
- Validation: `rtk timeout 240s .venv/bin/accelerate launch --num_processes 8 --main_process_port 0 scripts/smoke_balalaika_accelerate.py --mode probe-rank-fault` exited 0 before timeout. Exactly rank 7 injected CUDA OOM after local backward for candidate 2 while peers completed local optimizer work; every rank selected prior candidate 1, then reported normal prepared accumulation `[false,true]` and optimizer step 1. No hang or cursor shift occurred.
- Validation: The reordered original smoke passed under `rtk timeout 180s .venv/bin/accelerate launch --num_processes 1 ... --mode train-checkpoint` and `rtk timeout 240s .venv/bin/accelerate launch --num_processes 8 --main_process_port 0 ... --mode train-checkpoint`. The full CPU suite passes 220 tests. Fix-round logs are stored beside prior Task 10 artifacts.
- Decision: The second allowed minimal Accelerate fix passed the rank-fault and original smokes; Accelerate remains primary and native DDP remains unimplemented.
- Follow-up: The minor smoke-test RNG assertion finding remains deferred as directed; production CPU/local-CUDA RNG rollback remains covered.

## 2026-08-02 - Distributed Balalaika evaluator

- Decision: Keep the evaluator's production boundary dependency-injected: the runtime, Task-5 selection, boundary-local Task-7 ledger, per-rank ASR factory, and rank-aware tracking manager are supplied by the trainer. The evaluator derives and validates all item inputs from the current checkpoint, boundary, benchmark rows, prompt files, and selection fingerprint.
- Decision: Treat aggregate JSON/JSONL as a deterministic durable snapshot. W&B logging must finish before `validation-complete.json` is atomically published; only a matching durable completion permits retention.
- Decision: Retain the four selection-owned WAVs in a compact per-boundary archive before pruning older completed boundaries' full `wavs/` trees. Incomplete boundaries are never pruned.
- Changed: The evaluator restores the unwrapped model and retained AudioVAE modes only after the per-rank ASR adapter has been closed. Generation temporarily attaches the AudioVAE, while every ledger mutation receives the live token/epoch claim.
- Validation: RED was the absent `evaluation.py` import and the absent smoke `validation` CLI. GREEN passes 9 focused evaluator tests, 66 evaluator/ledger/metrics/tracking/runtime tests, and all 229 repository tests. Black, `py_compile`, and `git diff --check` pass.
- Validation: The final-tree required no-network smoke passed with eight real Accelerate processes and 32 synthetic items. Every rank generated four unique position-modulo IDs, released one fake ASR session, and verified the completion; rank 0 published one aggregate, one completion, and one tracking record. Artifacts are under `.superpowers/sdd/2026-08-02-balalaika-two-stage-lora-training/task-11-artifacts/validation-8x-1785702862763622381`.
- Environment: Two earlier 240/150-second diagnostics used an allocated PTY and timed out before script entry because the Accelerate parent was job-control stopped (`T` state). The unchanged required command without a PTY exited 0 in about 112 seconds; this was a harness issue, not an evaluator fix.
