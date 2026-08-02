# Task 12 Report: Two-stage LoRA trainer

## Result

Implemented the Task 12 two-stage trainer and the required Task 9 recovery-checkpoint extension. Verification uses tiny CPU-only model/runtime/evaluator/checkpoint doubles; no real 3B model, proprietary corpus, GPU training, or network operation ran.

## Architecture and interfaces

- `build_model(config, stage, adapter_checkpoint=None)` verifies every file in the pinned local VoxCPM2 Hub manifest, constructs the repository VoxCPM2 LoRA configuration, freezes every non-`lora_` parameter, retains/freeze the AudioVAE, and detaches it from the train wrapper. Optional adapter loading accepts only `VerifiedAdapterCheckpoint`, so direct unverified paths cannot bypass checkpoint identity/tensor validation.
- `BalalaikaTrainer.run_stage(stage) -> Path` takes injected runtime, checkpoint manager, evaluator factory, approval verifier, identity, and test seams for model/data/optimizer/scheduler/probe construction. Stage 1 verifies manual approval before model setup and binds it to the current base/data/LoRA identities.
- Stage 2 loads a fully verified final stage-1 boundary adapter before creating its fresh optimizer/scheduler, copies global progress, and resets all stage-relative fields. A same-stage stage-2 resume instead verifies/restores its own full checkpoint and does not require stage 1.
- The local unwrapped model/optimizer are available to the optional microbatch selector before one joint `runtime.prepare(model, optimizer, loader, scheduler)` call. AdamW defaults to only `requires_grad` LoRA tensors; cosine/warmup total steps are sized to the selected stage geometry.
- Each microbatch uses `runtime.accumulate`, weighted `loss/*`, and `runtime.backward`. Clip/optimizer/zero occur only on synchronized groups; scheduler/progress advance only when the optimizer was not skipped. Data exhaustion after an AMP skip replays the deterministic epoch until the planned real update count is reached.
- Boundary transactions are ordered checkpoint -> resumable evaluator -> atomic completion marker. A restored unfinished boundary completes before the next loader update; recovery-kind resume does not falsely replay its already completed boundary cursor.

## Recovery checkpoint contract

- Checkpoint schema 3 records `checkpoint_kind` (`boundary` or `recovery`) and `accumulation_microstep`.
- `save_same_stage` retains exact eighth-boundary geometry. `save_recovery` accepts any safe optimizer position only when the accumulation remainder is zero and the boundary cursor exactly matches all boundaries already reached.
- Same-stage restore accepts either verified kind with the prior exact identity, LoRA-key/shape, Accelerate state-file, RNG, and progress checks. Stage transition explicitly rejects recovery kind even if it is at final-stage geometry.
- Signals are cooperative: finish the current real synchronized step, gather the stop decision, barrier, save recovery, and stop. Rank-local loader/forward/optimizer errors publish status before the next distributed operation. A broken backward collective never attempts a divergent barrier/save and raises `TrainingRestartRequired` with the last durable checkpoint.

## TDD and validation evidence

- Initial trainer RED: `ModuleNotFoundError: voxcpm.training.balalaika.trainer`.
- Recovery RED: three failures because `CheckpointManager.save_recovery` did not exist; GREEN: 41 checkpoint tests.
- Additional targeted RED/GREEN covered DataLoader failure coordination, stage-2 recovery-only resume, and approval identity binding. A deliberate skipped-step mutation made the scheduler/progress regression fail, then passed after restoring the implementation.
- Focused integration: `85 passed` across trainer, checkpoint, evaluator, runtime, and probe tests.
- Full CPU suite: `256 passed, 5 warnings in 94.36s`.
- Static: Black check passed; `python -m compileall -q src scripts` passed; `git diff --check` passed.

## Ordering and state assertions

- Stage 1 emits exactly 16 and stage 2 exactly 24 checkpoint/evaluator/marker triples; step zero is absent and epoch-end appears once.
- Tests assert checkpoint exists before evaluator entry and completion marking occurs only after evaluator return.
- Boundary resume event order is restore -> evaluator -> marker -> first accumulation. Recovery resume skips false validation replay and continues at the next microbatch.
- Non-boundary signal at optimizer step 11 writes recovery step 11 after scheduler/progress update. Coordinated pre-backward exception writes the last safe progress and re-raises the original exception. Broken backward leaves the step-10 boundary as the last durable state and performs no recovery save.

## Concerns and scope

- Distributed failure safety is intentionally conservative: exceptions inside a broken DDP backward collective require launcher/process restart because no safe collective checkpoint can be guaranteed.
- Full-scale performance of per-phase status gathers and real VoxCPM2 forward memory remains for the already-planned GPU smoke/preflight tasks; this task deliberately did not launch them.

## Fix Round 1

- Scheduler ownership is now explicit: `Accelerator(step_scheduler_with_optimizer=False)` leaves the trainer's one `scheduler.step()` per real global optimizer update as the sole schedule owner. The trainer still performs one joint `prepare(model, optimizer, loader, scheduler)`.
- Recovery safety uses a separate in-flight accumulation cursor. A loader/forward/optimizer failure after an unsynchronized backward requires process restart; recovery is allowed only at cursor zero after a non-skipped synchronized optimizer, scheduler, and progress update. Stop decisions remain latched across skipped attempts.
- Same-stage recovery at an exact validation boundary verifies the durable marker. Missing, malformed, or identity-mismatched markers rerun evaluator completion and are atomically replaced before another update; matching markers suppress duplicate evaluation.
- Accelerate state save, checkpoint setup/state/finalize/visibility, and rank-zero boundary marker publication now gather a single success outcome before any rank raises. Broken outcome collectives produce restart-required errors and never trigger another recovery collective.
- The mandated curriculum is enforced before model construction: stage 1 is exactly 2 epochs at `1e-4`; stage 2 is exactly 3 epochs at `5e-5`, preserving 16 and 24 validations respectively.
- Targeted RED/GREEN coverage includes accumulation>1 mid-group loader/forward faults, skipped-step signal latching, exact-boundary recovery before/after/mismatched marker, same-epoch replay seeding and next-epoch increment, curriculum drift, rank-local/peer state-save, finalize and marker failures, and broken collectives. Focused integration passes `107` tests; the full CPU suite passes `278` tests with 5 warnings.
- A final real eight-process Accelerate 1.14 CPU/Gloo regression passed on the formatted tree. All ranks reported one joint prepare, 4 synchronized attempts including one synthetic skipped update, 3 real optimizer steps, exactly 3 scheduler steps (not 24), identical sample prefixes when replaying sampler epoch 17, and a changed order after advancing once to epoch 18.

## Fix Round 2

An exact-boundary signal recovery previously lost the durable proof produced by the immediately preceding boundary transaction. The marker remained keyed and fingerprint-bound to `boundary-0010`, while resume looked only for a marker keyed to `recovery-0010`, so validation ran twice.

Recovery metadata now carries a strict `completed_boundary_proof`: boundary checkpoint name, checkpoint fingerprint, and exact evaluator boundary identity. The recovery metadata fingerprint protects that proof. On resume, rank zero accepts it only after fully verifying the referenced immutable boundary checkpoint against the restored progress and expected identity, and then verifying its exact durable marker. Every rank receives the same decision. Missing, malformed, corrupt, or mismatched proof remains incomplete and follows the existing evaluator/marker completion path.

RED command and exact result:

```text
rtk run '.venv/bin/pytest -q tests/training/balalaika/test_trainer.py::test_exact_boundary_signal_recovery_reuses_production_boundary_marker_on_resume'

FAILED tests/training/balalaika/test_trainer.py::test_exact_boundary_signal_recovery_reuses_production_boundary_marker_on_resume
E       assert [10] == []
1 failed, 4 warnings in 4.17s
```

GREEN command and exact result:

```text
rtk run '.venv/bin/pytest -q tests/training/balalaika/test_trainer.py::test_exact_boundary_signal_recovery_reuses_production_boundary_marker_on_resume'

1 passed, 4 warnings in 4.09s
```

Focused trainer/checkpoint command and exact result:

```text
rtk run '.venv/bin/pytest -q tests/training/balalaika/test_trainer.py tests/training/balalaika/test_checkpoint.py'

76 passed, 4 warnings in 6.14s
```

Broader relevant command and exact result:

```text
rtk run '.venv/bin/pytest -q tests/training/balalaika/test_trainer.py tests/training/balalaika/test_checkpoint.py tests/training/balalaika/test_evaluation.py tests/training/balalaika/test_runtime.py tests/training/balalaika/test_probe.py'

108 passed, 4 warnings in 8.01s
```
