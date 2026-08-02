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
