# Balalaika Two-Stage LoRA Training Design

## Objective

Extend VoxCPM with a reproducible, resumable pipeline that fine-tunes
`OpenBMB/VoxCPM2` on `/workspace/balalaika_proprietary_v2` using eight GPUs.
Training uses a two-stage ASR-agreement curriculum, is gated by a four-example
memorization experiment, and evaluates Russian number rendering in Weights &
Biases (W&B) every one eighth of an epoch.

## Fixed Inputs and Decisions

- Base model: `OpenBMB/VoxCPM2`, pinned to an immutable Hub revision during
  preparation.
- Corpus audio and source metadata: the 519
  `/workspace/balalaika_proprietary_v2/train/shard_*.tar` files.
- Agreement metadata: the canonical ROVER result archive at
  `/workspace/balalaika_proprietary_v2/punctuation_artifacts/20260729T135419Z/balalaika-rover-results-20260729T135419Z.tar.zst`.
- Training text:
  `/workspace/balalaika_proprietary_v2/combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl`,
  specifically its `rover_punctuated_accented` field.
- The combined sidecar metadata declares 4,075,032 complete rows and a trusted
  provenance chain. Preparation must verify its recorded SHA-256 before use.
- The 309 rows with null `asr_agreement_mean` are excluded from both stages and
  reported in the preparation audit.
- Stage 1 contains non-null rows with `asr_agreement_mean < 0.95` and trains for
  two epochs.
- Stage 2 contains rows with `asr_agreement_mean >= 0.95`, initializes from the
  final stage-1 LoRA weights, resets optimizer and scheduler state, and trains
  for three epochs.
- Benchmark: the private
  `bitmanagerai/hard_number_eval_for_tts` dataset, pinned to an immutable Hub
  revision resolved using authenticated Hugging Face tooling.
- Benchmark synthesis input: `stressed`. Scoring reference:
  `normalized_gold`.
- Validation ASR: GigaAM v3 RNN-T, pinned to an immutable model/runtime
  identity in run metadata.
- Validation uses 20 fixed reference voices and a fixed random assignment from
  all 2,000 benchmark rows to those voices. Both selections are seed-controlled
  and remain unchanged across checkpoints.
- The memorization experiment and large training are separate commands. Large
  training requires an explicit manual approval record tied to the completed
  memorization W&B run and checkpoint.

## User Workflow

The operator-facing workflow has explicit phases:

1. Download and pin the VoxCPM2 base model and benchmark revisions.
2. Run corpus preparation and review the generated audit.
3. Run the four-example memorization experiment.
4. Review its reference/generated audio pairs and diagnostics in W&B.
5. Record explicit memorization approval.
6. Launch stage 1 on eight GPUs.
7. Launch stage 2 from stage 1's final adapter.

Preparation, memorization, approval, and both training stages are distinct CLI
operations. This prevents a completed memorization experiment from
automatically starting a multi-day training run.

## Architecture

### Preparation

Preparation performs a strict identity join on `source_relative_path` among:

- the tar audio member and source JSON metadata;
- the ROVER agreement row; and
- the combined stressed/punctuated text row.

It writes a compact indexed representation with the tar path, audio member
offset and size, JSON member offset and size, normalized identity, selected
text, agreement fields, speaker identity, duration, and relevant quality
fields. Audio is not extracted or duplicated. The indexed audio reader uses
direct offsets and cached read-only tar file handles.

Preparation rejects:

- duplicate, missing, or unexpected identities;
- missing or inconsistent audio/JSON pairs;
- disagreement among source paths after normalization;
- missing or empty combined training text;
- sidecar or archive identity/hash mismatches;
- unexpected corpus row or shard counts; and
- malformed agreement values.

The preparation audit includes total and eligible rows, exact stage sizes,
null-agreement exclusions, other filtering reasons, per-shard counts, duration
statistics, agreement distributions, valid-ASR-count distributions, and
speaker counts. Each output records a fingerprint over all input identities,
hashes, filtering rules, and preparation configuration.

### Indexed Dataset

The training dataset is map-style and has an exact length. It reads compressed
audio bytes directly from indexed tar offsets, decodes and resamples them only
when requested, and tokenizes `rover_punctuated_accented` for VoxCPM2.

Training uses ordinary fixed-size shuffled batches. Duration bucketing is
explicitly out of scope. On the primary path, the input DataLoader is not
pre-sharded with a PyTorch `DistributedSampler`; Accelerate alone shards its
seeded fixed-batch sampler across ranks. This avoids double sharding. The
prepared loader gives each rank a disjoint subset and the same number of
complete batches, exposes epoch reseeding through Accelerate's prepared-loader
contract, and makes sample order a pure function of the recorded seed and
epoch. Any remainder that cannot make an equal complete distributed batch is
dropped and counted in run metadata. The controlled native DDP fallback uses
`DistributedSampler` with equivalent seed, `drop_last`, and `set_epoch()`
semantics because Accelerate is absent on that path.

DataLoader workers partition reads without duplicating samples. Each worker
maintains a bounded file-handle cache so it does not repeatedly reopen the 519
tar files.

### Deterministic Selections

Preparation creates versioned selection manifests for:

- four memorization examples; and
- 20 validation voice prompts.

Both selections come from the `>= 0.95` pool, use distinct speakers, require
single-speaker audio, and apply configurable duration and quality constraints.
The default voice-prompt duration window is 3--10 seconds. If the configured
constraints cannot produce enough distinct speakers, preparation fails and
reports which constraint exhausted the pool rather than silently relaxing it.

The 2,000 benchmark-to-voice assignments are generated once from a recorded
seed and stored by benchmark row ID. The prompt audio and the corresponding
`rover_punctuated_accented` text are passed together to VoxCPM2 continuation
generation.

## Memorization Gate

The memorization command loads the pinned base model, creates the configured
LoRA adapter, and repeatedly trains on the four selected samples. Its number of
updates, learning rate, generation settings, and seed are configurable and
recorded. It produces a normal LoRA checkpoint plus a W&B artifact containing:

- all four reference audio files;
- all four generated audio files;
- their training text;
- loss curves;
- GigaAM transcripts and text error rates; and
- non-gating audio similarity diagnostics where practical.

Because diffusion synthesis is not expected to reproduce a waveform sample by
sample, no arbitrary waveform-distance threshold decides success. The command
stops after logging results. The operator reviews the pairs and runs a separate
approval command. Approval writes a small record containing the W&B run ID,
checkpoint fingerprint, base-model revision, data-index fingerprint, LoRA
configuration, approver identity available from the environment, and
timestamp. Stage 1 refuses an absent or mismatched approval record.

## Distributed LoRA Training

Training primarily uses Hugging Face Accelerate, launched with
`accelerate launch` as one process per GPU, with NCCL-backed distributed data
parallelism and BF16 mixed precision on eight RTX 5090 GPUs. Model, optimizer,
dataloader, and scheduler preparation, gradient accumulation, gradient
clipping, cross-rank synchronization, and state restoration use Accelerate's
public APIs. Only LoRA parameters are optimized. The initial adapter
configuration is:

- language-model LoRA enabled;
- diffusion-transformer LoRA enabled;
- projection LoRA disabled;
- rank 32;
- alpha 32; and
- dropout 0.

The configuration remains externally configurable but becomes part of the
checkpoint and resume fingerprint.

Native PyTorch DDP is a controlled fallback, not a second routinely supported
launcher. The implementation first performs an Accelerate integration spike
covering eight-rank sample partitioning, gradient accumulation without
unnecessary synchronization, LoRA gradient reduction, checkpoint/resume, and
distributed validation aggregation. It falls back to `torchrun` plus native
DDP only if that spike has a reproducible blocking failure that cannot be
resolved without changing model behavior or expanding scope materially. The
failure reproduction and fallback decision are recorded in implementation
notes and run metadata. Both paths must preserve the same configuration,
dataset, checkpoint, and metric semantics.

A bounded startup probe tests candidate fixed per-GPU microbatch sizes on a
representative long eligible sample and selects the largest size that completes
forward, backward, optimizer, and synchronization steps on every rank. The
selected microbatch is then fixed for the run. Operators may bypass the probe
with an explicit fixed value. Gradient accumulation is configurable and is
reported together with per-GPU and global effective batch sizes.

Default optimization settings are:

- stage-1 learning rate: `1e-4`;
- stage-2 learning rate: `5e-5`;
- cosine decay;
- proportional warmup, stored as a fraction of stage optimizer steps;
- gradient clipping; and
- optimizer steps defined after gradient accumulation.

Stage 2 loads only the final stage-1 LoRA parameters. It creates a fresh
optimizer and learning-rate scheduler and begins a new stage-relative step
count. A global progress counter is also logged so charts can be viewed across
both stages.

## Fractional-Epoch Scheduling

An epoch is defined from the prepared stage row count, world size, fixed
per-GPU batch size, and gradient accumulation. Each epoch has eight validation
boundaries, calculated with integer arithmetic from completed optimizer steps.
Every boundary is emitted exactly once. There is no full validation at step
zero, and an eighth boundary that coincides with epoch end is not emitted a
second time.

The same boundaries trigger checkpoint creation. Consequently stage 1 has 16
planned full validations and stage 2 has 24 planned full validations.

## Validation Pipeline

At each boundary, training synchronizes all ranks and enters evaluation mode.
The 2,000 benchmark rows are divided deterministically across the eight ranks.
Each row is synthesized from `stressed` using its preassigned voice prompt.
Generated files are stored beneath a validation directory keyed by stage,
epoch, boundary, benchmark revision, checkpoint fingerprint, and generation
configuration.

Each generated waveform is transcribed with the pinned GigaAM v3 RNN-T
evaluator. Item-level manifests make both generation and ASR resumable. A
completed item is reused only when all recorded input and output hashes match.
After all ranks finish, rank 0 verifies exactly 2,000 unique generated IDs and
2,000 completed ASR attempts, then aggregates metrics. An ASR call that
completes successfully with an empty hypothesis is retained and scored as an
empty hypothesis; it is not treated as missing evaluation data.

Scoring normalizes case, Russian `ё`/`е`, stress marks, whitespace, and
punctuation according to one tested function shared by utterance and number
metrics. Number spans are derived by aligning the benchmark's digit-bearing
`text` to `normalized_gold`, then mapping that gold span through the alignment
to the ASR hypothesis.

The required metrics are corpus-level micro-averages computed from summed edit
counts and correct unit-specific denominators:

- `val/num_cer`;
- `val/num_wer`;
- `val/utt_cer`; and
- `val/utt_wer`.

Metrics by the benchmark's 12 categories, edit counts, generation and ASR
failure counts, duration, throughput, and latency are also logged. Metric code
must not reuse word counts as character-error denominators.

## W&B Structure

Memorization, stage 1, and stage 2 use separate resumable W&B runs under a
shared group and experiment identifier. Rank 0 is the only process that writes
to W&B. Each full validation logs:

- all required aggregate and per-category metrics;
- stage-relative and global progress;
- configuration and input fingerprints;
- a table with 2,000 item IDs, categories, input/gold/stressed text, voice ID,
  ASR hypothesis, number spans, and item errors; and
- four fixed benchmark generations as `wandb.Audio` entries.

The four audio-log IDs are chosen once, stored in the selection manifest, and
held fixed across checkpoints. All 2,000 generated files remain local until
generation, ASR, scoring, and W&B artifact creation succeed. Retention then
keeps the latest full validation directory and all four logged examples; this
policy is configurable.

## Checkpoint and Resume Semantics

Checkpoints are written atomically and include:

- LoRA weights;
- optimizer and scheduler state for the current stage;
- Accelerate mixed-precision and distributed state, including scaler state
  when applicable;
- stage, epoch, fractional boundary, microstep, and optimizer step;
- random-number-generator state;
- distributed sampler seed and epoch;
- base-model and evaluator revisions;
- data-index and selection-manifest fingerprints;
- LoRA and optimization configuration; and
- W&B run/group identities.

Resume verifies all immutable fields before loading. A mismatch in data,
world size, base model, LoRA structure, stage definition, batch parameters, or
selection manifests is an error. Stage-2 initialization is distinct from
same-stage resume: it accepts stage-1 LoRA weights but intentionally rejects
stage-1 optimizer/scheduler state.

## Failure Handling

Indexes, selection manifests, approval records, checkpoints, per-item
evaluation manifests, and aggregate metrics are published by atomic rename.
Interrupted temporary outputs are never treated as complete.

A full validation succeeds only after exactly 2,000 generated waveforms and
2,000 completed ASR records pass identity and hash checks. Empty hypotheses
from otherwise successful ASR calls remain valid records and contribute their
full edit error. Generation or ASR exceptions receive a bounded number of
deterministic retries. A persistent exception saves the current training state
and stops the run. Resume finishes the interrupted validation before another
training optimizer step.

W&B run IDs and local validation artifacts are persisted before upload.
Temporary network failure is retried and can be resumed without regenerating
audio. The run does not mark the validation boundary complete until the W&B
payload or an explicitly configured offline W&B transaction is durably saved.

## Verification Strategy

Automated tests cover:

- strict identity joining and provenance/hash checks;
- tar-member offset reads and bounded file-handle caching;
- the exact `< 0.95` and `>= 0.95` split boundary;
- exclusion and reporting of null agreement;
- deterministic four-example, 20-voice, and 2,000-assignment manifests;
- Accelerate distributed sample uniqueness, equal batch counts, dropped
  remainders, gradient synchronization, and epoch reseeding;
- exact one-eighth-epoch trigger calculation;
- number-span extraction and golden CER/WER results using character and word
  denominators;
- atomic checkpoint publication and strict resume mismatch handling;
- stage-2 adapter-only initialization with optimizer reset; and
- validation item retry/resume and exact 2,000-item aggregation.

A synthetic integration fixture constructs small tar shards and sidecars and
exercises preparation, both agreement splits, data loading, checkpoint/resume,
and validation aggregation without proprietary data. GPU verification then
proceeds in increasing cost:

1. a single-GPU forward/backward/checkpoint smoke test;
2. an eight-GPU Accelerate smoke test that checks disjoint samples,
   synchronized optimizer steps, LoRA gradients, accumulation behavior, and
   restore behavior;
3. a small distributed validation smoke test; and
4. the real four-example memorization experiment and manual W&B review.

If and only if the documented Accelerate spike triggers the controlled native
DDP fallback, the same eight-GPU and distributed-validation smoke tests are
rerun through that fallback before memorization.

The large stage-1 run is not launched as part of implementation or verification;
it begins only after the operator creates the matching approval record and
explicitly invokes the stage-1 command.

## Scope Boundaries

This work does not modify the proprietary source tar files or their sidecars,
does not extract the corpus into millions of standalone files, does not add
duration-bucketed batching, does not train or alter the validation ASR, and
does not automatically launch large training after memorization. TensorBoard
may remain available for backward compatibility, but W&B is the authoritative
tracker for this pipeline.
