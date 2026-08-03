# Balalaika two-stage LoRA operator runbook

This workflow fine-tunes `OpenBMB/VoxCPM2` against the immutable Balalaika
corpus. The corpus tar files and combined sidecar under
`/workspace/balalaika_proprietary_v2` are read-only inputs. Generated indexes,
selections, checkpoints, validation audio, and W&B state live under
`/workspace/balalaika_lora_training` as configured in
`conf/voxcpm_v2/balalaika_lora.yaml`.

The workflow has a hard manual boundary: `memorize` stops after the four-pair
diagnostic is logged. A person must inspect that W&B run and separately invoke
`approve`. Neither `prepare`, `memorize`, nor `approve` starts stage training.

## Environment and authentication

Run commands from the VoxCPM repository root:

```bash
rtk uv sync --extra dev --extra balalaika
hf auth whoami
wandb login
```

If `hf auth whoami` is not the account authorized for the private
`bitmanagerai/hard_number_eval_for_tts` dataset, use `hf auth login` and retry.
The `pin` command invokes only the modern `hf` CLI through the pipeline's Hub
module: it resolves immutable revisions with `hf models info` / `hf datasets
info` and downloads them with `hf download`. Tokens remain in the operator's
Hugging Face credential store or `HF_TOKEN`; the YAML contains no credential.

Set `wandb.entity`, `wandb.project`, `wandb.group`, and `wandb.mode` in the
YAML. Memorization approval requires `mode: online`, because the operator must
inspect the four uploaded reference/generated pairs. Stage training and
validation may use `mode: offline`; after connectivity returns, synchronize a
run directory with:

```bash
rtk uv run wandb sync /workspace/balalaika_lora_training/wandb/wandb/offline-run-*
```

## Pin, prepare, and audit

Run the read-only input preflight in order:

```bash
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml pin
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml prepare
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml audit
```

`pin` must report immutable revisions and hashed local files for VoxCPM2, the
private number benchmark, and `istupakov/gigaam-v3-onnx`. `prepare` verifies
the trusted corpus manifests and the bytes of all inputs before atomically
publishing its index and fixed selections. It never rewrites or extracts the
source corpus.

A successful production `audit` reports all of the following:

- 519 source shards and 4,075,032 joined rows;
- exactly 309 null-agreement rows excluded from both stages;
- stage 1 (`asr_agreement_mean < 0.95`) and stage 2
  (`asr_agreement_mean >= 0.95`) counts whose sum is 4,074,723;
- four unique high-agreement memorization samples;
- 20 unique, seed-controlled corpus prompts;
- 2,000 fixed benchmark-to-prompt assignments and four fixed audio-log IDs;
- an index fingerprint, index byte size, and no pin, identity, or hash error.

Stop if any expected count changes. Do not edit the expectation merely to make
the audit pass; reconcile the corpus provenance first.

## Memorize, inspect, and approve

Launch the four-example diagnostic on eight processes:

```bash
rtk uv run accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml memorize
```

The command trains only the fixed four examples, logs the four reference and
generated pairs plus loss/ASR diagnostics, saves a LoRA checkpoint, and stops.
It does not decide whether the waveform is good enough and does not start a
large run. Inspect the complete W&B pair table, audio, loss curve, GigaAM
transcripts, and error diagnostics. Record approval only if the displayed run
ID is the run you reviewed:

```bash
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml approve --wandb-run-id RUN_ID
```

Approval binds the reviewed W&B run, local completion evidence, all eight
audio files, checkpoint, pinned base revision, index/selection fingerprints,
and LoRA configuration. Changing any bound artifact invalidates it.

## Train stage 1 and stage 2

Stage 1 re-verifies approval before runtime, model, or dataset setup and
rejects any process count other than eight. Launch it with:

```bash
rtk uv run accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml train --stage 1
```

Stage 1 has two epochs and therefore 16 exact one-eighth-epoch validation and
checkpoint boundaries. After it finishes, locate the final boundary-8 adapter
reported by the command. Stage 2 requires that verified completed adapter,
loads only its LoRA weights, resets optimizer/scheduler state, and runs three
epochs with 24 validation boundaries:

```bash
rtk uv run accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml train --stage 2 --stage1-checkpoint /workspace/balalaika_lora_training/runs/stage1/checkpoints/FINAL_CHECKPOINT
```

After reviewing the selected final checkpoint, the canonical command may omit
the explicit path; the CLI then resolves `stage1/checkpoints/latest.json` and
still verifies that it names the completed epoch-2 boundary-8 adapter:

```bash
rtk uv run accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml train --stage 2
```

Before a restart, verify a checkpoint against current pinned/data identities:

```bash
rtk uv run accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml validate --checkpoint CHECKPOINT
```

Resume the same stage with the exact prior checkpoint and unchanged config:

```bash
rtk uv run accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml train --stage 1 --resume CHECKPOINT
```

For stage 2, use `--stage 2 --resume CHECKPOINT`; do not also pass a stage-1
transition checkpoint. A checkpoint at a partially completed validation is
resumed through the same training command: the evaluator reuses only
hash-matching item manifests and finishes the exact 2,000-item boundary before
the next optimizer step. Do not delete its incomplete validation directory.

SIGINT and SIGTERM are cooperative. The trainer finishes the current complete
gradient-accumulation group, synchronizes ranks, writes a recovery checkpoint,
and exits before another microbatch. A fault inside an unsafe distributed
collective reports the last durable checkpoint and requires a fresh launch.

Disk retention keeps the newest complete boundary's full 2,000 WAV tree and
the four fixed W&B examples from older completed boundaries. Incomplete
boundaries are never pruned. Preserve checkpoints needed for rollback and
stage transition; archive or delete other completed artifacts only after W&B
publication and local hashes have been verified.

## Verification scope

Implementation verification is intentionally limited to synthetic CPU tests
and the documented synthetic GPU smoke script:

```bash
rtk uv run accelerate launch --num_processes 1 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint
rtk uv run accelerate launch --num_processes 8 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint
rtk uv run accelerate launch --num_processes 8 scripts/smoke_balalaika_accelerate.py --mode validation --items 32
```

Those commands do not load production corpus rows or VoxCPM2 weights. No real
memorization, approval record, stage-1 run, or stage-2 run is started as part
of implementation verification. Large training begins only when an operator
has reviewed W&B, created the matching approval, and explicitly launched it.
