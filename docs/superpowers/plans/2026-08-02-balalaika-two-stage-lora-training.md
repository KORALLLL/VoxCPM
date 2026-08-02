# Balalaika Two-Stage LoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a manually gated, resumable two-stage LoRA pipeline that trains pinned VoxCPM2 on the Balalaika corpus with eight GPUs and evaluates all 2,000 hard-number prompts in W&B every one eighth of an epoch.

**Architecture:** A one-time preparation command creates an atomic SQLite index that joins tar offsets, ROVER agreement, and combined stressed text without extracting the corpus. Focused modules then provide indexed reads, deterministic selections, hard-number scoring, GigaAM ASR, W&B logging, Accelerate-first distributed training, LoRA-only checkpoints, validation, and the memorization approval gate. Native PyTorch DDP is implemented only if the required Accelerate spike produces a documented blocking failure.

**Tech Stack:** Python 3.10+, PyTorch/torchaudio/torchcodec, Hugging Face Accelerate, SQLite, zstandard, Hugging Face `hf` CLI, onnx-asr with CUDA, W&B, safetensors, pytest, eight RTX 5090 GPUs.

## Global Constraints

- Base model is `OpenBMB/VoxCPM2`, resolved and pinned to an immutable Hub revision.
- Corpus root is `/workspace/balalaika_proprietary_v2`; never modify its tar files or sidecars.
- Training text is only `rover_punctuated_accented` from `combined_sidecars/rover-punctuation-stress-v1/rover-punctuation-stress.jsonl`.
- Stage 1 is non-null `asr_agreement_mean < 0.95` for two epochs at `1e-4`.
- Stage 2 is `asr_agreement_mean >= 0.95` for three epochs at `5e-5`; load LoRA weights only and reset optimizer/scheduler.
- Exclude and audit all 309 null-agreement rows.
- LoRA defaults are LM=true, DiT=true, projection=false, rank=32, alpha=32, dropout=0.
- Use fixed-size shuffled batches; do not add duration bucketing.
- Accelerate is the primary runtime and solely owns primary-path dataloader sharding.
- Validate after each completed 1/8 epoch, excluding step zero and duplicate epoch-end triggers.
- Every validation generates and scores exactly 2,000 benchmark rows using fixed seeded assignments to 20 fixed audio prompts sampled uniformly from the joined corpus.
- Synthesize benchmark `stressed`; score GigaAM v3 RNN-T hypotheses against `normalized_gold`.
- Log `val/num_cer`, `val/num_wer`, `val/utt_cer`, `val/utt_wer`, a 2,000-row W&B table, and four fixed audio examples.
- Large training requires a manual approval record tied to the completed four-example memorization run and checkpoint.
- All commands in this repository must be prefixed with `rtk`.
- Follow test-driven development and update `implementation-notes.md` with material decisions, Accelerate spike evidence, and validation results.

## File Structure

- `src/voxcpm/training/balalaika/config.py`: validated YAML configuration models.
- `src/voxcpm/training/balalaika/artifacts.py`: atomic files, hashes, fingerprints, and run metadata.
- `src/voxcpm/training/balalaika/hub.py`: authenticated `hf` CLI pin/download operations.
- `src/voxcpm/training/balalaika/index.py`: corpus scan, strict join, SQLite schema, and audit.
- `src/voxcpm/training/balalaika/dataset.py`: direct tar-offset reads and VoxCPM batches.
- `src/voxcpm/training/balalaika/selection.py`: memorization examples, random audio prompts, and benchmark assignments.
- `src/voxcpm/training/balalaika/metrics.py`: normalization, alignments, number spans, CER/WER, aggregation.
- `src/voxcpm/training/balalaika/asr.py`: pinned GigaAM v3 RNN-T adapter.
- `src/voxcpm/training/balalaika/ledger.py`: resumable per-item validation records and retries.
- `src/voxcpm/training/balalaika/tracking.py`: rank-zero W&B lifecycle, tables, and audio.
- `src/voxcpm/training/balalaika/schedule.py`: exact epoch geometry and 1/8 boundaries.
- `src/voxcpm/training/balalaika/checkpoint.py`: LoRA-only Accelerate save/load hooks and strict metadata.
- `src/voxcpm/training/balalaika/runtime.py`: Accelerate runtime and the conditional native-DDP protocol.
- `src/voxcpm/training/balalaika/probe.py`: synchronized fixed-microbatch probe.
- `src/voxcpm/training/balalaika/evaluation.py`: eight-rank generation, ASR, aggregation, and W&B handoff.
- `src/voxcpm/training/balalaika/trainer.py`: stage training loop and stage-2 initialization.
- `src/voxcpm/training/balalaika/memorization.py`: four-item overfit and manual approval gate.
- `src/voxcpm/training/balalaika/cli.py`: `pin`, `prepare`, `memorize`, `approve`, `train`, and `validate` commands.
- `conf/voxcpm_v2/balalaika_lora.yaml`: complete local configuration.
- `scripts/smoke_balalaika_accelerate.py`: real one/eight-GPU distributed smoke entrypoint.
- `docs/balalaika_training.md`: operator runbook and recovery procedures.
- `tests/training/balalaika/`: focused unit and synthetic integration tests.

---

### Task 1: Configuration, atomic artifacts, and dependency boundary

**Files:**
- Create: `src/voxcpm/training/balalaika/__init__.py`
- Create: `src/voxcpm/training/balalaika/config.py`
- Create: `src/voxcpm/training/balalaika/artifacts.py`
- Create: `tests/training/balalaika/test_config.py`
- Create: `implementation-notes.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `BalalaikaConfig.load(path: Path) -> BalalaikaConfig`.
- Produces: `atomic_json(path: Path, value: Mapping[str, Any]) -> None`.
- Produces: `sha256_file(path: Path) -> str` and `fingerprint(value: Any) -> str`.
- Dependency extra: `balalaika` containing Accelerate, W&B, zstandard, and `onnx-asr[gpu,hub]`.

- [ ] **Step 1: Write failing configuration and atomic-write tests**

```python
def minimal_config(tmp_path):
    return {
        "data": {"corpus_root": str(tmp_path / "corpus")},
        "output_dir": str(tmp_path / "runs"),
    }

def test_default_curriculum_is_exact(tmp_path):
    cfg = BalalaikaConfig.model_validate(minimal_config(tmp_path))
    assert cfg.stage1.agreement == "lt"
    assert cfg.stage1.threshold == 0.95
    assert cfg.stage1.epochs == 2
    assert cfg.stage2.agreement == "ge"
    assert cfg.stage2.epochs == 3
    assert cfg.lora.model_dump() == {
        "enable_lm": True, "enable_dit": True, "enable_proj": False,
        "r": 32, "alpha": 32, "dropout": 0.0,
    }

def test_atomic_json_never_leaves_temporary_file(tmp_path):
    target = tmp_path / "state.json"
    atomic_json(target, {"step": 7})
    assert json.loads(target.read_text()) == {"step": 7}
    assert list(tmp_path.glob(".*.tmp")) == []
```

- [ ] **Step 2: Run the focused tests and confirm missing modules fail**

Run: `rtk uv run pytest tests/training/balalaika/test_config.py -q`

Expected: FAIL during import because `voxcpm.training.balalaika.config` does not exist.

- [ ] **Step 3: Add validated Pydantic models and atomic primitives**

Implement `DataConfig`, `HubConfig`, `LoRAConfig`, `StageConfig`,
`ValidationConfig`, `WandbConfig`, and `BalalaikaConfig`. Reject thresholds
outside `[0, 1]`, nonpositive epochs/batches, stage operators other than `lt`
and `ge`, benchmark sizes other than 2,000, prompt counts other than 20, and
audio log counts other than four. Resolve YAML-relative output paths against
the config file directory while preserving the absolute proprietary input
paths.

Use same-directory temporary files, `flush()`, `os.fsync()`, and `os.replace()`
inside `atomic_json`. Canonicalize mappings with sorted JSON keys in
`fingerprint`.

Add this optional dependency group and regenerate the lock:

```toml
[project.optional-dependencies]
balalaika = [
    "accelerate>=1.10,<2",
    "wandb>=0.21,<1",
    "zstandard>=0.23,<1",
    "onnx-asr[gpu,hub]",
    "pyyaml>=6,<7",
]
```

Initialize `implementation-notes.md` with a dated task heading and the approved
Accelerate-first/native-DDP-fallback decision; do not copy secrets or tokens.

- [ ] **Step 4: Sync and run tests**

Run: `rtk uv sync --extra dev --extra balalaika`

Run: `rtk uv run pytest tests/training/balalaika/test_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the foundation**

```bash
rtk git add pyproject.toml uv.lock implementation-notes.md src/voxcpm/training/balalaika tests/training/balalaika/test_config.py
rtk git commit -m "feat: add balalaika training configuration"
```

### Task 2: Authenticated Hub pinning through `hf` CLI

**Files:**
- Create: `src/voxcpm/training/balalaika/hub.py`
- Create: `tests/training/balalaika/test_hub.py`
- Modify: `src/voxcpm/training/balalaika/config.py`

**Interfaces:**
- Consumes: `atomic_json`, `sha256_file`, `HubConfig`.
- Produces: `HubPin(kind, repo_id, revision, local_dir, files)`, a Pydantic model.
- Produces: `pin_hub_inputs(config: HubConfig, runner: CommandRunner) -> dict[str, HubPin]`.
- `CommandRunner.run(argv: Sequence[str]) -> CompletedProcess[str]` never accepts a shell string.

- [ ] **Step 1: Write failing tests with a fake argv runner**

```python
def test_pin_private_benchmark_uses_hf_cli_and_immutable_sha(tmp_path):
    runner = FakeRunner({
        ("hf", "datasets", "info", DATASET): json.dumps({"sha": "abc123"}),
        ("hf", "download", DATASET, "--repo-type", "dataset", "--revision", "abc123",
         "--local-dir", str(tmp_path / "benchmark")): str(tmp_path / "benchmark"),
    })
    pins = pin_hub_inputs(hub_config(tmp_path), runner)
    assert pins["benchmark"].revision == "abc123"
    assert all("--token" not in argv for argv in runner.calls)
```

Also test model pinning with `hf models info`, rejection of a missing `sha`, and
reuse only when an existing pin manifest has the same repo/revision/file
hashes.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_hub.py -q`

Expected: FAIL because `hub.py` is absent.

- [ ] **Step 3: Implement pin/download operations**

Call the authenticated CLI exactly as argv lists: `hf models info`,
`hf datasets info`, and `hf download --revision <sha> --local-dir <path>`.
Pin `OpenBMB/VoxCPM2`, `bitmanagerai/hard_number_eval_for_tts`, and the
configured GigaAM ONNX repository. Persist `hub-pins.json` atomically with the
resolved SHA and hashes of downloaded metadata/data files. Never read, print,
or pass `HF_TOKEN` explicitly.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_hub.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/hub.py src/voxcpm/training/balalaika/config.py tests/training/balalaika/test_hub.py
rtk git commit -m "feat: pin balalaika Hub inputs"
```

### Task 3: Strict corpus join and compact SQLite index

**Files:**
- Create: `src/voxcpm/training/balalaika/index.py`
- Create: `tests/training/balalaika/conftest.py`
- Create: `tests/training/balalaika/test_index.py`

**Interfaces:**
- Consumes: `DataConfig`, `atomic_json`, `sha256_file`, `fingerprint`.
- Produces: `build_index(config: DataConfig, expectations: BuildExpectations) -> IndexAudit`.
- Produces SQLite tables `samples`, `stage_ordinals`, and `metadata`.
- `samples.stage` is 1 for `<0.95`, 2 for `>=0.95`, and null for excluded rows.

- [ ] **Step 1: Build a synthetic two-shard fixture and failing join tests**

The fixture must create uncompressed tar files containing paired JSON/audio
members, a zstd-compressed tar containing per-shard ROVER JSONL, a combined
sidecar, and matching metadata hashes.

```python
def test_build_index_joins_by_identity_and_splits_exact_boundary(synthetic_corpus):
    audit = build_index(synthetic_corpus.config, synthetic_corpus.expectations)
    with sqlite3.connect(audit.index_path) as db:
        rows = db.execute(
            "SELECT source_relative_path, agreement, stage FROM samples ORDER BY source_relative_path"
        ).fetchall()
    assert rows == [
        ("000000/a.wav", 0.949999, 1),
        ("000000/b.wav", 0.95, 2),
        ("000001/c.wav", None, None),
    ]
    assert audit.excluded_null_agreement == 1

def test_build_index_rejects_missing_combined_text(synthetic_corpus):
    synthetic_corpus.remove_combined_row("000000/b.wav")
    with pytest.raises(IndexIntegrityError, match="missing combined text"):
        build_index(synthetic_corpus.config, synthetic_corpus.expectations)
```

Add separate tests for duplicate identities, missing audio/JSON pairs,
unexpected shard/row counts, malformed agreement, empty text, and incorrect
combined-sidecar SHA-256.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_index.py -q`

Expected: FAIL because the index builder is absent.

- [ ] **Step 3: Implement streaming passes and schema**

Use `zstandard.ZstdDecompressor().stream_reader()` with `tarfile.open(...,
mode="r|")` to scan the ROVER archive exactly once. Insert agreement rows in
batched SQLite transactions. Scan the combined sidecar in binary mode and store
line byte offsets/sizes instead of duplicating its 1.3 GB of text. Scan each
uncompressed source tar once, using `TarInfo.offset_data`/`size`; parse only JSON
members and record matching audio offsets.

Create the final join with SQL, verify one-to-one counts before dropping staging
tables, assign dense per-stage ordinals, run `PRAGMA integrity_check`, write the
audit JSON, fsync the database, and atomically rename both outputs. Configure
SQLite for bulk build (`journal_mode=WAL`, bounded cache, explicit transactions)
without weakening final integrity checks.

- [ ] **Step 4: Run index tests**

Run: `rtk uv run pytest tests/training/balalaika/test_index.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/index.py tests/training/balalaika/conftest.py tests/training/balalaika/test_index.py
rtk git commit -m "feat: index balalaika tar corpus"
```

### Task 4: Indexed tar dataset and fixed batches

**Files:**
- Create: `src/voxcpm/training/balalaika/dataset.py`
- Create: `tests/training/balalaika/test_dataset.py`
- Modify: `src/voxcpm/training/data.py`

**Interfaces:**
- Consumes: SQLite schema from Task 3.
- Produces: `IndexedBalalaikaDataset(index_path, sidecar_path, stage, tokenizer, sample_rate)`.
- Produces: `build_unsharded_dataloader(dataset, batch_size, workers, seed, world_size, accumulation) -> DataLoader`.
- Reuses a public `VoxCPMCollator` extracted from the current `HFVoxCPMDataset.collate_fn`.

- [ ] **Step 1: Write failing random-access and batching tests**

```python
def test_indexed_dataset_reads_audio_and_text_without_extraction(synthetic_index):
    ds = IndexedBalalaikaDataset(
        synthetic_index.path, synthetic_index.sidecar, stage=1,
        tokenizer=lambda text: [len(text)], sample_rate=16_000,
    )
    item = ds[0]
    assert item["text_ids"] == [len("тест один")]
    assert item["audio_sampling_rate"] == 16_000
    assert item["audio_array"].ndim == 1
    assert not list(synthetic_index.root.rglob("extracted-*"))

def test_unsharded_loader_has_no_distributed_sampler(dataset):
    loader = build_unsharded_dataloader(dataset, batch_size=2, workers=0, seed=13,
                                        world_size=8, accumulation=4)
    assert not isinstance(loader.sampler, DistributedSampler)
    assert loader.batch_sampler.drop_last is True
    assert len(loader.batch_sampler) % (8 * 4) == 0
```

Also test worker-process reconnection, bounded descriptor eviction, ordinal
bounds, corrupt byte ranges, resampling, and deterministic shuffle for the same
seed.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_dataset.py -q`

Expected: FAIL because `dataset.py` is absent.

- [ ] **Step 3: Implement direct reads and shared collator**

Use process-local read-only SQLite connections and an LRU of file descriptors.
Read bytes with `os.pread(fd, size, offset)` so concurrent workers never share a
seek position. Decode from `io.BytesIO` with torchaudio/torchcodec, downmix to
mono, resample to 16 kHz, and return the existing VoxCPM sample keys. Read and
parse the exact combined-sidecar line by byte range and verify its identity
before tokenizing.

Move the generic padding/collation code in `training/data.py` to
`VoxCPMCollator`; keep `HFVoxCPMDataset.collate_fn` as a compatibility alias so
existing tests and users remain valid. Wrap the seeded random sampler in a
fixed `TruncatedBatchSampler` whose global batch count is the largest multiple
of `world_size * accumulation`; report its dropped sample count.

- [ ] **Step 4: Run new and legacy data tests**

Run: `rtk uv run pytest tests/training/balalaika/test_dataset.py tests/test_validate.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/dataset.py src/voxcpm/training/data.py tests/training/balalaika/test_dataset.py
rtk git commit -m "feat: stream indexed balalaika samples"
```

### Task 5: Deterministic memorization and random prompt manifests

**Files:**
- Create: `src/voxcpm/training/balalaika/selection.py`
- Create: `tests/training/balalaika/test_selection.py`

**Interfaces:**
- Consumes: indexed samples and pinned benchmark JSONL.
- Produces: `create_selection_manifests(index_path, benchmark_path, output_dir, seed) -> SelectionBundle`.
- Produces JSON: `memorization.json`, `prompts.json`, `benchmark-prompts.json`, and `audio-log-ids.json`.

- [ ] **Step 1: Write failing deterministic-selection tests**

```python
def test_selections_are_fixed_and_sample_without_replacement(selection_fixture):
    first = create_selection_manifests(**selection_fixture.kwargs, seed=29)
    second = create_selection_manifests(**selection_fixture.kwargs, seed=29)
    assert first.model_dump() == second.model_dump()
    assert len({x.source_relative_path for x in first.memorization}) == 4
    assert all(x.agreement >= 0.95 for x in first.memorization)
    assert len({x.source_relative_path for x in first.prompts}) == 20
    assert len(first.benchmark_prompt_by_id) == 2_000
    assert set(first.benchmark_prompt_by_id.values()) <= {x.prompt_id for x in first.prompts}
```

Add tests that permit null speaker/single-speaker metadata, select prompts from
both agreement stages, select four fixed W&B IDs, fail clearly when fewer than
20 usable joined audio/text rows exist, and change selections/assignments under
a different seed.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_selection.py -q`

Expected: FAIL because `selection.py` is absent.

- [ ] **Step 3: Implement seeded selection and prompt extraction**

Stream sorted stage-2 identities through Algorithm R reservoir sampling with
`random.Random(seed)` and reservoir size four for memorization. Stream all
sorted joined rows with usable audio and combined text through a second
reservoir of size 20 using a separately derived seed for validation prompts.
This gives uniform sampling without holding millions of identities in memory.
Do not inspect or rank speaker, single-speaker, or quality fields and never use
SQLite `random()`. Extract only the selected clips to WAV files under the
selection artifact directory and hash them. Store matching
`rover_punctuated_accented` prompt text. Assign all sorted benchmark IDs to the
20 prompts with the same local RNG and atomically publish fingerprinted JSON.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_selection.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/selection.py tests/training/balalaika/test_selection.py
rtk git commit -m "feat: select fixed balalaika prompts"
```

### Task 6: Golden hard-number metrics

**Files:**
- Create: `src/voxcpm/training/balalaika/metrics.py`
- Create: `tests/training/balalaika/test_metrics.py`

**Interfaces:**
- Produces: `normalize_ru(text: str) -> str`.
- Produces: `score_utterance(row: BenchmarkRow, hypothesis: str) -> ItemScore`.
- Produces: `aggregate_scores(scores: Sequence[ItemScore]) -> AggregateScore`.
- `AggregateScore` exposes `num_cer`, `num_wer`, `utt_cer`, `utt_wer`, edit counts, and category metrics.

- [ ] **Step 1: Write failing normalization, span, and denominator tests**

```python
def test_micro_cer_uses_character_denominator_not_word_count():
    rows = [
        BenchmarkRow(id=1, category="agree", text="Долг 12 рублей.",
                     normalized_gold="Долг двенадцать рублей.", stressed=""),
        BenchmarkRow(id=2, category="agree", text="Долг 3 рубля.",
                     normalized_gold="Долг три рубля.", stressed=""),
    ]
    scores = [score_utterance(rows[0], "долг двенатцать рублей"),
              score_utterance(rows[1], "долг три рубля")]
    aggregate = aggregate_scores(scores)
    assert aggregate.utt_cer == pytest.approx(1 / 32)
    assert aggregate.utt_wer == pytest.approx(1 / 6)

def test_empty_hypothesis_is_scored_not_dropped(benchmark_row):
    score = score_utterance(benchmark_row, "")
    assert score.utt_deletions == len(normalize_ru(benchmark_row.normalized_gold).split())
    assert score.utt_wer == 1.0
```

Add golden cases for `ё`/`е`, stress and punctuation removal, repeated digits,
decimal kopecks, dates, insertions around the number span, empty hypotheses,
and per-category aggregation. Calculate literal expected denominators in the
test rather than copying implementation output.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_metrics.py -q`

Expected: FAIL because `metrics.py` is absent.

- [ ] **Step 3: Implement alignment and SDI aggregation**

Port the benchmark's alignment semantics but return explicit substitution,
deletion, insertion, correct, and reference-unit counts for both characters and
words. Derive the number span by aligning digit-bearing `text` to
`normalized_gold`, then map that gold interval into the hypothesis. Compute all
four corpus metrics from summed error counts divided by their own summed
reference counts.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_metrics.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/metrics.py tests/training/balalaika/test_metrics.py
rtk git commit -m "feat: score hard-number TTS output"
```

### Task 7: GigaAM adapter and resumable validation ledger

**Files:**
- Create: `src/voxcpm/training/balalaika/asr.py`
- Create: `src/voxcpm/training/balalaika/ledger.py`
- Create: `tests/training/balalaika/test_asr.py`
- Create: `tests/training/balalaika/test_ledger.py`

**Interfaces:**
- Produces: `GigaAMRNNT(model_dir: Path, device_id: int, model_fingerprint: str)`.
- Produces: `GigaAMRNNT.transcribe(wav_path: Path) -> str`.
- Produces: `ValidationLedger.claim(id)`, `record_generation`, `record_asr`, `record_failure`, and `complete_ids`.

- [ ] **Step 1: Write failing adapter and ledger tests**

```python
def test_gigaam_preserves_successful_empty_hypothesis(fake_onnx_asr, wav_path):
    fake_onnx_asr.result = ""
    model = GigaAMRNNT(Path("model"), device_id=3, model_fingerprint="sha")
    assert model.transcribe(wav_path) == ""
    assert fake_onnx_asr.providers[0][0] == "CUDAExecutionProvider"
    assert fake_onnx_asr.providers[0][1]["device_id"] == 3

def test_ledger_reuses_only_hash_matching_complete_item(tmp_path):
    ledger = ValidationLedger(tmp_path, generation_fingerprint="g1", asr_fingerprint="a1")
    ledger.record_generation(7, wav_sha256="w1", path="00007.wav")
    ledger.record_asr(7, hypothesis="текст")
    assert ledger.is_complete(7)
    assert not ValidationLedger(tmp_path, "g2", "a1").is_complete(7)
```

Also test atomic per-item JSON, bounded retry counts, exception records, missing
WAV files, hash mismatch, and multi-rank filenames that cannot collide.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_asr.py tests/training/balalaika/test_ledger.py -q`

Expected: FAIL because both modules are absent.

- [ ] **Step 3: Implement the pinned ONNX adapter and ledger**

Load the pre-pinned local `gigaam-v3-rnnt` files with `onnx_asr.load_model`,
explicit CUDA provider/device ID, and CPU fallback provider only for unsupported
operators. Resample input to mono float32 16 kHz before `recognize`. Normalize
the library's list/string return type but do not turn an empty successful result
into an exception.

Store one atomic JSON record per benchmark ID with all input fingerprints,
attempt seeds, waveform hash/path, ASR result, timings, and errors. A completed
record is reusable only if every fingerprint/hash still matches.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_asr.py tests/training/balalaika/test_ledger.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/asr.py src/voxcpm/training/balalaika/ledger.py tests/training/balalaika/test_asr.py tests/training/balalaika/test_ledger.py
rtk git commit -m "feat: add resumable GigaAM evaluation"
```

### Task 8: Rank-zero W&B tracking

**Files:**
- Create: `src/voxcpm/training/balalaika/tracking.py`
- Create: `tests/training/balalaika/test_tracking.py`

**Interfaces:**
- Produces: `WandbRunManager.start(job_type, run_state_path, config) -> WandbRunManager`.
- Produces: `log_train(metrics, global_step)` and `log_validation(result, audio_paths, global_step)`.
- Produces: `ValidationPayload(metrics, category_metrics, items, audio_paths, artifact_dir)` for the evaluator handoff.
- Non-main processes receive `NullRunManager` and never import/init a W&B run.

- [ ] **Step 1: Write failing rank and payload tests**

```python
def test_validation_logs_required_metrics_table_and_four_audio(fake_wandb, validation_result):
    manager = WandbRunManager.start("stage1", run_state_path=validation_result.run_state,
                                    config=wandb_config(), wandb_module=fake_wandb)
    manager.log_validation(validation_result, validation_result.audio_paths[:4], global_step=10)
    payload = fake_wandb.run.logged[-1]
    assert {"val/num_cer", "val/num_wer", "val/utt_cer", "val/utt_wer"} <= payload.keys()
    assert len(payload["val/examples"]) == 4
    assert len(payload["val/items"].data) == 2_000

def test_non_main_process_never_initializes_wandb(fake_wandb):
    manager = create_run_manager(is_main_process=False, config=wandb_config())
    assert isinstance(manager, NullRunManager)
    assert fake_wandb.init_calls == []
```

Also test stable run-ID persistence before `wandb.init`, `resume="allow"`, shared
group with distinct job types, offline mode, fixed audio captions, and W&B
failure leaving the validation boundary incomplete.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_tracking.py -q`

Expected: FAIL because `tracking.py` is absent.

- [ ] **Step 3: Implement W&B lifecycle and immutable per-boundary tables**

Generate/persist the run ID atomically before initialization. Call
`wandb.init(project, entity, id, resume="allow", group, job_type, mode, dir,
config)`. Log only on rank zero. Build a new immutable `wandb.Table` for every
validation boundary and four `wandb.Audio(path, caption=...)` values for the
fixed IDs. Mark the local upload manifest complete only after `run.log`
returns and the W&B run directory is durably present.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_tracking.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/tracking.py tests/training/balalaika/test_tracking.py
rtk git commit -m "feat: track balalaika runs in wandb"
```

### Task 9: Epoch geometry and LoRA-only Accelerate checkpoints

**Files:**
- Create: `src/voxcpm/training/balalaika/schedule.py`
- Create: `src/voxcpm/training/balalaika/checkpoint.py`
- Create: `tests/training/balalaika/test_schedule.py`
- Create: `tests/training/balalaika/test_checkpoint.py`

**Interfaces:**
- Produces: `EpochGeometry.from_counts(dataset_rows, world_size, microbatch, accumulation)`.
- Produces: `EpochGeometry.validation_steps(epoch) -> tuple[int, ...]` with eight unique optimizer steps.
- Produces: `TrainingProgress.state_dict/load_state_dict`.
- Produces: `CheckpointManager.save_same_stage`, `resume_same_stage`, and `load_stage_adapter`.

- [ ] **Step 1: Write failing boundary and mismatch tests**

```python
def test_fractional_boundaries_are_eight_unique_and_end_at_epoch():
    geometry = EpochGeometry.from_counts(1_003_000, world_size=8, microbatch=2, accumulation=4)
    steps = geometry.validation_steps(epoch=0)
    assert len(steps) == len(set(steps)) == 8
    assert steps[-1] == geometry.optimizer_steps_per_epoch
    assert 0 not in steps

def test_stage2_load_rejects_optimizer_state(dummy_checkpoint, model):
    manager = CheckpointManager(dummy_checkpoint.root)
    manager.load_stage_adapter(model, dummy_checkpoint.path, expected=stage2_fingerprint())
    assert model.loaded_lora
    assert not model.loaded_optimizer

def test_resume_rejects_world_size_change(dummy_checkpoint):
    with pytest.raises(CheckpointMismatch, match="world_size"):
        CheckpointManager(dummy_checkpoint.root).verify(dummy_checkpoint.path, {"world_size": 4})
```

Also test dropped batch and accumulation remainders, no duplicate epoch-end
boundary, LoRA-only safetensors keys, atomic checkpoint rename, latest pointer,
RNG/progress restoration, and mismatches in base/data/selection/LoRA/batch
fingerprints.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_schedule.py tests/training/balalaika/test_checkpoint.py -q`

Expected: FAIL because the modules are absent.

- [ ] **Step 3: Implement exact integer scheduling and Accelerate hooks**

Use `ceil(k * optimizer_steps_per_epoch / 8)` for `k=1..8` and reject stage
geometries too small to create eight unique optimizer boundaries. Drop
microbatches beyond the largest complete accumulation group and record their
sample count.

Register `TrainingProgress` for checkpointing. Install Accelerate save/load
pre-hooks that write only keys containing `lora_` to safetensors and suppress
the full base-model state while still allowing `accelerator.save_state()` to
save optimizer, scheduler, RNG, and registered progress. Save into a sibling
temporary directory, synchronize ranks, fsync on rank zero, atomically rename,
then atomically update `latest.json`. Implement adapter-only stage transition
as a separate path that never calls `accelerator.load_state()`.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_schedule.py tests/training/balalaika/test_checkpoint.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/schedule.py src/voxcpm/training/balalaika/checkpoint.py tests/training/balalaika/test_schedule.py tests/training/balalaika/test_checkpoint.py
rtk git commit -m "feat: checkpoint fractional balalaika training"
```

### Task 10: Accelerate runtime and synchronized microbatch probe

**Files:**
- Create: `src/voxcpm/training/balalaika/runtime.py`
- Create: `src/voxcpm/training/balalaika/probe.py`
- Create: `scripts/smoke_balalaika_accelerate.py`
- Create: `tests/training/balalaika/test_runtime.py`
- Create: `tests/training/balalaika/test_probe.py`
- Modify: `implementation-notes.md`

**Interfaces:**
- Produces protocol `TrainingRuntime` with rank/device/barrier/gather/unwrap/prepare/save/load operations.
- Produces: `AccelerateRuntime.create(config) -> AccelerateRuntime`.
- Produces: `probe_microbatch(runtime, candidates, sample_factory, step_fn) -> ProbeResult`.

- [ ] **Step 1: Write failing CPU runtime/probe tests**

```python
def test_accelerate_runtime_configures_bf16_and_unsplit_batches(monkeypatch):
    runtime = AccelerateRuntime.create(runtime_config(accumulation=4), accelerator_cls=FakeAccelerator)
    assert runtime.accelerator.kwargs["mixed_precision"] == "bf16"
    assert runtime.accelerator.kwargs["gradient_accumulation_steps"] == 4
    assert runtime.accelerator.kwargs["dataloader_config"].split_batches is False
    assert runtime.accelerator.kwargs["dataloader_config"].even_batches is False

def test_probe_uses_minimum_success_across_ranks(fake_runtime):
    fake_runtime.rank_results = {1: True, 2: True, 4: False}
    result = probe_microbatch(fake_runtime, [1, 2, 4], sample_factory(), fake_step)
    assert result.microbatch == 2
    assert result.all_rank_success is True
```

Also test OOM cleanup, restoration of original LoRA weights/RNG, explicit
microbatch bypass, and that prepared dataloaders are not pre-wrapped in a
`DistributedSampler`.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_runtime.py tests/training/balalaika/test_probe.py -q`

Expected: FAIL because runtime modules are absent.

- [ ] **Step 3: Implement primary Accelerate runtime and safe probe**

Construct `Accelerator(mixed_precision="bf16",
gradient_accumulation_steps=..., dataloader_config=DataLoaderConfiguration(
split_batches=False, even_batches=False, use_seedable_sampler=True),
kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False,
broadcast_buffers=False)])`. Wrap public APIs only: `prepare`, `accumulate`,
`backward`, `clip_grad_norm_`, `gather_for_metrics`, `wait_for_everyone`,
`unwrap_model`, `save_state`, and `load_state`.

The probe snapshots LoRA tensors and RNG, tries candidates in ascending order,
all-reduces success, catches CUDA OOM only, restores state after each candidate,
and chooses the largest all-rank success. Any non-OOM exception is terminal.

- [ ] **Step 4: Run the required Accelerate spike**

Run CPU/unit tests first:

`rtk uv run pytest tests/training/balalaika/test_runtime.py tests/training/balalaika/test_probe.py -q`

Then run one GPU:

`rtk accelerate launch --num_processes 1 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint`

Then run eight GPUs:

`rtk accelerate launch --num_processes 8 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint`

Expected: each rank reports disjoint sample IDs; accumulated LoRA gradients and
optimizer steps match the reference; checkpoint restore reproduces parameters,
optimizer step, and next sample position.

Record commands, package/CUDA versions, results, and peak memory in
`implementation-notes.md`.

- [ ] **Step 5: Apply the controlled fallback rule if required**

If the eight-GPU Accelerate spike passes, record `Decision: Accelerate primary
path passed; native DDP not implemented` and continue.

If it fails, preserve the failing log and first implement up to two minimal
fixes confined to `runtime.py`/`probe.py`, rerunning the same command after each.
Only if the same model-behavior, synchronization, or state-restore failure
persists, add `NativeDDPRuntime` implementing the exact `TrainingRuntime`
protocol with NCCL, `DistributedDataParallel`, `DistributedSampler`,
`no_sync`, BF16 autocast, and the same checkpoint metadata. Add parametrized
runtime tests and rerun both one- and eight-GPU smoke commands with
`--runtime native-ddp`. Record the reproduction and fallback decision.

- [ ] **Step 6: Commit the passing runtime**

```bash
rtk git add src/voxcpm/training/balalaika/runtime.py src/voxcpm/training/balalaika/probe.py scripts/smoke_balalaika_accelerate.py tests/training/balalaika/test_runtime.py tests/training/balalaika/test_probe.py implementation-notes.md
rtk git commit -m "feat: run balalaika training with accelerate"
```

### Task 11: Distributed 2,000-item evaluator

**Files:**
- Create: `src/voxcpm/training/balalaika/evaluation.py`
- Create: `tests/training/balalaika/test_evaluation.py`
- Modify: `scripts/smoke_balalaika_accelerate.py`

**Interfaces:**
- Consumes: `TrainingRuntime`, `SelectionBundle`, `ValidationLedger`, `GigaAMRNNT`, metric interfaces, and `ValidationPayload` from Task 8.
- Produces: `DistributedEvaluator.run(model, audio_vae, checkpoint, boundary) -> ValidationPayload`.

- [ ] **Step 1: Write failing distributed partition and completion tests**

```python
def test_partition_is_disjoint_and_complete():
    partitions = [set(partition_ids(range(1, 2001), rank=r, world_size=8)) for r in range(8)]
    assert set.union(*partitions) == set(range(1, 2001))
    assert sum(map(len, partitions)) == 2_000

def test_evaluator_requires_exactly_2000_records(evaluator_fixture):
    evaluator_fixture.ledger.remove(2000)
    with pytest.raises(IncompleteValidation, match="1999/2000"):
        evaluator_fixture.evaluator.aggregate()

def test_validation_resume_does_not_regenerate_completed_audio(evaluator_fixture):
    evaluator_fixture.ledger.complete(1)
    evaluator_fixture.evaluator.run_rank(rank=0, world_size=1)
    assert 1 not in evaluator_fixture.generator.calls
```

Also test deterministic per-item seeds, prompt audio/text pairing, bounded
retries, audio hash checks, empty ASR scoring, four fixed audio IDs, model
train/eval restoration, W&B completion ordering, and retention removing older
full WAV directories only after a newer validation is durably complete.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_evaluation.py -q`

Expected: FAIL because `evaluation.py` is absent.

- [ ] **Step 3: Implement generation, ASR, and aggregation phases**

Partition sorted benchmark IDs by `position % world_size`. Temporarily attach
the retained AudioVAE to the unwrapped VoxCPM2 model, call `generate` with
`target_text=row.stressed`, the assigned `prompt_text`/`prompt_wav_path`, and a
seed derived from validation and item fingerprints, then detach the VAE and
restore training mode in `finally`.

Write WAV and ledger records atomically. After generation, load GigaAM on each
rank, transcribe that rank's completed WAVs, and release its session before
training resumes. Barrier, then let rank zero verify 2,000 unique complete
records, score them, write `metrics.json`/`items.jsonl`, log W&B, and publish
`validation-complete.json`. Every rank checks that completion record before
returning. Apply the configured retention policy only after publication: keep
the latest full validation directory and copy/retain the four fixed logged WAVs
from every completed boundary.

- [ ] **Step 4: Run unit and distributed smoke tests**

Run: `rtk uv run pytest tests/training/balalaika/test_evaluation.py -q`

Run: `rtk accelerate launch --num_processes 8 scripts/smoke_balalaika_accelerate.py --mode validation --items 32`

Expected: PASS; 32 unique IDs, four per rank, one aggregate file, and no
non-main W&B initialization.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/evaluation.py tests/training/balalaika/test_evaluation.py scripts/smoke_balalaika_accelerate.py
rtk git commit -m "feat: evaluate balalaika training across GPUs"
```

### Task 12: Two-stage LoRA trainer

**Files:**
- Create: `src/voxcpm/training/balalaika/trainer.py`
- Create: `tests/training/balalaika/test_trainer.py`
- Modify: `src/voxcpm/training/balalaika/__init__.py`

**Interfaces:**
- Consumes all preceding data/runtime/checkpoint/evaluation interfaces.
- Produces: `BalalaikaTrainer.run_stage(stage: Literal[1, 2]) -> Path`.
- Produces: `build_model(config, stage, adapter_checkpoint=None) -> (model, audio_vae, tokenizer)`.

- [ ] **Step 1: Write failing stage and accumulation tests**

```python
def test_stage2_loads_adapter_but_resets_optimizer_and_scheduler(trainer_fixture):
    stage1 = trainer_fixture.stage1_checkpoint(optimizer_step=123)
    trainer = trainer_fixture.make_stage2(stage1)
    trainer.initialize()
    assert trainer.model.lora_digest() == stage1.lora_digest
    assert trainer.optimizer.state == {}
    assert trainer.scheduler.last_epoch in {-1, 0}

def test_validation_fires_once_at_each_boundary(trainer_fixture):
    trainer = trainer_fixture.make(optimizer_steps_per_epoch=80, epochs=2)
    trainer.run()
    assert trainer.evaluator.steps == [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160]

def test_only_lora_parameters_are_trainable(model):
    assert all(("lora_" in name) == parameter.requires_grad for name, parameter in model.named_parameters())
```

Also test no validation at step zero, no incomplete accumulation update,
scheduler steps only with optimizer steps, gradient clipping only when
`runtime.sync_gradients`, same-stage resume before another optimizer step, and
stage-1 approval enforcement.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_trainer.py -q`

Expected: FAIL because `trainer.py` is absent.

- [ ] **Step 3: Implement model setup and Accelerate training loop**

Load the pinned local VoxCPM2 revision with repository LoRA classes, retain the
AudioVAE for `BatchProcessor`/generation, and remove it from the wrapped train
model as the current trainer does. Build AdamW over `requires_grad` parameters
only and a stage-sized cosine/warmup scheduler. Pass model, optimizer,
unsharded dataloader, and scheduler to one `runtime.prepare` call.

For each microbatch use `with runtime.accumulate(model)`, compute weighted
losses, call `runtime.backward`, clip only at synchronized boundaries, then
optimizer/scheduler/zero-grad. Advance `TrainingProgress` only after a real
optimizer step. At each exact boundary checkpoint first, run/resume validation,
then atomically mark the boundary complete. On signal or exception save a
same-stage checkpoint after synchronizing all ranks.

- [ ] **Step 4: Run trainer tests and synthetic integration**

Run: `rtk uv run pytest tests/training/balalaika/test_trainer.py tests/training/balalaika/test_checkpoint.py tests/training/balalaika/test_evaluation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/trainer.py src/voxcpm/training/balalaika/__init__.py tests/training/balalaika/test_trainer.py
rtk git commit -m "feat: train balalaika LoRA curriculum"
```

### Task 13: Four-example memorization and approval gate

**Files:**
- Create: `src/voxcpm/training/balalaika/memorization.py`
- Create: `tests/training/balalaika/test_memorization.py`

**Interfaces:**
- Produces: `run_memorization(config, runtime) -> MemorizationResult`.
- Produces: `approve_memorization(result_dir, wandb_run_id, approver) -> Path`.
- Produces: `verify_approval(path, expected_fingerprints) -> ApprovalRecord`.

- [ ] **Step 1: Write failing repetition, stop, and approval tests**

```python
def test_memorization_repeats_only_four_selected_items(mem_fixture):
    result = run_memorization(mem_fixture.config, mem_fixture.runtime)
    assert set(result.seen_sample_ids) == set(mem_fixture.selected_ids)
    assert len(result.reference_audio) == len(result.generated_audio) == 4
    assert result.large_training_started is False

def test_approval_is_bound_to_checkpoint_and_wandb_run(mem_result):
    path = approve_memorization(mem_result.dir, "wandb123", "operator")
    record = verify_approval(path, mem_result.fingerprints)
    assert record.wandb_run_id == "wandb123"
    tampered = {**mem_result.fingerprints, "checkpoint": "different"}
    with pytest.raises(ApprovalMismatch, match="checkpoint"):
        verify_approval(path, tampered)
```

Also test that four distinct sample identities are required, all
reference/generated pairs are logged, GigaAM diagnostics are non-gating, missing W&B completion
cannot be approved, and stage 1 refuses an approval from another index/base/
LoRA configuration.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_memorization.py -q`

Expected: FAIL because `memorization.py` is absent.

- [ ] **Step 3: Implement repeated four-item training and explicit approval**

Wrap exactly the selected four samples in a deterministic repeat dataset, run
the configured number of updates through the same runtime/model/loss/checkpoint
path as stage training, generate each training text without a conditioning prompt,
transcribe it, and log reference/generated pairs plus loss and text diagnostics
to a `memorization` W&B run. Stop after writing `memorization-result.json`.

The separate approval function requires the completed result, W&B run ID, and
checkpoint files, hashes all referenced artifacts, and atomically writes
`memorization-approval.json`. It never launches training.

- [ ] **Step 4: Run tests**

Run: `rtk uv run pytest tests/training/balalaika/test_memorization.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/voxcpm/training/balalaika/memorization.py tests/training/balalaika/test_memorization.py
rtk git commit -m "feat: gate training with memorization run"
```

### Task 14: CLI, complete config, runbook, and end-to-end verification

**Files:**
- Create: `src/voxcpm/training/balalaika/cli.py`
- Create: `conf/voxcpm_v2/balalaika_lora.yaml`
- Create: `docs/balalaika_training.md`
- Create: `tests/training/balalaika/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `implementation-notes.md`

**Interfaces:**
- Adds console script: `voxcpm-balalaika = "voxcpm.training.balalaika.cli:main"`.
- Commands: `pin`, `prepare`, `memorize`, `approve`, `train --stage {1,2}`, `validate`, and `audit`.

- [ ] **Step 1: Write failing CLI routing and safety tests**

```python
def test_train_stage1_requires_matching_approval(cli_runner, config_path):
    result = cli_runner.invoke(["--config", str(config_path), "train", "--stage", "1"])
    assert result.exit_code != 0
    assert "memorization approval" in result.stderr.lower()

def test_prepare_does_not_mutate_corpus(cli_runner, synthetic_corpus, config_path):
    before = synthetic_corpus.hashes()
    result = cli_runner.invoke(["--config", str(config_path), "prepare"])
    assert result.exit_code == 0
    assert synthetic_corpus.hashes() == before
```

Add routing tests for all commands, `train --stage 2` requiring a completed
stage-1 adapter, `validate` requiring a checkpoint, non-eight process rejection
for large training unless `--smoke`, and audit output with the 309 expected
null rows in production configuration.

- [ ] **Step 2: Run tests and verify failure**

Run: `rtk uv run pytest tests/training/balalaika/test_cli.py -q`

Expected: FAIL because CLI/config files are absent.

- [ ] **Step 3: Implement CLI and checked-in configuration**

Use `argparse` subparsers and call the typed module interfaces; keep CLI code
free of training logic. Fill the YAML with the approved absolute corpus paths,
Hub IDs, stage defaults, LoRA defaults, seed, output roots, batch candidates,
generation settings, GigaAM model, and configurable W&B entity/project/mode.
Do not embed credentials.

- [ ] **Step 4: Write the operator runbook**

Document exact commands:

```bash
rtk uv sync --extra dev --extra balalaika
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml pin
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml prepare
rtk accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml memorize
rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml approve --wandb-run-id RUN_ID
rtk accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml train --stage 1
rtk accelerate launch --num_processes 8 -m voxcpm.training.balalaika.cli --config conf/voxcpm_v2/balalaika_lora.yaml train --stage 2
```

Include preflight/audit interpretation, W&B login/offline sync, expected 16 and
24 validation boundaries, checkpoint resume, interrupted validation recovery,
disk retention, signal handling, and explicit confirmation that no large run is
started by implementation verification.

- [ ] **Step 5: Run the complete CPU suite and static checks**

Run: `rtk uv run pytest -q`

Run: `rtk uv run python -m compileall -q src scripts`

Run: `rtk git diff --check`

Expected: all pass.

- [ ] **Step 6: Run real-data preflight without launching training**

Run: `rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml pin`

Run: `rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml prepare`

Run: `rtk uv run voxcpm-balalaika --config conf/voxcpm_v2/balalaika_lora.yaml audit`

Expected: 519 shards, 4,075,032 joined rows, exactly 309 null-agreement
exclusions, no identity/hash errors, four unique high-agreement memorization
samples, 20 unique randomly sampled corpus prompts, and 2,000 fixed benchmark assignments. Record
runtime, index size, stage counts, and audit fingerprint in
`implementation-notes.md`.

- [ ] **Step 7: Run final GPU smoke checks, not memorization or large training**

Run: `rtk accelerate launch --num_processes 1 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint`

Run: `rtk accelerate launch --num_processes 8 scripts/smoke_balalaika_accelerate.py --mode train-checkpoint`

Run: `rtk accelerate launch --num_processes 8 scripts/smoke_balalaika_accelerate.py --mode validation --items 32`

Expected: all pass with synchronized steps, unique samples, successful restore,
and one 32-item aggregate. Do not run the real memorization command until the
operator is ready to inspect W&B, and do not create an approval record on their
behalf.

- [ ] **Step 8: Commit the completed pipeline**

```bash
rtk git add pyproject.toml uv.lock README.md conf/voxcpm_v2/balalaika_lora.yaml docs/balalaika_training.md src/voxcpm/training/balalaika tests/training/balalaika implementation-notes.md
rtk git commit -m "feat: add balalaika LoRA training pipeline"
```

## Implementation References

- Approved design: `docs/superpowers/specs/2026-08-02-balalaika-two-stage-lora-training-design.md`
- Accelerate migration, accumulation, dataloader preparation, and checkpointing:
  `https://huggingface.co/docs/accelerate/en/basic_tutorials/migration`
- Accelerate checkpointing: `https://huggingface.co/docs/accelerate/usage_guides/checkpoint`
- Accelerate dataloader wrappers: `https://huggingface.co/docs/accelerate/main/package_reference/torch_wrappers`
- W&B run resume: `https://docs.wandb.ai/models/runs/resuming`
- W&B audio: `https://docs.wandb.ai/ref/python/data-types/audio/`
- W&B tables: `https://docs.wandb.ai/models/ref/python/data-types/table`
- GigaAM model owner: `https://github.com/salute-developers/GigaAM`
- onnx-asr GPU installation: `https://istupakov.github.io/onnx-asr/installation/`
