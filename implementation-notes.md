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
