## 2026-08-02 - Balalaika training foundation

- Decision: Use Accelerate as the primary distributed-training interface, with a native DDP fallback retained for environments where Accelerate is unavailable or unsuitable.
- Scope: This foundation validates the two-stage curriculum contract and keeps only the configuration fields needed by the initial pipeline boundary.
- Validation: `rtk uv sync --extra dev --extra balalaika`, the focused Balalaika tests, formatting checks, and the full `tests/` suite completed successfully.

## 2026-08-02 - GigaAM adapter and validation ledger

- Decision: Load only the named `gigaam-v3-rnnt` model from its already pinned local directory, and verify CUDA availability before opening a session; the loader is never given a remote repository identifier.
- Decision: Store each benchmark ID as an independently locked, atomically replaced JSON record so distributed ranks can safely own disjoint IDs and a crash cannot publish a partial record.
- Assumption: Generation and ASR fingerprints identify all deterministic inputs supplied by the later evaluator; WAV content hashes are recomputed before reuse.
- Validation: RED collection failed while the modules were absent; focused adapter/ledger tests are green (16 passed), `py_compile`, Black, and `git diff --check` pass; the full `tests/` suite passes (134 tests).
