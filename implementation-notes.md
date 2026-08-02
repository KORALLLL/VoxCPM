## 2026-08-02 - Balalaika training foundation

- Decision: Use Accelerate as the primary distributed-training interface, with a native DDP fallback retained for environments where Accelerate is unavailable or unsuitable.
- Scope: This foundation validates the two-stage curriculum contract and keeps only the configuration fields needed by the initial pipeline boundary.
- Validation: `rtk uv sync --extra dev --extra balalaika`, the focused Balalaika tests, formatting checks, and the full `tests/` suite completed successfully.
