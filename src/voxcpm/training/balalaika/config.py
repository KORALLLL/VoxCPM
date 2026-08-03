"""Validated configuration for the two-stage Balalaika LoRA curriculum."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_ConfigModel):
    corpus_root: Path
    index_dir: Path = Path("artifacts/balalaika-index")
    verification_manifest: Path | None = None
    combined_metadata: Path | None = None
    expected_shards: int = Field(default=519, gt=0)
    expected_rows: int = Field(default=4_075_032, gt=0)
    expected_null_agreement: int = Field(default=309, ge=0)
    expected_stage1_rows: int = Field(default=2_486_821, ge=0)
    expected_stage2_rows: int = Field(default=1_587_902, ge=0)


class HubConfig(_ConfigModel):
    model_repo_id: str = "OpenBMB/VoxCPM2"
    benchmark_repo_id: str = "bitmanagerai/hard_number_eval_for_tts"
    gigaam_repo_id: str | None = None
    local_dir: Path = Path("hub")
    benchmark_file: Path = Path("data.jsonl")


class LoRAConfig(_ConfigModel):
    enable_lm: bool = True
    enable_dit: bool = True
    enable_proj: bool = False
    r: int = Field(default=32, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


class StageConfig(_ConfigModel):
    agreement: Literal["lt", "ge"]
    threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    batch_size: int = Field(gt=0)


class RuntimeConfig(_ConfigModel):
    accumulation: int = Field(default=4, gt=0)
    workers: int = Field(default=4, ge=0)
    seed: int = 20_260_802
    batch_candidates: list[int] = Field(default_factory=lambda: [1, 2, 4], min_length=1)
    fixed_microbatch: int | None = Field(default=None, gt=0)
    cpu: bool = False


class MemorizationConfig(_ConfigModel):
    updates: int = Field(default=256, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0.0)
    batch_size: int = Field(default=1, gt=0)
    accumulation: int = Field(default=4, gt=0)
    seed: int = 20_260_802
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    loss_weights: dict[str, float] = Field(default_factory=lambda: {"loss/diff": 1.0, "loss/stop": 1.0})


class GenerationConfig(_ConfigModel):
    cfg_value: float = 2.0
    inference_timesteps: int = Field(default=10, gt=0)
    max_length: int = Field(default=2_048, gt=0)
    retries: int = Field(default=3, gt=0)
    seed: int = 20_260_802


class OptimizationConfig(_ConfigModel):
    scheduler: Literal["cosine"] = "cosine"
    warmup_fraction: float = Field(default=0.03, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    loss_weights: dict[str, float] = Field(default_factory=lambda: {"loss/diff": 1.0, "loss/stop": 1.0})


class ValidationConfig(_ConfigModel):
    benchmark_size: Literal[2000] = 2000
    prompt_count: Literal[20] = 20
    audio_log_count: Literal[4] = 4
    retries: int = Field(default=3, gt=0)
    keep_full_boundaries: int = Field(default=1, gt=0)


class WandbConfig(_ConfigModel):
    project: str = "voxcpm-balalaika"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "disabled"
    group: str | None = None
    dir: Path | None = None


class BalalaikaConfig(_ConfigModel):
    data: DataConfig
    output_dir: Path
    hub: HubConfig = Field(default_factory=HubConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    stage1: StageConfig = Field(
        default_factory=lambda: StageConfig(agreement="lt", threshold=0.95, epochs=2, learning_rate=1e-4, batch_size=1)
    )
    stage2: StageConfig = Field(
        default_factory=lambda: StageConfig(agreement="ge", threshold=0.95, epochs=3, learning_rate=5e-5, batch_size=1)
    )
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    memorization: MemorizationConfig = Field(default_factory=MemorizationConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    selection_dir: Path | None = None

    @classmethod
    def load(cls, path: Path) -> "BalalaikaConfig":
        """Load YAML and resolve output/cache paths relative to its directory."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration file {path} must contain a top-level mapping.")

        config = cls.model_validate(value)
        base_dir = path.parent.resolve()
        output_dir = _resolve_relative(config.output_dir, base_dir)
        local_dir = _resolve_relative(config.hub.local_dir, base_dir)
        index_dir = _resolve_relative(config.data.index_dir, base_dir)
        verification_manifest = _resolve_optional(config.data.verification_manifest, base_dir)
        combined_metadata = _resolve_optional(config.data.combined_metadata, base_dir)
        selection_dir = _resolve_relative(config.selection_dir or output_dir / "selection", base_dir)
        wandb_dir = _resolve_optional(config.wandb.dir, base_dir)
        return config.model_copy(
            update={
                "output_dir": output_dir,
                "hub": config.hub.model_copy(update={"local_dir": local_dir}),
                "data": config.data.model_copy(
                    update={
                        "index_dir": index_dir,
                        "verification_manifest": verification_manifest,
                        "combined_metadata": combined_metadata,
                    }
                ),
                "selection_dir": selection_dir,
                "wandb": config.wandb.model_copy(update={"dir": wandb_dir}),
            }
        )


def _resolve_relative(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path


def _resolve_optional(path: Path | None, base_dir: Path) -> Path | None:
    return None if path is None else _resolve_relative(path, base_dir)
