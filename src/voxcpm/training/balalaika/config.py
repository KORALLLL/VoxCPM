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


class HubConfig(_ConfigModel):
    model_repo_id: str = "OpenBMB/VoxCPM2"
    benchmark_repo_id: str = "bitmanagerai/hard_number_eval_for_tts"
    gigaam_repo_id: str | None = None
    local_dir: Path = Path("hub")


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


class ValidationConfig(_ConfigModel):
    benchmark_size: Literal[2000] = 2000
    prompt_count: Literal[20] = 20
    audio_log_count: Literal[4] = 4


class WandbConfig(_ConfigModel):
    project: str = "voxcpm-balalaika"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled"] = "disabled"


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
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

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
        return config.model_copy(
            update={
                "output_dir": output_dir,
                "hub": config.hub.model_copy(update={"local_dir": local_dir}),
                "data": config.data.model_copy(update={"index_dir": index_dir}),
            }
        )


def _resolve_relative(path: Path, base_dir: Path) -> Path:
    return path if path.is_absolute() else base_dir / path
