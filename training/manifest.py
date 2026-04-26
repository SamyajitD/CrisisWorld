"""Run manifest schema for training/evaluation pipelines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunManifest(BaseModel):
    """Describes a training or evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_name: str
    run_type: str = Field(
        description="One of: router_sft, single_policy_sft, cortex_eval, single_eval"
    )
    dataset_repo: str = ""
    dataset_split: str = "train"
    base_models: dict[str, str] = {}  # role_name -> model_id
    adapter_output_repo: str = ""
    sft_config: SFTConfig = Field(default_factory=lambda: SFTConfig())
    eval_config: EvalConfig = Field(default_factory=lambda: EvalConfig())
    seeds: tuple[int, ...] = (42, 43, 44)
    quantization: str = "4bit"  # "4bit", "8bit", "none"
    hub_results_repo: str = ""


class SFTConfig(BaseModel):
    """SFT hyperparameters for TRL/SFTTrainer."""

    model_config = ConfigDict(frozen=True)

    learning_rate: float = Field(default=2e-4, gt=0)
    num_epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=4, gt=0)
    max_seq_length: int = Field(default=2048, gt=0)
    lora_r: int = Field(default=16, gt=0)
    lora_alpha: int = Field(default=32, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)


class EvalConfig(BaseModel):
    """Evaluation configuration for post-training assessment."""

    model_config = ConfigDict(frozen=True)

    num_regions: int = Field(default=4, gt=0)
    max_turns: int = Field(default=20, gt=0)
    eval_seeds: tuple[int, ...] = (100, 101, 102, 103, 104)
    conditions: tuple[str, ...] = ("flat-fat", "cortex-full", "cortex-llm")
    budget: int = Field(default=30, gt=0)
