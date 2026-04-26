"""Results schema for training and evaluation runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingMetrics(BaseModel):
    """Metrics from an SFT training run."""

    model_config = ConfigDict(frozen=True)

    train_loss: float = 0.0
    eval_loss: float = 0.0
    num_examples: int = Field(default=0, ge=0)
    num_epochs: int = Field(default=0, ge=0)
    training_time_s: float = 0.0
    base_model: str = ""
    adapter_repo: str = ""


class EvalMetrics(BaseModel):
    """Metrics from a CrisisWorld evaluation run."""

    model_config = ConfigDict(frozen=True)

    condition: str = ""
    num_episodes: int = Field(default=0, ge=0)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_duration: float = 0.0
    catastrophic_count: int = Field(default=0, ge=0)
    contained_count: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    fallback_count: int = Field(default=0, ge=0)


class RunResult(BaseModel):
    """Complete result of a training + evaluation run."""

    model_config = ConfigDict(frozen=True)

    run_name: str
    run_type: str
    training: TrainingMetrics | None = None
    evaluations: tuple[EvalMetrics, ...] = ()
    comparison_table: str = ""
    manifest_snapshot: dict[str, Any] = {}
    dataset_hash: str = ""
    adapter_ref: str = ""
    trace_refs: tuple[str, ...] = ()
