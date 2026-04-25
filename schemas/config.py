"""Configuration schemas for cortex and experiments."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import EnvConfig


class CortexConfig(BaseModel):
    """Cortex deliberation configuration."""

    model_config = ConfigDict(frozen=True)

    total_budget: int = Field(default=20, gt=0)
    perception_cost: int = Field(default=1, gt=0)
    world_modeler_cost: int = Field(default=2, gt=0)
    planner_cost: int = Field(default=2, gt=0)
    critic_cost: int = Field(default=2, gt=0)
    executive_cost: int = Field(default=1, gt=0)
    max_inner_iterations: int = Field(default=10, gt=0)
    memory_enabled: bool = True
    critic_enabled: bool = True


class ExperimentConfig(BaseModel):
    """Full experiment specification."""

    model_config = ConfigDict(frozen=True)

    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    env_config: EnvConfig = Field(default_factory=EnvConfig)
    cortex_config: CortexConfig = Field(default_factory=CortexConfig)
    output_dir: str = "results"
    trace_dir: str = "traces"

    @model_validator(mode="after")
    def _check_non_empty(self) -> ExperimentConfig:
        if not self.seeds:
            raise ValueError("seeds must be non-empty")
        if not self.conditions:
            raise ValueError("conditions must be non-empty")
        return self
