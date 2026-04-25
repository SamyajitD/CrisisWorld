"""Internal data structures for CrisisWorld server. Not exported."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from models import Constraint, RegionState, ResourcePool


class EpiParams(BaseModel):
    """Epidemiological parameters for the SIR model."""

    model_config = ConfigDict(frozen=True)

    beta: float = Field(ge=0.15, le=0.45)
    gamma: float = Field(ge=0.05, le=0.15)
    mu: float = Field(ge=0.005, le=0.03)
    inter_region_spread: float = Field(ge=0.01, le=0.10)
    noise_scale: float = Field(ge=0.01, le=0.05)


class ScheduledEffect(BaseModel):
    """Delayed action effect applied on a future turn."""

    model_config = ConfigDict(frozen=True)

    apply_on_turn: int = Field(ge=0)
    effect_type: str
    target_region: str | None = None
    payload: dict[str, Any] = {}


class ScenarioParams(BaseModel):
    """Deterministic scenario generated from a seed."""

    model_config = ConfigDict(frozen=True)

    origin_region: str
    initial_infected: int = Field(ge=1)
    epi_params: EpiParams
    initial_resources: ResourcePool
    initial_constraints: tuple[Constraint, ...] = ()
    max_turns: int = Field(gt=0)


class InternalState(BaseModel):
    """Mutable server-side state during an episode.

    NOT frozen -- the server mutates this between steps.
    """

    model_config = ConfigDict(extra="allow")

    episode_id: str = ""
    turn: int = Field(default=0, ge=0)
    regions: tuple[RegionState, ...] = ()
    adjacency: dict[str, list[str]] = {}
    resources: ResourcePool = Field(default_factory=ResourcePool)
    constraints: tuple[Constraint, ...] = ()
    pending_effects: tuple[ScheduledEffect, ...] = ()
    action_history: list[dict[str, Any]] = []
    epi_params: EpiParams | None = None
    scenario: ScenarioParams | None = None
