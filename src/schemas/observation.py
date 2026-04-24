"""Observation schemas delivered to the agent each turn."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.budget import BudgetStatus
from src.schemas.state import Constraint, RegionState, ResourcePool


class StakeholderSignal(BaseModel):
    """Noisy signal from a stakeholder source."""

    model_config = ConfigDict(frozen=True)

    source: str
    urgency: float = Field(ge=0.0, le=1.0)
    message: str
    turn: int = Field(ge=0)


class IncidentReport(BaseModel):
    """Single reported outbreak incident."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    severity: float = Field(ge=0.0, le=1.0)
    reported_turn: int = Field(ge=0)
    description: str = ""


class Telemetry(BaseModel):
    """Aggregated numeric outbreak indicators."""

    model_config = ConfigDict(frozen=True)

    total_infected: int = Field(default=0, ge=0)
    total_recovered: int = Field(default=0, ge=0)
    total_deceased: int = Field(default=0, ge=0)
    data_staleness: int = Field(default=0, ge=0)


class Observation(BaseModel):
    """Full observation bundle delivered to the agent each turn."""

    model_config = ConfigDict(frozen=True)

    turn: int = Field(ge=0)
    regions: tuple[RegionState, ...]
    stakeholder_signals: tuple[StakeholderSignal, ...] = ()
    incidents: tuple[IncidentReport, ...] = ()
    telemetry: Telemetry
    resources: ResourcePool
    active_constraints: tuple[Constraint, ...] = ()
    budget_status: BudgetStatus
