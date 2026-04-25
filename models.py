"""CrisisWorld environment models — Action, Observation, State types.

Extends OpenEnv base models. This is the leaf node of the project's
dependency graph — no imports from any other project module.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from openenv.core.env_server.types import (
    Action as BaseAction,
    EnvironmentMetadata,
    Observation as BaseObservation,
    State as BaseState,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Budget snapshot (env-contract type -- server can import this)
# ---------------------------------------------------------------------------


class BudgetStatusSnapshot(BaseModel):
    """Budget state snapshot for Observation. Env-contract equivalent of schemas.budget.BudgetStatus."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(ge=0)
    spent: int = Field(ge=0)
    remaining: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_invariant(self) -> BudgetStatusSnapshot:
        if self.remaining != self.total - self.spent:
            raise ValueError(
                f"remaining ({self.remaining}) must equal total - spent ({self.total - self.spent})"
            )
        return self


# ---------------------------------------------------------------------------
# Observation components
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# State components
# ---------------------------------------------------------------------------


class RegionState(BaseModel):
    """Epidemiological state of one region."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    population: int = Field(gt=0)
    infected: int = Field(default=0, ge=0)
    recovered: int = Field(default=0, ge=0)
    deceased: int = Field(default=0, ge=0)
    restricted: bool = False

    @model_validator(mode="after")
    def _check_population_invariant(self) -> RegionState:
        total = self.infected + self.recovered + self.deceased
        if total > self.population:
            raise ValueError(
                f"infected ({self.infected}) + recovered ({self.recovered}) "
                f"+ deceased ({self.deceased}) = {total} "
                f"exceeds population ({self.population})"
            )
        return self


class ResourcePool(BaseModel):
    """Available operational resources."""

    model_config = ConfigDict(frozen=True)

    medical: int = Field(default=0, ge=0)
    personnel: int = Field(default=0, ge=0)
    funding: int = Field(default=0, ge=0)


class Constraint(BaseModel):
    """Active policy or legal constraint."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    active: bool = True


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


class RewardComponents(BaseModel):
    """Individual reward terms."""

    model_config = ConfigDict(frozen=True)

    outcome: float = 0.0
    timeliness: float = 0.0
    inner_compute_cost: float = Field(default=0.0, ge=0.0)
    safety_violations: float = Field(default=0.0, ge=0.0)
    comms_quality: float = 0.0


_VALID_WEIGHT_KEYS = frozenset(RewardComponents.model_fields.keys())


class CompositeReward(BaseModel):
    """Weighted composite result with total, components, weights."""

    model_config = ConfigDict(frozen=True)

    total: float
    components: RewardComponents
    weights: dict[str, float]

    @model_validator(mode="after")
    def _check_weight_keys(self) -> CompositeReward:
        invalid = set(self.weights.keys()) - _VALID_WEIGHT_KEYS
        if invalid:
            msg = (
                f"Unknown weight keys: {sorted(invalid)}. "
                f"Valid keys: {sorted(_VALID_WEIGHT_KEYS)}"
            )
            raise ValueError(msg)
        return self


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class RewardWeights(BaseModel):
    """Weights for the composite reward function."""

    model_config = ConfigDict(frozen=True)

    outcome: float = Field(default=1.0, ge=0.0)
    timeliness: float = Field(default=0.5, ge=0.0)
    inner_compute_cost: float = Field(default=0.1, ge=0.0)
    safety_violations: float = Field(default=1.0, ge=0.0)
    comms_quality: float = Field(default=0.3, ge=0.0)


class EnvConfig(BaseModel):
    """CrisisWorld configuration."""

    model_config = ConfigDict(frozen=True)

    num_regions: int = Field(default=4, gt=0)
    max_turns: int = Field(default=50, gt=0)
    initial_infected: int = Field(default=10, ge=0)
    noise_level: float = Field(default=0.1, ge=0.0, le=1.0)
    telemetry_lag: int = Field(default=1, ge=0)
    reward_weights: RewardWeights = Field(default_factory=RewardWeights)


# ---------------------------------------------------------------------------
# Observation (extends OpenEnv)
# ---------------------------------------------------------------------------


class Observation(BaseObservation):
    """Full observation bundle. Inherits done, reward, metadata from OpenEnv."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    turn: int = Field(ge=0)
    regions: tuple[RegionState, ...]
    stakeholder_signals: tuple[StakeholderSignal, ...] = ()
    incidents: tuple[IncidentReport, ...] = ()
    telemetry: Telemetry
    resources: ResourcePool
    active_constraints: tuple[Constraint, ...] = ()
    budget_status: BudgetStatusSnapshot


# ---------------------------------------------------------------------------
# Actions (extend OpenEnv)
# ---------------------------------------------------------------------------


class OuterAction(BaseAction):
    """Base action. Inherits metadata from OpenEnv."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str


class DeployResource(OuterAction):
    """Send resources to a region."""

    kind: Literal["deploy_resource"] = "deploy_resource"
    resource: str
    region_id: str
    amount: int = Field(gt=0)


class RestrictMovement(OuterAction):
    """Impose movement restrictions on a region."""

    kind: Literal["restrict_movement"] = "restrict_movement"
    region_id: str
    level: int = Field(ge=0, le=3)


class RequestData(OuterAction):
    """Request better data from a source (costs budget)."""

    kind: Literal["request_data"] = "request_data"
    source: str


class PublicCommunication(OuterAction):
    """Issue a public statement."""

    kind: Literal["public_communication"] = "public_communication"
    audience: str
    message: str = Field(min_length=1)


class Escalate(OuterAction):
    """Escalate to a higher authority."""

    kind: Literal["escalate"] = "escalate"
    agency: str


class ReallocateBudget(OuterAction):
    """Shift budget between categories."""

    kind: Literal["reallocate_budget"] = "reallocate_budget"
    from_category: str
    to_category: str
    amount: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_categories_differ(self) -> ReallocateBudget:
        if self.from_category == self.to_category:
            raise ValueError("from_category and to_category must differ")
        return self


class NoOp(OuterAction):
    """Do nothing this turn."""

    kind: Literal["noop"] = "noop"


ActionUnion = Annotated[
    Union[
        DeployResource,
        RestrictMovement,
        RequestData,
        PublicCommunication,
        Escalate,
        ReallocateBudget,
        NoOp,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# State (extends OpenEnv -- NOT frozen, server mutates step_count)
# ---------------------------------------------------------------------------


class CrisisState(BaseState):
    """Server-side episode state. Inherits episode_id, step_count from OpenEnv.

    Sole exception to the frozen=True rule: CrisisState uses
    ConfigDict(extra='allow') because the server mutates step_count
    between steps.
    """

    regions: tuple[RegionState, ...] = ()
    resources: ResourcePool | None = None


# EnvironmentMetadata is imported from openenv.core.env_server.types above.
