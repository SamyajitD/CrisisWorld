"""Reward schemas for composite reward computation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
