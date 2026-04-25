"""Composite reward computation for CrisisWorld."""

from __future__ import annotations

from models import (
    CompositeReward,
    OuterAction,
    PublicCommunication,
    RegionState,
    RewardComponents,
    RewardWeights,
)


def compute_reward(
    prev_regions: tuple[RegionState, ...],
    curr_regions: tuple[RegionState, ...],
    action: OuterAction,
    violations: list[str],
    weights: RewardWeights,
    turn: int,
    max_turns: int,
    termination_reason: str = "",
) -> CompositeReward:
    """Compute composite reward from state transition."""
    total_pop = sum(r.population for r in curr_regions)
    total_infected = sum(r.infected for r in curr_regions)
    total_deceased = sum(r.deceased for r in curr_regions)

    # R_outcome
    if total_pop == 0:
        r_outcome = 0.0
    else:
        mortality_rate = total_deceased / total_pop
        infection_rate = total_infected / total_pop
        r_outcome = max(-1.0, min(1.0, 1.0 - 2.0 * mortality_rate - infection_rate))

    # R_timeliness
    prev_infected = sum(r.infected for r in prev_regions)
    if total_pop > 0 and total_infected > prev_infected:
        delta = total_infected - prev_infected
        r_timeliness = -(delta / total_pop)
    else:
        r_timeliness = 0.0

    # C_compute (set by evaluation layer)
    c_compute = 0.0

    # R_violations
    r_violations = len(violations) * 0.1

    # R_comms
    if isinstance(action, PublicCommunication):
        r_comms = 0.05 if violations else 0.2
    else:
        r_comms = 0.0

    components = RewardComponents(
        outcome=r_outcome,
        timeliness=r_timeliness,
        inner_compute_cost=c_compute,
        safety_violations=r_violations,
        comms_quality=r_comms,
    )

    total = (
        weights.outcome * r_outcome
        + weights.timeliness * r_timeliness
        - weights.inner_compute_cost * c_compute
        - weights.safety_violations * r_violations
        + weights.comms_quality * r_comms
    )

    # Terminal bonuses
    if termination_reason == "contained":
        total += 2.0
    elif termination_reason in ("catastrophic_mortality", "catastrophic_spread"):
        total -= 3.0
    elif termination_reason == "max_turns":
        total -= 1.0

    return CompositeReward(
        total=total,
        components=components,
        weights={
            "outcome": weights.outcome,
            "timeliness": weights.timeliness,
            "inner_compute_cost": weights.inner_compute_cost,
            "safety_violations": weights.safety_violations,
            "comms_quality": weights.comms_quality,
        },
    )
