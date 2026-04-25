"""Policy constraint checking for CrisisWorld actions."""

from __future__ import annotations

from models import (
    Constraint,
    DeployResource,
    Escalate,
    NoOp,
    OuterAction,
    ReallocateBudget,
    RegionState,
    RestrictMovement,
)


def _region_by_id(
    region_id: str, regions: tuple[RegionState, ...]
) -> RegionState | None:
    for r in regions:
        if r.region_id == region_id:
            return r
    return None


def _is_active(name: str, active: tuple[Constraint, ...]) -> bool:
    return any(c.name == name and c.active for c in active)


def check_constraints(
    action: OuterAction,
    active: tuple[Constraint, ...],
    regions: tuple[RegionState, ...],
) -> list[str]:
    """Return a list of violation descriptions for *action*."""
    if isinstance(action, NoOp):
        return []

    violations: list[str] = []

    # 1. no_restrict_low_infection
    if (
        _is_active("no_restrict_low_infection", active)
        and isinstance(action, RestrictMovement)
    ):
        region = _region_by_id(action.region_id, regions)
        if region is not None and region.infected / region.population < 0.05:
            violations.append(
                f"no_restrict_low_infection: region {action.region_id} "
                f"has < 5% infected"
            )

    # 2. resource_equity -- deploying to the healthiest (lowest infection rate) region
    if (
        _is_active("resource_equity", active)
        and isinstance(action, DeployResource)
    ):
        region = _region_by_id(action.region_id, regions)
        valid_regions = [r for r in regions if r.population > 0]
        if region is not None and region.population > 0 and valid_regions:
            target_rate = region.infected / region.population
            min_rate = min(
                r.infected / r.population for r in valid_regions
            )
            if target_rate <= min_rate and len(valid_regions) > 1:
                violations.append(
                    f"resource_equity: region {action.region_id} "
                    f"has the lowest infection rate"
                )

    # 3. escalation_requires_threshold
    if (
        _is_active("escalation_requires_threshold", active)
        and isinstance(action, Escalate)
    ):
        any_above = any(
            r.infected / r.population > 0.20 for r in regions
        )
        if not any_above:
            violations.append(
                "escalation_requires_threshold: no region > 20% infected"
            )

    # 4. max_restriction_duration -- skip for now, needs history

    # 5. budget_floor
    if (
        _is_active("budget_floor", active)
        and isinstance(action, ReallocateBudget)
    ):
        if action.amount < 10:
            violations.append(
                "budget_floor: cannot reallocate when funding < 10"
            )

    return violations
