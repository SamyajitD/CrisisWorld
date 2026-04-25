"""Episode termination conditions for CrisisWorld."""

from __future__ import annotations

from ..models import RegionState, ResourcePool


def check_termination(
    regions: tuple[RegionState, ...],
    resources: ResourcePool,
    initial_resources: ResourcePool,
    turn: int,
    max_turns: int,
) -> tuple[bool, str]:
    """Check termination conditions in priority order.

    Returns (done, reason). First match wins.
    """
    if not regions:
        return (False, "")

    total_pop = sum(r.population for r in regions)
    if total_pop == 0:
        return (False, "")

    total_infected = sum(r.infected for r in regions)
    total_deceased = sum(r.deceased for r in regions)

    # 1. Contained: all regions < 1% infected AND turn >= 5
    if turn >= 5 and total_pop > 0:
        all_contained = all(
            r.infected / r.population < 0.01 for r in regions if r.population > 0
        )
        if all_contained:
            return (True, "contained")

    # 2. Catastrophic mortality: deceased > 5% total pop
    if total_pop > 0 and total_deceased / total_pop > 0.05:
        return (True, "catastrophic_mortality")

    # 3. Catastrophic spread: infected > 60% total pop
    if total_pop > 0 and total_infected / total_pop > 0.60:
        return (True, "catastrophic_spread")

    # 4. Resource collapse: all resources < 5% of initial
    init_total = initial_resources.medical + initial_resources.personnel + initial_resources.funding
    if init_total > 0:
        curr_total = resources.medical + resources.personnel + resources.funding
        if curr_total / init_total < 0.05:
            return (True, "resource_collapse")

    # 5. Max turns
    if turn >= max_turns:
        return (True, "max_turns")

    return (False, "")
