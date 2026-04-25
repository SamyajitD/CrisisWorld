"""SIR epidemiological model (discrete-time stochastic) for CrisisWorld."""

from __future__ import annotations

from math import exp

import numpy as np

from ..models import RegionState
from ._internal import EpiParams


def _advance_single_region(
    region: RegionState,
    params: EpiParams,
    rng: np.random.Generator,
) -> RegionState:
    """Intra-region SIR step. Returns updated region (new object)."""
    if region.infected == 0:
        return region

    pop = region.population
    s = pop - region.infected - region.recovered - region.deceased

    if s <= 0:
        # No susceptible left -- only recovery and death
        new_infections = 0
    else:
        effective_beta = params.beta * 0.5 if region.restricted else params.beta
        p_infect = 1.0 - exp(-effective_beta * region.infected / pop)
        p_infect = max(0.0, min(1.0, p_infect))
        new_infections = int(rng.binomial(s, p_infect))

    new_recoveries = int(rng.binomial(region.infected, params.gamma))
    new_deaths = int(rng.binomial(region.infected, params.mu))

    # Clamp: recoveries + deaths cannot exceed current infected
    total_outflow = new_recoveries + new_deaths
    if total_outflow > region.infected:
        scale = region.infected / total_outflow
        new_recoveries = int(new_recoveries * scale)
        new_deaths = region.infected - new_recoveries

    infected = max(0, region.infected + new_infections - new_recoveries - new_deaths)
    recovered = max(0, region.recovered + new_recoveries)
    deceased = max(0, region.deceased + new_deaths)

    # Final safety: ensure I+R+D <= population
    total = infected + recovered + deceased
    if total > pop:
        excess = total - pop
        infected = max(0, infected - excess)

    return region.model_copy(
        update={
            "infected": infected,
            "recovered": recovered,
            "deceased": deceased,
        }
    )


def _compute_spillovers(
    regions: tuple[RegionState, ...],
    adjacency: dict[str, list[str]],
    params: EpiParams,
    rng: np.random.Generator,
) -> dict[str, int]:
    """Compute inter-region spillover infections.

    Returns a mapping of region_id -> total incoming spillover count.
    """
    by_id = {r.region_id: r for r in regions}
    incoming: dict[str, int] = {r.region_id: 0 for r in regions}

    for region in regions:
        if region.infected == 0:
            continue
        neighbors = adjacency.get(region.region_id, [])
        for neighbor_id in neighbors:
            spillover = int(rng.binomial(region.infected, params.inter_region_spread))
            if region.restricted:
                spillover = int(spillover * 0.1)
            # Cap at target susceptible count
            target = by_id.get(neighbor_id)
            if target is None:
                continue
            target_s = target.population - target.infected - target.recovered - target.deceased
            spillover = min(spillover, max(0, target_s))
            incoming[neighbor_id] += spillover

    return incoming


def advance_epi_state(
    regions: tuple[RegionState, ...],
    adjacency: dict[str, list[str]],
    params: EpiParams,
    rng: np.random.Generator,
) -> tuple[RegionState, ...]:
    """Advance all regions by one SIR time step.

    1. Intra-region dynamics (infection, recovery, death).
    2. Inter-region spillover from adjacent infected regions.

    All stochastic draws use the provided ``rng`` for reproducibility.
    """
    # Early exit: nothing to do if no region is infected
    if all(r.infected == 0 for r in regions):
        return regions

    # Phase 1: intra-region dynamics
    updated = tuple(_advance_single_region(r, params, rng) for r in regions)

    # Phase 2: inter-region spillover (computed from *original* regions)
    incoming = _compute_spillovers(regions, adjacency, params, rng)

    # Apply spillover to updated regions
    result: list[RegionState] = []
    for r in updated:
        spill = incoming.get(r.region_id, 0)
        if spill <= 0:
            result.append(r)
            continue

        # Cap spillover at susceptible in the *updated* region
        s = r.population - r.infected - r.recovered - r.deceased
        spill = min(spill, max(0, s))
        new_infected = r.infected + spill

        # Final safety clamp
        total = new_infected + r.recovered + r.deceased
        if total > r.population:
            new_infected = r.population - r.recovered - r.deceased

        result.append(r.model_copy(update={"infected": max(0, new_infected)}))

    return tuple(result)
