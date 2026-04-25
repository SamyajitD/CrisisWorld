"""Partial observation assembly for CrisisWorld."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..models import (
    Constraint,
    IncidentReport,
    Observation,
    RegionState,
    ResourcePool,
    StakeholderSignal,
    Telemetry,
    BudgetStatusSnapshot
)


def _noisy_count(true_val: int, rng: np.random.Generator, noise_scale: float = 0.2) -> int:
    """Add Gaussian noise proportional to true value, clamp >= 0."""
    if true_val == 0:
        return 0
    noise = rng.normal(0, noise_scale * true_val)
    return max(0, int(round(true_val + noise)))


def _clamp_to_pop(infected: int, recovered: int, deceased: int, population: int) -> tuple[int, int, int]:
    """Ensure I+R+D <= population."""
    total = infected + recovered + deceased
    if total <= population:
        return infected, recovered, deceased
    scale = population / total if total > 0 else 0
    return int(infected * scale), int(recovered * scale), int(deceased * scale)


def assemble_observation(
    regions: tuple[RegionState, ...],
    resources: ResourcePool,
    constraints: tuple[Constraint, ...],
    signals: tuple[StakeholderSignal, ...],
    budget_status: BudgetStatusSnapshot,
    turn: int,
    rng: np.random.Generator,
    done: bool = False,
    reward: float | None = None,
    metadata: dict[str, Any] | None = None,
    prev_regions: tuple[RegionState, ...] | None = None,
    noise_scale: float = 0.2,
) -> Observation:
    """Build a partial, noisy observation for the agent."""
    # Noisy region states
    noisy_regions: list[RegionState] = []
    for r in regions:
        ni = _noisy_count(r.infected, rng, noise_scale)
        nr = _noisy_count(r.recovered, rng, noise_scale)
        nd = _noisy_count(r.deceased, rng, noise_scale)
        ni, nr, nd = _clamp_to_pop(ni, nr, nd, r.population)
        noisy_regions.append(
            RegionState(
                region_id=r.region_id,
                population=r.population,
                infected=ni,
                recovered=nr,
                deceased=nd,
                restricted=r.restricted,
            )
        )

    # Telemetry (lagged 1 turn -- use prev_regions if available, else zeros)
    source = prev_regions if prev_regions is not None and turn > 0 else regions
    telemetry = Telemetry(
        total_infected=sum(r.infected for r in source),
        total_recovered=sum(r.recovered for r in source),
        total_deceased=sum(r.deceased for r in source),
        data_staleness=1 if turn > 0 else 0,
    )

    # Incident reports from regions with high infection
    incidents: list[IncidentReport] = []
    for r in regions:
        if r.population > 0 and r.infected / r.population > 0.1:
            incidents.append(
                IncidentReport(
                    region_id=r.region_id,
                    severity=min(1.0, r.infected / r.population),
                    reported_turn=turn,
                )
            )

    return Observation(
        turn=turn,
        regions=tuple(noisy_regions),
        stakeholder_signals=signals,
        incidents=tuple(incidents),
        telemetry=telemetry,
        resources=resources,
        active_constraints=constraints,
        budget_status=budget_status,
        done=done,
        reward=reward,
        metadata=metadata or {},
    )
