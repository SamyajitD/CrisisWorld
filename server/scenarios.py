"""Deterministic scenario generation from a seed."""

from __future__ import annotations

import numpy as np

from ..models import Constraint, EnvConfig, ResourcePool
from ._internal import EpiParams, ScenarioParams


def generate_scenario(config: EnvConfig, seed: int) -> ScenarioParams:
    """Generate a reproducible scenario from *config* and *seed*."""
    rng = np.random.default_rng(seed)

    origin_region = f"r{rng.integers(0, config.num_regions)}"
    initial_infected = int(rng.integers(10, 501))

    epi_params = EpiParams(
        beta=round(float(rng.uniform(0.15, 0.45)), 4),
        gamma=round(float(rng.uniform(0.05, 0.15)), 4),
        mu=round(float(rng.uniform(0.005, 0.03)), 4),
        inter_region_spread=round(float(rng.uniform(0.01, 0.10)), 4),
        noise_scale=round(float(rng.uniform(0.01, 0.05)), 4),
    )

    initial_resources = ResourcePool(
        medical=int(rng.integers(50, 201)),
        personnel=int(rng.integers(30, 151)),
        funding=int(rng.integers(100, 501)),
    )

    return ScenarioParams(
        origin_region=origin_region,
        initial_infected=initial_infected,
        epi_params=epi_params,
        initial_resources=initial_resources,
        initial_constraints=(),
        max_turns=config.max_turns,
    )
