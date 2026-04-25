"""Grid/region state management for CrisisWorld."""

from __future__ import annotations

import math

import numpy as np

from models import EnvConfig, RegionState


def init_regions(
    config: EnvConfig, rng: np.random.Generator
) -> tuple[RegionState, ...]:
    """Create regions with random populations. All start healthy."""
    n = config.num_regions
    cols = max(1, int(math.ceil(math.sqrt(n))))
    regions: list[RegionState] = []
    for i in range(n):
        pop = int(rng.integers(50_000, 200_001))
        regions.append(
            RegionState(region_id=f"r{i}", population=pop)
        )
    return tuple(regions)


def build_adjacency(
    ids: list[str], cols: int
) -> dict[str, list[str]]:
    """Build 4-connected adjacency from a grid layout."""
    n = len(ids)
    adj: dict[str, list[str]] = {rid: [] for rid in ids}
    for idx in range(n):
        row, col = divmod(idx, cols)
        # right
        if col + 1 < cols and idx + 1 < n:
            adj[ids[idx]].append(ids[idx + 1])
            adj[ids[idx + 1]].append(ids[idx])
        # down
        down = idx + cols
        if down < n:
            adj[ids[idx]].append(ids[down])
            adj[ids[down]].append(ids[idx])
    # deduplicate preserving order
    for k in adj:
        adj[k] = list(dict.fromkeys(adj[k]))
    return adj


def seed_infection(
    regions: tuple[RegionState, ...],
    origin: str,
    count: int,
) -> tuple[RegionState, ...]:
    """Place initial infected in the origin region, clamped to population."""
    result: list[RegionState] = []
    for r in regions:
        if r.region_id == origin:
            clamped = min(count, r.population)
            result.append(r.model_copy(update={"infected": clamped}))
        else:
            result.append(r)
    return tuple(result)
