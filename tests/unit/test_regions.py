"""Unit tests for server/regions.py — grid management."""

from __future__ import annotations

import numpy as np

from CrisisWorld.models import EnvConfig


class TestRegions:
    def test_init_regions_count(self) -> None:
        from CrisisWorld.server.regions import init_regions

        regions = init_regions(EnvConfig(num_regions=9), np.random.default_rng(42))
        assert len(regions) == 9

    def test_init_regions_all_healthy(self) -> None:
        from CrisisWorld.server.regions import init_regions

        regions = init_regions(EnvConfig(num_regions=9), np.random.default_rng(42))
        for r in regions:
            assert r.infected == 0

    def test_build_adjacency_corners(self) -> None:
        from CrisisWorld.server.regions import build_adjacency

        ids = [f"r{i}" for i in range(9)]
        adj = build_adjacency(ids, cols=3)
        assert len(adj["r0"]) == 2

    def test_build_adjacency_edge(self) -> None:
        from CrisisWorld.server.regions import build_adjacency

        ids = [f"r{i}" for i in range(9)]
        adj = build_adjacency(ids, cols=3)
        assert len(adj["r1"]) == 3

    def test_build_adjacency_center(self) -> None:
        from CrisisWorld.server.regions import build_adjacency

        ids = [f"r{i}" for i in range(9)]
        adj = build_adjacency(ids, cols=3)
        assert len(adj["r4"]) == 4

    def test_seed_infection(self) -> None:
        from CrisisWorld.server.regions import init_regions, seed_infection

        regions = init_regions(EnvConfig(num_regions=4), np.random.default_rng(42))
        seeded = seed_infection(regions, origin="r0", count=50)
        assert seeded[0].infected == 50
        for r in seeded[1:]:
            assert r.infected == 0

    def test_seed_infection_clamp(self) -> None:
        from CrisisWorld.server.regions import init_regions, seed_infection

        regions = init_regions(EnvConfig(num_regions=4), np.random.default_rng(42))
        pop = regions[0].population
        seeded = seed_infection(regions, origin="r0", count=pop + 1000)
        assert seeded[0].infected <= seeded[0].population

    def test_single_region_grid(self) -> None:
        from CrisisWorld.server.regions import build_adjacency, init_regions

        regions = init_regions(EnvConfig(num_regions=1), np.random.default_rng(42))
        assert len(regions) == 1
        adj = build_adjacency([r.region_id for r in regions], cols=1)
        assert adj[regions[0].region_id] == []

    def test_determinism_same_seed(self) -> None:
        from CrisisWorld.server.regions import init_regions

        cfg = EnvConfig(num_regions=9)
        r1 = init_regions(cfg, np.random.default_rng(42))
        r2 = init_regions(cfg, np.random.default_rng(42))
        for a, b in zip(r1, r2):
            assert a.population == b.population
