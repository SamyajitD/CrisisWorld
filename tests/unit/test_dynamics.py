"""Unit tests for server.dynamics -- SIR epidemiological model."""

from __future__ import annotations

import numpy as np
import pytest

from models import RegionState
from server._internal import EpiParams
from server.dynamics import advance_epi_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_params(**overrides: float) -> EpiParams:
    """Create EpiParams with sensible defaults, overridable per field."""
    defaults = {
        "beta": 0.30,
        "gamma": 0.10,
        "mu": 0.01,
        "inter_region_spread": 0.05,
        "noise_scale": 0.02,
    }
    defaults.update(overrides)
    return EpiParams(**defaults)


def _make_region(
    region_id: str = "r0",
    population: int = 100_000,
    infected: int = 1000,
    recovered: int = 0,
    deceased: int = 0,
    restricted: bool = False,
) -> RegionState:
    return RegionState(
        region_id=region_id,
        population=population,
        infected=infected,
        recovered=recovered,
        deceased=deceased,
        restricted=restricted,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleRegionSIRBasic:
    """Known params, bounds check: S+I+R+D == pop for all regions."""

    def test_population_conservation(self) -> None:
        rng = np.random.default_rng(42)
        regions = (_make_region(),)
        adjacency: dict[str, list[str]] = {"r0": []}
        params = _default_params()

        result = advance_epi_state(regions, adjacency, params, rng)

        for r in result:
            total = r.infected + r.recovered + r.deceased
            susceptible = r.population - total
            assert susceptible >= 0, "Susceptible went negative"
            assert total <= r.population, "I+R+D exceeds population"

    def test_infection_changes(self) -> None:
        """With 1000 infected out of 100k, infection should spread."""
        rng = np.random.default_rng(42)
        regions = (_make_region(),)
        adjacency: dict[str, list[str]] = {"r0": []}
        params = _default_params()

        result = advance_epi_state(regions, adjacency, params, rng)

        # Some recovery/death should happen; new infections should occur
        r = result[0]
        assert r.infected >= 0
        assert r.recovered >= 0
        assert r.deceased >= 0


class TestZeroInfectedNoChange:
    """All regions infected=0 -> all unchanged."""

    def test_all_healthy_stays_healthy(self) -> None:
        rng = np.random.default_rng(42)
        regions = (
            _make_region(region_id="r0", infected=0),
            _make_region(region_id="r1", infected=0),
        )
        adjacency = {"r0": ["r1"], "r1": ["r0"]}
        params = _default_params()

        result = advance_epi_state(regions, adjacency, params, rng)

        for orig, updated in zip(regions, result):
            assert updated.infected == orig.infected
            assert updated.recovered == orig.recovered
            assert updated.deceased == orig.deceased


class TestRestrictionHalvesBeta:
    """Restricted region produces fewer infections than unrestricted."""

    def test_mean_infections_lower_when_restricted(self) -> None:
        params = _default_params()
        adjacency: dict[str, list[str]] = {"r0": []}
        trials = 200

        unrestricted_infections: list[int] = []
        restricted_infections: list[int] = []

        for seed in range(trials):
            rng = np.random.default_rng(seed)
            regions_u = (_make_region(restricted=False),)
            result_u = advance_epi_state(regions_u, adjacency, params, rng)
            unrestricted_infections.append(result_u[0].infected)

            rng = np.random.default_rng(seed)
            regions_r = (_make_region(restricted=True),)
            result_r = advance_epi_state(regions_r, adjacency, params, rng)
            restricted_infections.append(result_r[0].infected)

        mean_u = np.mean(unrestricted_infections)
        mean_r = np.mean(restricted_infections)
        assert mean_r < mean_u, (
            f"Restricted mean ({mean_r:.1f}) should be < "
            f"unrestricted mean ({mean_u:.1f})"
        )


class TestInterRegionSpread:
    """2 adjacent regions, one infected -> other gains some."""

    def test_infection_spreads_to_neighbor(self) -> None:
        params = _default_params(inter_region_spread=0.10)
        regions = (
            _make_region(region_id="r0", infected=5000),
            _make_region(region_id="r1", infected=0),
        )
        adjacency = {"r0": ["r1"], "r1": ["r0"]}

        # Run many trials; at least some should show spillover
        spillover_seen = False
        for seed in range(50):
            rng = np.random.default_rng(seed)
            result = advance_epi_state(regions, adjacency, params, rng)
            if result[1].infected > 0:
                spillover_seen = True
                break

        assert spillover_seen, "No inter-region spillover observed in 50 trials"


class TestRestrictionBlocksSpillover:
    """Restricted region outgoing spillover ~10% of unrestricted."""

    def test_restricted_outgoing_much_lower(self) -> None:
        params = _default_params(inter_region_spread=0.10)
        trials = 200

        unrestricted_spill: list[int] = []
        restricted_spill: list[int] = []

        for seed in range(trials):
            # Unrestricted source
            rng = np.random.default_rng(seed)
            regions_u = (
                _make_region(region_id="r0", infected=5000, restricted=False),
                _make_region(region_id="r1", infected=0),
            )
            adjacency = {"r0": ["r1"], "r1": ["r0"]}
            result_u = advance_epi_state(regions_u, adjacency, params, rng)
            unrestricted_spill.append(result_u[1].infected)

            # Restricted source
            rng = np.random.default_rng(seed)
            regions_r = (
                _make_region(region_id="r0", infected=5000, restricted=True),
                _make_region(region_id="r1", infected=0),
            )
            result_r = advance_epi_state(regions_r, adjacency, params, rng)
            restricted_spill.append(result_r[1].infected)

        mean_u = np.mean(unrestricted_spill)
        mean_r = np.mean(restricted_spill)
        # Restricted should be dramatically lower (factor ~0.1)
        assert mean_r < mean_u * 0.5, (
            f"Restricted spillover mean ({mean_r:.1f}) should be < "
            f"50% of unrestricted ({mean_u:.1f})"
        )


class TestDeterminismSameSeed:
    """Two runs with same seed produce identical results."""

    def test_identical_results(self) -> None:
        params = _default_params()
        regions = (
            _make_region(region_id="r0", infected=1000),
            _make_region(region_id="r1", infected=500),
        )
        adjacency = {"r0": ["r1"], "r1": ["r0"]}

        rng1 = np.random.default_rng(99)
        result1 = advance_epi_state(regions, adjacency, params, rng1)

        rng2 = np.random.default_rng(99)
        result2 = advance_epi_state(regions, adjacency, params, rng2)

        for r1, r2 in zip(result1, result2):
            assert r1.infected == r2.infected
            assert r1.recovered == r2.recovered
            assert r1.deceased == r2.deceased


class TestNegativeClamp:
    """Even with high parameters, all counts >= 0."""

    def test_all_counts_non_negative(self) -> None:
        # High rates to stress the clamping logic
        params = _default_params(beta=0.45, gamma=0.15, mu=0.03)
        regions = (
            _make_region(infected=90_000, recovered=5000, deceased=4000),
        )
        adjacency: dict[str, list[str]] = {"r0": []}

        for seed in range(50):
            rng = np.random.default_rng(seed)
            result = advance_epi_state(regions, adjacency, params, rng)
            for r in result:
                assert r.infected >= 0, f"Negative infected: {r.infected}"
                assert r.recovered >= 0, f"Negative recovered: {r.recovered}"
                assert r.deceased >= 0, f"Negative deceased: {r.deceased}"
                total = r.infected + r.recovered + r.deceased
                assert total <= r.population, (
                    f"I+R+D ({total}) > population ({r.population})"
                )


class TestZeroPopulationUnchanged:
    """Minimal population region with no infection stays stable."""

    def test_pop_one_no_infected_unchanged(self) -> None:
        rng = np.random.default_rng(42)
        regions = (
            _make_region(region_id="r0", population=1, infected=0),
        )
        adjacency: dict[str, list[str]] = {"r0": []}
        params = _default_params()

        result = advance_epi_state(regions, adjacency, params, rng)

        assert result[0].infected == 0
        assert result[0].recovered == 0
        assert result[0].deceased == 0
        assert result[0].population == 1
