"""Unit tests for server/rewards.py — composite reward computation."""

from __future__ import annotations

from models import (
    NoOp,
    PublicCommunication,
    RegionState,
    RewardWeights,
)


def _make_regions(
    infected: int = 0, deceased: int = 0, population: int = 1000
) -> tuple[RegionState, ...]:
    return (
        RegionState(
            region_id="r0",
            population=population,
            infected=infected,
            recovered=0,
            deceased=deceased,
        ),
    )


class TestRewards:
    def test_perfect_outcome(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions(infected=0, deceased=0)
        r = compute_reward(regions, regions, NoOp(), [], RewardWeights(), 1, 50)
        assert r.components.outcome == 1.0

    def test_high_mortality_penalty(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions(infected=0, deceased=600)
        r = compute_reward(regions, regions, NoOp(), [], RewardWeights(), 1, 50)
        assert r.components.outcome < 0  # 1.0 - 2*0.6 = -0.2

    def test_timeliness_growing_infection(self) -> None:
        from server.rewards import compute_reward

        prev = _make_regions(infected=10)
        curr = _make_regions(infected=50)
        r = compute_reward(prev, curr, NoOp(), [], RewardWeights(), 1, 50)
        assert r.components.timeliness < 0

    def test_timeliness_shrinking_infection(self) -> None:
        from server.rewards import compute_reward

        prev = _make_regions(infected=50)
        curr = _make_regions(infected=10)
        r = compute_reward(prev, curr, NoOp(), [], RewardWeights(), 1, 50)
        assert r.components.timeliness == 0.0

    def test_violations_stacking(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions()
        violations = ["v1", "v2", "v3"]
        r = compute_reward(regions, regions, NoOp(), violations, RewardWeights(), 1, 50)
        assert abs(r.components.safety_violations - 0.3) < 1e-9

    def test_comms_reward_clean(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions()
        action = PublicCommunication(audience="public", message="Stay safe")
        r = compute_reward(regions, regions, action, [], RewardWeights(), 1, 50)
        assert abs(r.components.comms_quality - 0.2) < 1e-9

    def test_comms_reward_with_violations(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions()
        action = PublicCommunication(audience="public", message="Stay safe")
        r = compute_reward(regions, regions, action, ["v1"], RewardWeights(), 1, 50)
        assert abs(r.components.comms_quality - 0.05) < 1e-9

    def test_composite_weighted_sum(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions(infected=0, deceased=0)
        w = RewardWeights(outcome=2.0, timeliness=0.0, inner_compute_cost=0.0,
                          safety_violations=0.0, comms_quality=0.0)
        r = compute_reward(regions, regions, NoOp(), [], w, 1, 50)
        assert abs(r.total - 2.0) < 1e-9  # 2.0 * 1.0 outcome

    def test_zero_population_no_crash(self) -> None:
        from server.rewards import compute_reward

        # Use population=1 (minimum allowed by RegionState)
        regions = (RegionState(region_id="r0", population=1),)
        r = compute_reward(regions, regions, NoOp(), [], RewardWeights(), 1, 50)
        assert r.components.outcome == 1.0

    def test_terminal_bonus_contained(self) -> None:
        from server.rewards import compute_reward

        regions = _make_regions()
        r = compute_reward(
            regions, regions, NoOp(), [], RewardWeights(), 10, 50,
            termination_reason="contained",
        )
        # Total should include +2.0 bonus
        r_no_bonus = compute_reward(
            regions, regions, NoOp(), [], RewardWeights(), 10, 50,
        )
        assert abs(r.total - r_no_bonus.total - 2.0) < 1e-9
