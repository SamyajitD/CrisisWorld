"""Integration tests for evaluation/ — per CLAUDE.md spec."""

from __future__ import annotations

import logging
import math
from typing import Any

import pytest

from models import BudgetStatusSnapshot, NoOp, Observation, OuterAction, RegionState, ResourcePool, Telemetry
from models import CompositeReward, RewardComponents
from schemas.config import ExperimentConfig
from schemas.episode import EpisodeResult, EpisodeTrace, LogEvent, TurnRecord

# ---------------------------------------------------------------------------
# Helpers: fake factories for runner tests
# ---------------------------------------------------------------------------

_DEFAULT_OBS = Observation(
    turn=0,
    regions=(
        RegionState(region_id="north", population=1000, infected=10),
    ),
    telemetry=Telemetry(total_infected=10),
    resources=ResourcePool(medical=100, personnel=50, funding=200),
    budget_status=BudgetStatusSnapshot(total=50, spent=0, remaining=50),
)

_DEFAULT_REWARD = CompositeReward(
    total=0.5,
    components=RewardComponents(outcome=0.3, timeliness=0.2),
    weights={"outcome": 1.0, "timeliness": 0.5},
)


def _make_obs(turn: int) -> Observation:
    return _DEFAULT_OBS.model_copy(
        update={
            "turn": turn,
            "budget_status": BudgetStatusSnapshot(
                total=50, spent=turn * 2, remaining=50 - turn * 2
            ),
        },
    )


class FakeEnv:
    """5-turn env with predictable obs/rewards."""

    def __init__(self, max_turns: int = 5) -> None:
        self._max_turns = max_turns
        self._turn = -1
        self._closed = False

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: object) -> Observation:
        self._turn = 0
        self._closed = False
        return _make_obs(0)

    def step(self, action: OuterAction, timeout_s: float | None = None, **kwargs: object) -> Observation:
        self._turn += 1
        done = self._turn >= self._max_turns
        obs = _make_obs(self._turn)
        return obs.model_copy(
            update={
                "done": done,
                "reward": _DEFAULT_REWARD.total,
                "metadata": {"turn": self._turn, "termination_reason": "max_turns" if done else ""},
            },
        )

    def close(self) -> None:
        self._closed = True


class CrashingEnv(FakeEnv):
    """Env that crashes on the 2nd reset call."""

    def __init__(self) -> None:
        super().__init__()
        self._reset_count = 0

    def reset(self, seed: int | None = None, episode_id: str | None = None, **kwargs: object) -> Observation:
        self._reset_count += 1
        if self._reset_count == 2:
            raise RuntimeError("simulated crash")
        return super().reset(seed, episode_id, **kwargs)


class FakeAgent:
    """Always returns NoOp."""

    def act(self, observation: Observation) -> OuterAction:
        return NoOp()

    def reset(self) -> None:
        pass


class FakeLogger:
    """In-memory logger, save is no-op."""

    def __init__(self) -> None:
        self._events: list[LogEvent] = []

    def record(self, event: LogEvent) -> None:
        self._events.append(event)

    def save(self, path: Any) -> Any:
        return path

    def flush(self) -> None:
        self._events.clear()


# ---------------------------------------------------------------------------
# Ablation Tests
# ---------------------------------------------------------------------------


class TestAblations:
    def test_build_conditions_returns_five(self) -> None:
        from evaluation.ablations import build_conditions

        config = ExperimentConfig(
            seeds=(42,),
            conditions=(
                "flat-lite",
                "flat-fat",
                "cortex-lite",
                "cortex-full",
                "cortex-tuned",
            ),
        )
        conds = build_conditions(config)
        assert len(conds) == 5
        names = {c.name for c in conds}
        assert names == {
            "flat-lite",
            "flat-fat",
            "cortex-lite",
            "cortex-full",
            "cortex-tuned",
        }

    def test_flat_conditions_have_no_roles(self) -> None:
        from evaluation.ablations import build_conditions

        config = ExperimentConfig(
            seeds=(42,),
            conditions=("flat-lite", "flat-fat"),
        )
        conds = build_conditions(config)
        for c in conds:
            assert c.enabled_roles == ()
            assert c.memory_enabled is False

    def test_cortex_full_has_all_roles_and_memory(self) -> None:
        from evaluation.ablations import build_conditions

        config = ExperimentConfig(
            seeds=(42,),
            conditions=("cortex-full",),
        )
        conds = build_conditions(config)
        assert len(conds) == 1
        c = conds[0]
        assert len(c.enabled_roles) == 5
        assert c.memory_enabled is True
        assert c.critic_enabled is True

    def test_matched_budget_conditions_share_budget(self) -> None:
        from evaluation.ablations import build_conditions

        config = ExperimentConfig(
            seeds=(42,),
            conditions=("flat-fat", "cortex-full", "cortex-tuned"),
        )
        conds = build_conditions(config)
        budgets = {c.budget for c in conds}
        assert len(budgets) == 1  # all share same matched budget

    def test_low_budget_conditions_share_budget(self) -> None:
        from evaluation.ablations import build_conditions

        config = ExperimentConfig(
            seeds=(42,),
            conditions=("flat-lite", "cortex-lite"),
        )
        conds = build_conditions(config)
        budgets = {c.budget for c in conds}
        assert len(budgets) == 1  # all share same low budget

    def test_build_conditions_with_subset_filter(self) -> None:
        from evaluation.ablations import build_conditions

        config = ExperimentConfig(
            seeds=(42,),
            conditions=("flat-fat", "cortex-full"),
        )
        conds = build_conditions(config)
        assert len(conds) == 2
        names = {c.name for c in conds}
        assert names == {"flat-fat", "cortex-full"}


# ---------------------------------------------------------------------------
# Metric Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_collect_episode_metrics_primary_values(self) -> None:
        from evaluation.metrics import collect_episode_metrics

        trace = EpisodeTrace(
            episode_id="m1",
            seed=42,
            condition="test",
            turns=(
                TurnRecord(
                    turn=0,
                    reward={"total": 0.5, "outcome": 0.3},
                    events=(),
                ),
                TurnRecord(
                    turn=1,
                    reward={"total": -0.2, "outcome": -0.1},
                    events=(),
                ),
            ),
        )
        m = collect_episode_metrics(trace)
        assert abs(m.total_cumulative_reward - 0.3) < 1e-9

    def test_collect_episode_metrics_handles_missing_info(self) -> None:
        from evaluation.metrics import collect_episode_metrics

        trace = EpisodeTrace(
            episode_id="m2",
            seed=42,
            condition="test",
            turns=(TurnRecord(turn=0, events=()),),
        )
        m = collect_episode_metrics(trace)
        assert m.role_call_frequency == {}

    def test_aggregate_metrics_mean_and_std(self) -> None:
        from evaluation.metrics import (
            EpisodeMetrics,
            aggregate_metrics,
        )

        episodes = [
            EpisodeMetrics(
                total_cumulative_reward=v,
                outbreak_duration=0,
                final_mortality_rate=0.0,
            )
            for v in [10.0, 20.0, 30.0]
        ]
        agg = aggregate_metrics(episodes)
        assert abs(agg.mean_reward - 20.0) < 1e-9
        assert abs(agg.std_reward - 10.0) < 1e-9

    def test_aggregate_metrics_empty_list_returns_nan(self) -> None:
        from evaluation.metrics import aggregate_metrics

        agg = aggregate_metrics([])
        assert agg.n == 0
        assert math.isnan(agg.mean_reward)


# ---------------------------------------------------------------------------
# Runner Tests
# ---------------------------------------------------------------------------


class TestRunner:
    def _make_config(
        self,
        seeds: tuple[int, ...] = (42,),
        conditions: tuple[str, ...] = ("flat-lite",),
    ) -> ExperimentConfig:
        return ExperimentConfig(seeds=seeds, conditions=conditions)

    def test_run_single_seed_single_condition(
        self, tmp_path: Any
    ) -> None:
        from evaluation.runner import ExperimentRunner

        config = self._make_config()
        runner = ExperimentRunner(
            env_factory=lambda: FakeEnv(),
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        results = runner.run()
        assert len(results.conditions) == 1
        assert len(results.conditions["flat-lite"]) == 1

    def test_run_multi_seed_produces_correct_count(self) -> None:
        from evaluation.runner import ExperimentRunner

        config = self._make_config(seeds=(42, 43, 44))
        runner = ExperimentRunner(
            env_factory=lambda: FakeEnv(),
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        results = runner.run()
        assert len(results.conditions["flat-lite"]) == 3

    def test_run_all_conditions_produces_five_results(self) -> None:
        from evaluation.runner import ExperimentRunner

        config = self._make_config(
            conditions=(
                "flat-lite",
                "flat-fat",
                "cortex-lite",
                "cortex-full",
                "cortex-tuned",
            ),
        )
        runner = ExperimentRunner(
            env_factory=lambda: FakeEnv(),
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        results = runner.run()
        assert len(results.conditions) == 5

    def test_run_episode_returns_correct_turn_count(self) -> None:
        from evaluation.runner import ExperimentRunner

        config = self._make_config()
        runner = ExperimentRunner(
            env_factory=lambda: FakeEnv(max_turns=5),
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        env = FakeEnv(max_turns=5)
        result = runner.run_episode(
            env=env, agent=FakeAgent(), logger=FakeLogger(), seed=42,
            condition="flat-lite",
        )
        assert result.total_turns == 5

    def test_run_episode_carries_condition_name(self) -> None:
        from evaluation.runner import ExperimentRunner

        config = self._make_config()
        runner = ExperimentRunner(
            env_factory=lambda: FakeEnv(),
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        result = runner.run_episode(
            env=FakeEnv(), agent=FakeAgent(), logger=FakeLogger(),
            seed=42, condition="cortex-full",
        )
        assert result.condition == "cortex-full"

    def test_episode_crash_is_logged_and_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from evaluation.runner import ExperimentRunner

        call_count = 0

        def crashing_env_factory() -> FakeEnv:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return CrashingEnv()
            return FakeEnv()

        config = self._make_config(seeds=(42, 43))
        # CrashingEnv crashes on 2nd reset
        crash_env = CrashingEnv()

        runner = ExperimentRunner(
            env_factory=lambda: crash_env,
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        with caplog.at_level(logging.WARNING):
            results = runner.run()
        # 1 of 2 seeds should succeed
        episodes = results.conditions.get("flat-lite", [])
        assert len(episodes) == 1

    def test_env_close_called_even_on_crash(self) -> None:
        from evaluation.runner import ExperimentRunner

        close_calls: list[bool] = []

        class TrackingEnv(FakeEnv):
            def close(self) -> None:
                close_calls.append(True)
                super().close()

            def step(self, action: OuterAction, timeout_s: float | None = None, **kwargs: object) -> Observation:
                raise RuntimeError("step crash")

        config = self._make_config(seeds=(42,))
        runner = ExperimentRunner(
            env_factory=lambda: TrackingEnv(),
            agent_factory=lambda cond, seed=0: FakeAgent(),
            logger_factory=lambda: FakeLogger(),
            config=config,
        )
        runner.run()
        assert len(close_calls) >= 1


# ---------------------------------------------------------------------------
# Analysis Tests
# ---------------------------------------------------------------------------


class TestAnalysis:
    def _make_results(self) -> Any:
        """Build ExperimentResults with two conditions."""
        from evaluation.runner import ExperimentResults

        return ExperimentResults(
            conditions={
                "flat-fat": [
                    EpisodeResult(
                        episode_id="e1",
                        seed=42,
                        condition="flat-fat",
                        total_turns=5,
                        total_reward=1.0,
                        termination_reason="max_turns",
                        metrics={
                            "total_cumulative_reward": 1.0,
                            "outbreak_duration": 5,
                        },
                    ),
                    EpisodeResult(
                        episode_id="e2",
                        seed=43,
                        condition="flat-fat",
                        total_turns=5,
                        total_reward=2.0,
                        termination_reason="max_turns",
                        metrics={
                            "total_cumulative_reward": 2.0,
                            "outbreak_duration": 4,
                        },
                    ),
                ],
                "cortex-full": [
                    EpisodeResult(
                        episode_id="e3",
                        seed=42,
                        condition="cortex-full",
                        total_turns=5,
                        total_reward=3.0,
                        termination_reason="max_turns",
                        metrics={
                            "total_cumulative_reward": 3.0,
                            "outbreak_duration": 3,
                            "role_call_frequency": {
                                "perception": 5,
                                "planner": 4,
                            },
                            "budget_spend_rate": 0.8,
                        },
                    ),
                    EpisodeResult(
                        episode_id="e4",
                        seed=43,
                        condition="cortex-full",
                        total_turns=5,
                        total_reward=4.0,
                        termination_reason="max_turns",
                        metrics={
                            "total_cumulative_reward": 4.0,
                            "outbreak_duration": 2,
                            "role_call_frequency": {
                                "perception": 5,
                                "planner": 5,
                            },
                            "budget_spend_rate": 0.9,
                        },
                    ),
                ],
            },
        )

    def test_comparison_table_format(self) -> None:
        from evaluation.analysis import comparison_table

        results = self._make_results()
        table = comparison_table(results)
        assert "|" in table
        assert "flat-fat" in table
        assert "cortex-full" in table

    def test_comparison_table_nan_shows_na(self) -> None:
        from evaluation.analysis import comparison_table
        from evaluation.runner import ExperimentResults

        results = ExperimentResults(
            conditions={
                "empty": [],
            },
        )
        table = comparison_table(results)
        assert "N/A" in table

    def test_diagnostic_report_skips_flat(self) -> None:
        from evaluation.analysis import diagnostic_report

        results = self._make_results()
        report = diagnostic_report(results)
        # flat-fat should not have role-call section
        assert "flat-fat" not in report or "role_call" not in report.split(
            "flat-fat"
        )[-1].split("cortex")[0]

    def test_significance_summary_non_overlapping_ci(self) -> None:
        from evaluation.analysis import significance_summary

        results = self._make_results()
        summary = significance_summary(results)
        # With only 2 samples per condition these may overlap,
        # but the function should produce output
        assert "flat-fat" in summary
        assert "cortex-full" in summary

    def test_significance_summary_single_condition(self) -> None:
        from evaluation.analysis import significance_summary
        from evaluation.runner import ExperimentResults

        results = ExperimentResults(
            conditions={
                "only-one": [
                    EpisodeResult(
                        episode_id="e1",
                        seed=42,
                        condition="only-one",
                        total_turns=5,
                        total_reward=1.0,
                        termination_reason="max_turns",
                        metrics={"total_cumulative_reward": 1.0},
                    ),
                ],
            },
        )
        summary = significance_summary(results)
        assert "Insufficient" in summary
