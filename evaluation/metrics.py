"""Metric collection and aggregation for experiments."""

from __future__ import annotations

import math
import statistics

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import t as t_dist

from schemas.episode import EpisodeTrace


class EpisodeMetrics(BaseModel):
    """Metrics extracted from a single episode."""

    model_config = ConfigDict(frozen=True)

    # Primary
    total_cumulative_reward: float = 0.0
    outbreak_duration: int = Field(default=0, ge=0)
    final_mortality_rate: float = 0.0

    # Secondary
    resource_efficiency: float = 0.0
    comms_quality_score: float = 0.0
    constraint_violation_count: int = Field(default=0, ge=0)

    # Diagnostic
    role_call_frequency: dict[str, int] = {}
    budget_spend_rate: float = 0.0
    inner_loop_iterations_per_turn: float = 0.0


class AggregateMetrics(BaseModel):
    """Aggregated metrics across episodes."""

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=0)
    mean_reward: float
    std_reward: float
    ci_reward: tuple[float, float]
    mean_duration: float
    std_duration: float


def collect_episode_metrics(trace: EpisodeTrace) -> EpisodeMetrics:
    """Extract metrics from a single episode trace."""
    total_reward = 0.0
    outbreak_turns = 0

    for tr in trace.turns:
        if tr.reward is not None:
            total_reward += tr.reward.get("total", 0.0)
        if tr.observation is not None:
            infected = tr.observation.get("total_infected", 0)
            if infected or any(
                isinstance(v, dict) and v.get("infected", 0) > 0
                for v in tr.observation.values()
            ):
                outbreak_turns += 1

    return EpisodeMetrics(
        total_cumulative_reward=total_reward,
        outbreak_duration=outbreak_turns,
    )


def aggregate_metrics(
    episodes: list[EpisodeMetrics],
) -> AggregateMetrics:
    """Compute mean, std, 95% CI across episodes."""
    n = len(episodes)
    if n == 0:
        return AggregateMetrics(
            n=0,
            mean_reward=float("nan"),
            std_reward=float("nan"),
            ci_reward=(float("nan"), float("nan")),
            mean_duration=float("nan"),
            std_duration=float("nan"),
        )

    rewards = [e.total_cumulative_reward for e in episodes]
    durations = [float(e.outbreak_duration) for e in episodes]

    mean_r = statistics.mean(rewards)
    std_r = statistics.stdev(rewards) if n > 1 else 0.0
    ci_r = compute_confidence_interval(rewards)

    mean_d = statistics.mean(durations)
    std_d = statistics.stdev(durations) if n > 1 else 0.0

    return AggregateMetrics(
        n=n,
        mean_reward=mean_r,
        std_reward=std_r,
        ci_reward=ci_r,
        mean_duration=mean_d,
        std_duration=std_d,
    )


def compute_confidence_interval(
    values: list[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Compute confidence interval via t-distribution."""
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))

    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    t_val = t_dist.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * se
    return (mean - margin, mean + margin)
