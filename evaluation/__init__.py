"""Evaluation — experiment orchestration and analysis."""

from .ablations import (
    AblationCondition,
    build_conditions,
    get_low_budget,
    get_matched_budget,
)
from .analysis import (
    comparison_table,
    diagnostic_report,
    significance_summary,
)
from .metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    collect_episode_metrics,
    compute_confidence_interval,
)
from .runner import ExperimentResults, ExperimentRunner

__all__ = [
    "AblationCondition",
    "AggregateMetrics",
    "EpisodeMetrics",
    "ExperimentResults",
    "ExperimentRunner",
    "aggregate_metrics",
    "build_conditions",
    "get_low_budget",
    "get_matched_budget",
    "collect_episode_metrics",
    "comparison_table",
    "compute_confidence_interval",
    "diagnostic_report",
    "significance_summary",
]
