"""Evaluation — experiment orchestration and analysis."""

from src.evaluation.ablations import AblationCondition, build_conditions
from src.evaluation.analysis import (
    comparison_table,
    diagnostic_report,
    significance_summary,
)
from src.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_metrics,
    collect_episode_metrics,
    compute_confidence_interval,
)
from src.evaluation.runner import ExperimentResults, ExperimentRunner

__all__ = [
    "AblationCondition",
    "AggregateMetrics",
    "EpisodeMetrics",
    "ExperimentResults",
    "ExperimentRunner",
    "aggregate_metrics",
    "build_conditions",
    "collect_episode_metrics",
    "comparison_table",
    "compute_confidence_interval",
    "diagnostic_report",
    "significance_summary",
]
