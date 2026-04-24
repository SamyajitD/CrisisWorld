"""Experiment analysis — comparison tables and diagnostics."""

from __future__ import annotations

import math
import statistics
from typing import Any

from src.evaluation.metrics import compute_confidence_interval
from src.evaluation.runner import ExperimentResults


def comparison_table(results: ExperimentResults) -> str:
    """Markdown table: conditions as rows, metrics as columns."""
    header = (
        "| Condition | N | Reward (mean +/- std) "
        "| Duration (mean +/- std) |"
    )
    sep = "|-----------|---|----------------------|" "------------------------|"
    rows = [header, sep]

    for cond_name, episodes in sorted(results.conditions.items()):
        if not episodes:
            rows.append(f"| {cond_name} | 0 | N/A | N/A |")
            continue

        rewards = [
            e.metrics.get("total_cumulative_reward", e.total_reward)
            for e in episodes
        ]
        durations = [
            e.metrics.get("outbreak_duration", e.total_turns)
            for e in episodes
        ]
        n = len(episodes)
        r_mean, r_std = _mean_std(rewards)
        d_mean, d_std = _mean_std(durations)
        rows.append(
            f"| {cond_name} | {n} "
            f"| {_fmt(r_mean)} +/- {_fmt(r_std)} "
            f"| {_fmt(d_mean)} +/- {_fmt(d_std)} |"
        )

    return "\n".join(rows)


def diagnostic_report(results: ExperimentResults) -> str:
    """Role-call frequency and budget spend rate for cortex conditions."""
    lines: list[str] = ["# Diagnostic Report", ""]

    for cond_name, episodes in sorted(results.conditions.items()):
        if not episodes:
            continue
        # Only report diagnostics for cortex conditions
        if not cond_name.startswith("cortex"):
            continue

        lines.append(f"## {cond_name}")
        lines.append("")

        # Role-call frequency
        role_counts: dict[str, list[int]] = {}
        spend_rates: list[float] = []
        for ep in episodes:
            freq = ep.metrics.get("role_call_frequency", {})
            if isinstance(freq, dict):
                for role, count in freq.items():
                    role_counts.setdefault(role, []).append(count)
            rate = ep.metrics.get("budget_spend_rate", 0.0)
            spend_rates.append(float(rate))

        if role_counts:
            lines.append("### Role-Call Frequency")
            lines.append("| Role | Mean | Total |")
            lines.append("|------|------|-------|")
            for role, counts in sorted(role_counts.items()):
                mean_c = statistics.mean(counts)
                total_c = sum(counts)
                lines.append(
                    f"| {role} | {mean_c:.1f} | {total_c} |"
                )
            lines.append("")

        if spend_rates:
            mean_rate = statistics.mean(spend_rates)
            lines.append(
                f"Budget spend rate: {mean_rate:.2f}"
            )
            lines.append("")

    return "\n".join(lines) if len(lines) > 2 else "No diagnostics."


def significance_summary(results: ExperimentResults) -> str:
    """Pairwise CI overlap analysis between conditions."""
    cond_names = sorted(results.conditions.keys())

    if len(cond_names) < 2:
        return "Insufficient conditions for comparison."

    lines: list[str] = ["# Significance Summary", ""]

    for i, name_a in enumerate(cond_names):
        for name_b in cond_names[i + 1 :]:
            eps_a = results.conditions[name_a]
            eps_b = results.conditions[name_b]
            rewards_a = [
                e.metrics.get(
                    "total_cumulative_reward", e.total_reward
                )
                for e in eps_a
            ]
            rewards_b = [
                e.metrics.get(
                    "total_cumulative_reward", e.total_reward
                )
                for e in eps_b
            ]

            ci_a = compute_confidence_interval(rewards_a)
            ci_b = compute_confidence_interval(rewards_b)

            if _any_nan(ci_a) or _any_nan(ci_b):
                verdict = "insufficient data"
            elif ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:
                verdict = "likely significant"
            else:
                verdict = "not significant"

            lines.append(
                f"- {name_a} vs {name_b}: {verdict}"
            )

    return "\n".join(lines)


def _mean_std(values: list[Any]) -> tuple[float, float]:
    """Compute mean and sample std, handling empty/NaN."""
    floats = [float(v) for v in values]
    if not floats:
        return (float("nan"), float("nan"))
    mean = statistics.mean(floats)
    std = statistics.stdev(floats) if len(floats) > 1 else 0.0
    return (mean, std)


def _fmt(value: float) -> str:
    """Format a float for display, showing N/A for NaN."""
    if math.isnan(value):
        return "N/A"
    return f"{value:.2f}"


def _any_nan(ci: tuple[float, float]) -> bool:
    """Check if either CI bound is NaN."""
    return math.isnan(ci[0]) or math.isnan(ci[1])
