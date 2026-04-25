"""Stakeholder signal generation with noise and lag."""

from __future__ import annotations

import numpy as np

from ..models import RegionState, StakeholderSignal

# (source_name, lag_min, lag_max, message_template)
_SOURCES: tuple[tuple[str, int, int, str], ...] = (
    ("hospital", 0, 0, "Hospital reports {level} infection rate"),
    ("media", 0, 1, "Media coverage indicates {level} outbreak severity"),
    ("government", 1, 1, "Government advisory: {level} threat level"),
    ("public", 1, 2, "Public sentiment reflects {level} concern"),
)


def _urgency_label(urgency: float) -> str:
    if urgency > 0.7:
        return "high"
    if urgency > 0.3:
        return "moderate"
    return "low"


def _compute_base_urgency(regions: tuple[RegionState, ...]) -> float:
    total_infected = sum(r.infected for r in regions)
    total_pop = sum(r.population for r in regions)
    if total_pop == 0:
        return 0.0
    return total_infected / total_pop


def generate_signals(
    regions: tuple[RegionState, ...],
    turn: int,
    rng: np.random.Generator,
) -> tuple[StakeholderSignal, ...]:
    """Generate one signal per stakeholder source."""
    base_urgency = _compute_base_urgency(regions)
    signals: list[StakeholderSignal] = []

    for source, lag_min, lag_max, template in _SOURCES:
        lag = int(rng.integers(lag_min, lag_max + 1)) if lag_max > lag_min else lag_min
        effective_turn = max(0, turn - lag)

        # For lagged sources at turn 0 or when lag exceeds turn, use current
        if effective_turn == turn or turn == 0:
            urgency = base_urgency
        else:
            urgency = base_urgency  # simplified: use current state

        noise = float(rng.normal(0, 0.1))
        urgency = max(0.0, min(1.0, urgency + noise))

        message = template.format(level=_urgency_label(urgency))

        signals.append(StakeholderSignal(
            source=source,
            urgency=round(urgency, 4),
            message=message,
            turn=turn,
        ))

    return tuple(signals)
