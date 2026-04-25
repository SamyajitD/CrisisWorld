"""WorldModelerRole — hidden variable estimation + forecasts."""

from __future__ import annotations

from typing import Any

from schemas.artifact import BeliefState, RoleInput

_DEFAULT_SPREAD_RATE = 1.1
_FORECAST_HORIZON = 5
_BASE_CONFIDENCE = 0.8


class WorldModelerRole:
    """Estimate hidden variables and produce forecast trajectories."""

    @property
    def role_name(self) -> str:
        return "world_modeler"

    @property
    def cost(self) -> int:
        return 2

    def invoke(self, role_input: RoleInput) -> BeliefState:
        clean = role_input.payload.get("clean_state", {})
        digest = role_input.payload.get("memory_digest", {})

        anomalies = clean.get("flagged_anomalies", ())
        regions = clean.get("cleaned_observation", {})
        if isinstance(regions, dict):
            regions = regions.get("regions", regions)

        # Current total infected
        total_infected = _sum_infected(regions)

        # Hidden variables
        anomaly_count = len(anomalies)
        multiplier = 2.0 + 0.1 * anomaly_count
        spread_rate = _estimate_spread_rate(digest)
        depletion = 0.0

        hidden = {
            "true_infected_multiplier": multiplier,
            "spread_rate": spread_rate,
            "resource_depletion_rate": depletion,
        }

        # Forecasts
        trajectories = _build_forecasts(
            total_infected, spread_rate, anomalies
        )

        # Confidence
        confidence = _compute_confidence(
            anomaly_count, digest
        )

        return BeliefState(
            hidden_var_estimates=hidden,
            forecast_trajectories=tuple(trajectories),
            confidence=confidence,
        )


def _sum_infected(regions: Any) -> int:
    if isinstance(regions, list):
        return sum(r.get("infected", 0) for r in regions)
    if isinstance(regions, dict):
        val = regions.get("infected", 0)
        if isinstance(val, int):
            return val
    return 0


def _estimate_spread_rate(
    digest: dict[str, Any],
) -> float:
    num_entries = digest.get("num_entries", 0)
    if num_entries < 2:
        return _DEFAULT_SPREAD_RATE
    return _DEFAULT_SPREAD_RATE


def _build_forecasts(
    base_infected: int,
    spread_rate: float,
    anomalies: Any,
) -> list[dict[str, Any]]:
    if base_infected == 0:
        zero = [0] * _FORECAST_HORIZON
        return [
            {"label": "optimistic", "values": zero},
            {"label": "baseline", "values": zero},
            {"label": "pessimistic", "values": zero},
        ]

    has_contradictions = any(
        "CONTRADICTION" in str(a)
        for a in (anomalies or ())
    )
    pessimistic_factor = 1.4 if has_contradictions else 1.2

    opt = _project(base_infected, spread_rate * 0.8)
    base = _project(base_infected, spread_rate)
    pess = _project(base_infected, pessimistic_factor)

    return [
        {"label": "optimistic", "values": opt},
        {"label": "baseline", "values": base},
        {"label": "pessimistic", "values": pess},
    ]


def _project(
    start: int, rate: float
) -> list[int]:
    vals: list[int] = []
    current = float(start)
    for t in range(1, _FORECAST_HORIZON + 1):
        current = start * (rate**t)
        vals.append(max(0, int(current)))
    return vals


def _compute_confidence(
    anomaly_count: int,
    digest: dict[str, Any],
) -> float:
    c = _BASE_CONFIDENCE
    c -= 0.1 * min(anomaly_count, 4)
    if digest.get("num_entries", 0) < 3:
        c -= 0.1
    return max(0.1, min(1.0, c))
