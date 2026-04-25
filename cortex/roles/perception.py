"""PerceptionRole — noise filtering + anomaly detection."""

from __future__ import annotations

from typing import Any

from schemas.artifact import CleanState, RoleInput

_STALENESS_THRESHOLD = 3


class PerceptionRole:
    """Clamp, consistency-check, and flag anomalies."""

    @property
    def role_name(self) -> str:
        return "perception"

    @property
    def cost(self) -> int:
        return 1

    def invoke(self, role_input: RoleInput) -> CleanState:
        obs = role_input.payload.get("observation", {})
        previous = role_input.payload.get("previous_clean")
        anomalies: list[str] = []

        regions = list(obs.get("regions", []))
        turn = obs.get("turn", 0)

        # Pass 1: clamp
        cleaned_regions = []
        for r in regions:
            clamped, flags = _clamp_region(r)
            cleaned_regions.append(clamped)
            anomalies.extend(flags)

        # Pass 2: consistency — I+R+D <= population
        for r in cleaned_regions:
            pop = r.get("population", 0)
            total = (
                r.get("infected", 0)
                + r.get("recovered", 0)
                + r.get("deceased", 0)
            )
            if pop > 0 and total > pop:
                scale = pop / total
                for k in ("infected", "recovered", "deceased"):
                    r[k] = int(r.get(k, 0) * scale)

        # Pass 3: staleness
        incidents = obs.get("incidents", [])
        fresh = [
            inc
            for inc in incidents
            if turn - inc.get("reported_turn", 0)
            <= _STALENESS_THRESHOLD
        ]
        if incidents and not fresh:
            anomalies.append("GAP:all_reports_stale")

        # Spikes
        if previous:
            anomalies.extend(
                _detect_spikes(cleaned_regions, previous)
            )

        # Contradictions
        signals = obs.get("stakeholder_signals", [])
        anomalies.extend(
            _detect_contradictions(signals, cleaned_regions)
        )

        # Salient changes
        changes: list[str] = []
        if previous:
            changes = _diff_observations(
                cleaned_regions, previous
            )

        cleaned_obs: dict[str, Any] = {
            "turn": turn,
            "regions": cleaned_regions,
            "incidents": fresh,
            "telemetry": obs.get("telemetry", {}),
            "resources": obs.get("resources", {}),
        }

        return CleanState(
            cleaned_observation=cleaned_obs,
            salient_changes=tuple(changes),
            flagged_anomalies=tuple(anomalies),
        )


def _clamp_region(
    region: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    r = dict(region)
    flags: list[str] = []
    pop = r.get("population", 0)
    rid = r.get("region_id", "unknown")

    for field in ("infected", "recovered", "deceased"):
        val = r.get(field, 0)
        if val is None or (
            isinstance(val, float) and val != val
        ):
            r[field] = 0
            flags.append(f"IMPOSSIBLE:{field}_nan in {rid}")
        elif val < 0:
            r[field] = 0
            flags.append(
                f"IMPOSSIBLE:{field}_negative in {rid}"
            )
        elif pop > 0 and val > pop:
            r[field] = pop
            flags.append(
                f"IMPOSSIBLE:{field}_exceeds_pop in {rid}"
            )

    return r, flags


def _detect_spikes(
    regions: list[dict[str, Any]],
    previous: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    cleaned = previous.get("cleaned_observation", previous)
    prev_r = cleaned.get("regions", cleaned)

    for r in regions:
        rid = r.get("region_id", "")
        cur = r.get("infected", 0)
        prev_data = (
            prev_r.get(rid, {})
            if isinstance(prev_r, dict)
            else {}
        )
        prev_val = prev_data.get("infected", 0)
        if prev_val > 0 and cur > 2 * prev_val:
            flags.append(f"SPIKE:{rid}")

    return flags


def _detect_contradictions(
    signals: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> list[str]:
    flags: list[str] = []
    total_pop = sum(r.get("population", 0) for r in regions)
    total_inf = sum(r.get("infected", 0) for r in regions)
    rate = total_inf / total_pop if total_pop > 0 else 0

    for sig in signals:
        if sig.get("urgency", 0) >= 0.8 and rate < 0.01:
            flags.append(
                f"CONTRADICTION:{sig.get('source', 'unknown')}"
            )

    return flags


def _diff_observations(
    regions: list[dict[str, Any]],
    previous: dict[str, Any],
) -> list[str]:
    changes: list[str] = []
    cleaned = previous.get("cleaned_observation", previous)
    prev_r = cleaned.get("regions", cleaned)

    for r in regions:
        rid = r.get("region_id", "")
        prev = (
            prev_r.get(rid, {})
            if isinstance(prev_r, dict)
            else {}
        )
        for field in ("infected", "recovered", "deceased"):
            cur = r.get(field, 0)
            prv = prev.get(field, 0)
            if cur != prv:
                changes.append(
                    f"{rid}.{field}: {prv} -> {cur}"
                )

    return changes
