"""CriticRole — failure mode analysis and risk scoring."""

from __future__ import annotations

from typing import Any

from schemas.artifact import Critique, RoleInput

_BASE_RISK = 0.1


class CriticRole:
    """Analyze candidate actions for failure modes and risk."""

    @property
    def role_name(self) -> str:
        return "critic"

    @property
    def cost(self) -> int:
        return 2

    def invoke(self, role_input: RoleInput) -> Critique:
        candidate = role_input.payload.get("candidate", {})
        belief = role_input.payload.get("belief_state", {})
        constraints = role_input.payload.get("constraints", [])

        action = candidate.get("action", {})
        belief_confidence = belief.get("confidence", 0.5)

        failures: list[str] = []
        violations: list[str] = []
        amendments: list[str] = []

        # Resource failure
        rf = _check_resource_failure(action)
        failures.extend(rf)

        # Timing failure
        tf = _check_timing_failure(action, belief_confidence)
        failures.extend(tf)

        # Cascade failure
        cf = _check_cascade_failure(action, belief)
        failures.extend(cf)

        # Policy violations
        pv = _check_policy_violations(
            action, constraints, belief_confidence
        )
        violations.extend(pv)

        # Amendments
        for f in failures:
            amendments.append(f"AMEND: address {f}")
        for v in violations:
            amendments.append(f"AMEND: resolve {v}")

        risk = _compute_risk_score(
            len(failures), belief_confidence
        )

        return Critique(
            failure_modes=tuple(failures),
            policy_violations=tuple(violations),
            recommended_amendments=tuple(amendments),
            risk_score=risk,
        )


def _check_resource_failure(
    action: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if action.get("kind") != "deploy_resource":
        return flags

    amount = action.get("amount", 0)
    total = action.get("total_available", 0)
    if total > 0:
        ratio = amount / total
        if ratio > 1.0:
            flags.append("RESOURCE_IMPOSSIBLE")
        elif ratio > 0.8:
            flags.append("RESOURCE_EXHAUSTION")

    return flags


def _check_timing_failure(
    action: dict[str, Any], confidence: float
) -> list[str]:
    flags: list[str] = []
    kind = action.get("kind", "")
    level = action.get("level", 0)

    is_irreversible = (
        kind == "restrict_movement" and level >= 3
    )
    if confidence < 0.4 and is_irreversible:
        flags.append("LOW_CONFIDENCE_IRREVERSIBLE")

    return flags


def _check_cascade_failure(
    action: dict[str, Any],
    belief: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    kind = action.get("kind", "")

    if kind not in ("noop", "request_data"):
        return flags

    trajectories = belief.get("forecast_trajectories", ())
    for traj in trajectories:
        if traj.get("label") == "pessimistic":
            vals = traj.get("values", [])
            if vals and len(vals) >= 3 and vals[2] > 50:
                flags.append("INACTION_CASCADE")
                break

    return flags


def _check_policy_violations(
    action: dict[str, Any],
    constraints: list[Any],
    confidence: float,
) -> list[str]:
    flags: list[str] = []
    kind = action.get("kind", "")

    if kind == "escalate" and confidence > 0.7:
        flags.append("UNNECESSARY_ESCALATION")

    return flags


def _compute_risk_score(
    failure_count: int, confidence: float
) -> float:
    risk = _BASE_RISK
    risk += 0.2 * min(failure_count, 3)
    risk += (1 - confidence) * 0.3
    return max(0.0, min(1.0, risk))
