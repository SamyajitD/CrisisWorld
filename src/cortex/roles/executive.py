"""ExecutiveRole — decision tree for deliberation control."""

from __future__ import annotations

from typing import Any

from src.schemas.artifact import ExecutiveDecision, RoleInput


class ExecutiveRole:
    """Evaluate artifacts and decide: act, call, wait, escalate, stop."""

    @property
    def role_name(self) -> str:
        return "executive"

    @property
    def cost(self) -> int:
        return 1

    def invoke(self, role_input: RoleInput) -> ExecutiveDecision:
        artifacts = role_input.payload.get("artifacts", [])
        budget = role_input.payload.get("budget_status", {})

        remaining = budget.get("remaining", 0)

        clean = _find_artifact(artifacts, "clean")
        belief = _find_artifact(artifacts, "belief")
        plan = _find_artifact(artifacts, "plan")
        critique = _find_artifact(artifacts, "critique")

        # 1. Budget exhausted
        if remaining <= 0:
            candidate = _best_candidate(plan)
            if candidate:
                return ExecutiveDecision(
                    decision="act",
                    target_action=candidate,
                    reasoning="Budget exhausted, acting now",
                )
            return ExecutiveDecision(
                decision="act",
                target_action={"kind": "noop"},
                reasoning="Budget exhausted, no plan, NoOp",
            )

        # 2. Budget critical
        if remaining <= 2:
            if plan:
                candidate = _best_candidate(plan)
                if candidate:
                    return ExecutiveDecision(
                        decision="act",
                        target_action=candidate,
                        reasoning="Budget critical, acting",
                    )
            return ExecutiveDecision(
                decision="call",
                target_role="planner",
                reasoning="Budget critical, need plan",
            )

        # 3. No CleanState
        if not clean:
            return ExecutiveDecision(
                decision="call",
                target_role="perception",
                reasoning="No perception data, safety net",
            )

        # 4. No BeliefState + anomalies
        anomalies = _get_anomalies(clean)
        if not belief and anomalies:
            return ExecutiveDecision(
                decision="call",
                target_role="world_modeler",
                reasoning="Anomalies detected, need model",
            )

        # 5. No Plan
        if not plan:
            return ExecutiveDecision(
                decision="call",
                target_role="planner",
                reasoning="No plan available",
            )

        # 6. Plan but no Critique
        if plan and not critique:
            candidate = _best_candidate(plan)
            conf = (candidate or {}).get("confidence", 0)
            if conf >= 0.8 and remaining <= 4:
                return ExecutiveDecision(
                    decision="act",
                    target_action=candidate,
                    reasoning="High confidence, low budget",
                )
            return ExecutiveDecision(
                decision="call",
                target_role="critic",
                reasoning="Plan needs critique",
            )

        # 7. Plan + Critique
        if plan and critique:
            risk = critique.get("risk_score", 0.5)
            if risk < 0.3:
                candidate = _best_candidate(plan)
                return ExecutiveDecision(
                    decision="act",
                    target_action=candidate,
                    reasoning=f"Low risk ({risk:.2f}), acting",
                )
            if risk >= 0.7:
                return ExecutiveDecision(
                    decision="escalate",
                    target_action={"kind": "escalate"},
                    reasoning=f"High risk ({risk:.2f})",
                )
            if remaining >= 4:
                return ExecutiveDecision(
                    decision="call",
                    target_role="planner",
                    reasoning="Medium risk, re-planning",
                )

        # 8. Fallback
        if remaining > 0:
            return ExecutiveDecision(
                decision="wait",
                reasoning="Waiting for more information",
            )
        return ExecutiveDecision(
            decision="act",
            target_action={"kind": "noop"},
            reasoning="Fallback NoOp",
        )


def _find_artifact(
    artifacts: list[Any], prefix: str
) -> dict[str, Any] | None:
    """Find the most recent artifact matching a type prefix."""
    result = None
    for a in artifacts:
        if not isinstance(a, dict):
            continue
        # Match by known keys
        if prefix == "clean" and "cleaned_observation" in a:
            result = a
        elif prefix == "belief" and "forecast_trajectories" in a:
            result = a
        elif prefix == "plan" and "candidates" in a:
            result = a
        elif prefix == "critique" and "risk_score" in a:
            result = a
    return result


def _best_candidate(
    plan: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not plan:
        return None
    candidates = plan.get("candidates", ())
    if not candidates:
        return None
    # Already sorted by confidence (planner guarantees this)
    return candidates[0].get("action") if isinstance(
        candidates[0], dict
    ) else None


def _get_anomalies(
    clean: dict[str, Any] | None,
) -> tuple[str, ...]:
    if not clean:
        return ()
    raw = clean.get("flagged_anomalies", ())
    return tuple(raw) if raw else ()
