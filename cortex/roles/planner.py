"""PlannerRole — candidate action generation and ranking."""

from __future__ import annotations

from typing import Any

from ...schemas.artifact import CandidateAction, Plan, RoleInput

_DEFAULT_MAX_CANDIDATES = 3


class PlannerRole:
    """Enumerate, score, and rank candidate actions."""

    @property
    def role_name(self) -> str:
        return "planner"

    @property
    def cost(self) -> int:
        return 2

    def invoke(self, role_input: RoleInput) -> Plan:
        belief = role_input.payload.get("belief_state", {})
        goals = role_input.payload.get("goals", [])
        constraints = role_input.payload.get("constraints", [])
        max_cands = role_input.payload.get(
            "max_candidates", _DEFAULT_MAX_CANDIDATES
        )

        confidence = belief.get("confidence", 0.5)
        trajectories = belief.get("forecast_trajectories", ())

        # If belief is empty/low confidence, prioritize data
        if not trajectories or confidence < 0.3:
            return Plan(
                candidates=(
                    CandidateAction(
                        action={"kind": "request_data", "source": "telemetry"},
                        rationale="Low confidence, need more data",
                        expected_effect="Improve situational awareness",
                        confidence=0.6,
                    ),
                    _noop_candidate(0.1),
                )
            )

        # Enumerate candidates
        raw = _enumerate_actions(belief, goals)

        # Prune by constraints
        active_names = {
            c.get("name", "")
            for c in constraints
            if c.get("active", True)
        }
        pruned = [
            c for c in raw if not _is_constrained(c, active_names)
        ]

        # Score and sort
        scored = [
            (c, _score_candidate(c, belief)) for c in pruned
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Top N
        top = scored[:max_cands]
        candidates = [
            CandidateAction(
                action=c["action"],
                rationale=c.get("rationale", ""),
                expected_effect=c.get("expected_effect", ""),
                confidence=round(score, 2),
            )
            for c, score in top
        ]

        # Ensure at least 1 candidate (NoOp fallback)
        if not candidates:
            candidates = [_noop_candidate(0.1)]

        return Plan(candidates=tuple(candidates))


def _enumerate_actions(
    belief: dict[str, Any],
    goals: list[Any],
) -> list[dict[str, Any]]:
    """Generate applicable actions from belief state."""
    candidates: list[dict[str, Any]] = []

    # Extract region data from belief (via artifacts chain)
    regions = belief.get("cleaned_observation", {}).get("regions", [])

    if regions:
        # Sort by infected count descending, target highest-infection regions
        sorted_regions = sorted(
            regions,
            key=lambda r: r.get("infected", 0),
            reverse=True,
        )
        for reg in sorted_regions[:2]:
            rid = reg.get("region_id", "unknown")
            # Deploy resource
            candidates.append({
                "action": {
                    "kind": "deploy_resource",
                    "resource": "medical",
                    "region_id": rid,
                    "amount": 10,
                },
                "rationale": f"Deploy medical resources to {rid}",
                "expected_effect": "Reduce infection rate",
            })
            # Restrict movement
            candidates.append({
                "action": {
                    "kind": "restrict_movement",
                    "region_id": rid,
                    "level": 1,
                },
                "rationale": f"Restrict movement in {rid} to contain spread",
                "expected_effect": "Slow transmission",
            })

    # Request data (always available)
    candidates.append({
        "action": {"kind": "request_data", "source": "telemetry"},
        "rationale": "Improve data quality",
        "expected_effect": "Better situational awareness",
    })

    # Escalate if pessimistic
    trajectories = belief.get("forecast_trajectories", ())
    for traj in trajectories:
        if traj.get("label") == "pessimistic":
            vals = traj.get("values", [])
            if vals and vals[-1] > 100:
                candidates.append({
                    "action": {
                        "kind": "escalate",
                        "agency": "national_health",
                    },
                    "rationale": "Severe pessimistic forecast",
                    "expected_effect": "External intervention",
                })
                break

    # NoOp always available
    candidates.append({
        "action": {"kind": "noop"},
        "rationale": "Wait and observe",
        "expected_effect": "No intervention",
    })

    return candidates


def _is_constrained(
    candidate: dict[str, Any],
    active_constraints: set[str],
) -> bool:
    kind = candidate.get("action", {}).get("kind", "")
    constraint_map = {
        "deploy_resource": "no_deploy",
        "restrict_movement": "no_restrict",
        "request_data": "no_request",
        "escalate": "no_escalate",
    }
    constraint_name = constraint_map.get(kind, "")
    return constraint_name in active_constraints


def _score_candidate(
    candidate: dict[str, Any],
    belief: dict[str, Any],
) -> float:
    kind = candidate.get("action", {}).get("kind", "")
    confidence = belief.get("confidence", 0.5)

    urgency = 0.5
    efficiency = 0.5
    alignment = 0.5

    if kind == "deploy_resource":
        urgency = 0.8
        efficiency = 0.6
        alignment = confidence
    elif kind == "restrict_movement":
        urgency = 0.7
        efficiency = 0.5
        alignment = confidence * 0.9
    elif kind == "request_data":
        urgency = 0.3
        efficiency = 0.8
        alignment = 1.0 - confidence
    elif kind == "escalate":
        urgency = 0.9
        efficiency = 0.3
        alignment = 0.7
    elif kind == "noop":
        urgency = 0.1
        efficiency = 1.0
        alignment = 0.2

    return 0.4 * urgency + 0.3 * efficiency + 0.3 * alignment


def _noop_candidate(confidence: float) -> CandidateAction:
    return CandidateAction(
        action={"kind": "noop"},
        rationale="No viable action",
        expected_effect="No change",
        confidence=confidence,
    )
