"""Canonical route labels and export helpers for router SFT."""

from __future__ import annotations

from typing import Any

ROUTE_LABELS = (
    "act",
    "call_critic",
    "call_planner",
    "call_world_modeler",
    "escalate",
    "stop",
    "wait",
)

_CALL_ROUTE_MAP = {
    "critic": "call_critic",
    "planner": "call_planner",
    "world_modeler": "call_world_modeler",
}


def canonical_route_label(decision: dict[str, Any]) -> str:
    """Map a raw ExecutiveDecision dict to a fixed route label."""
    choice = decision.get("decision")
    if choice == "call":
        target = decision.get("target_role")
        if target not in _CALL_ROUTE_MAP:
            raise ValueError(f"Unsupported target_role for router label: {target!r}")
        return _CALL_ROUTE_MAP[target]
    if choice not in ROUTE_LABELS:
        raise ValueError(f"Unsupported decision for router label: {choice!r}")
    return str(choice)


def normalize_sft_target(decision: dict[str, Any]) -> dict[str, Any]:
    """Normalize a teacher decision to a stable JSON target for SFT."""
    normalized = {
        "decision": decision.get("decision"),
        "target_action": decision.get("target_action"),
        "target_role": decision.get("target_role"),
        "reasoning": decision.get("reasoning", ""),
    }

    if normalized["decision"] == "call" and normalized["target_role"] is None:
        raise ValueError("call decision is missing target_role")
    if normalized["decision"] == "act" and normalized["target_action"] is None:
        normalized["target_action"] = {"kind": "noop"}
    return normalized
