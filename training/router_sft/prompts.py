"""Prompt templates for router-only supervised fine-tuning."""

from __future__ import annotations

import json
from typing import Any

ROUTER_SYSTEM_PROMPT = """You are the EXECUTIVE router for CrisisWorld.

Your only job is to choose the next deliberation control decision.

Allowed decisions:
- "act": commit to a concrete outer action in target_action
- "call": request exactly one role in target_role
- "wait": do nothing this inner iteration
- "escalate": escalate to external authority
- "stop": stop deliberation

Allowed target_role values:
- "world_modeler"
- "planner"
- "critic"

Respond with ONLY one valid JSON object with this schema:
{
  "decision": "call",
  "target_action": null,
  "target_role": "critic",
  "reasoning": "Short justification"
}
"""


def build_router_user_prompt(payload: dict[str, Any]) -> str:
    """Render the exact router payload into a compact prompt."""
    artifacts = payload.get("artifacts", [])
    budget = payload.get("budget_status", {})

    lines = [
        "Budget status:",
        json.dumps(budget, sort_keys=True),
        "",
        f"Artifacts produced this turn ({len(artifacts)} total, last 5 shown):",
    ]
    for artifact in artifacts[-5:]:
        lines.append(json.dumps(artifact, sort_keys=True, default=str)[:500])
    return "\n".join(lines)
