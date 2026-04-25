"""Prompt templates for LLM-backed Cortex roles."""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

SCENARIO_CONTEXT = """You are part of a crisis management AI system responding to a disease outbreak.
The world has multiple regions, each with population, infected, recovered, and deceased counts.
You have resources (medical, personnel, funding) that decay over time.
Valid action kinds: deploy_resource, restrict_movement, request_data, public_communication, escalate, reallocate_budget, noop."""


def _summarize_obs(obs: dict[str, Any]) -> str:
    """Render observation data as concise text."""
    parts = [f"Turn: {obs.get('turn', '?')}"]
    regions = obs.get("regions", [])
    if isinstance(regions, (list, tuple)):
        for r in regions[:5]:
            if isinstance(r, dict):
                parts.append(
                    f"  {r.get('region_id','?')}: pop={r.get('population',0)} "
                    f"inf={r.get('infected',0)} rec={r.get('recovered',0)} "
                    f"dec={r.get('deceased',0)} restricted={r.get('restricted',False)}"
                )
    res = obs.get("resources", {})
    if isinstance(res, dict):
        parts.append(f"Resources: medical={res.get('medical',0)} personnel={res.get('personnel',0)} funding={res.get('funding',0)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-role prompts
# ---------------------------------------------------------------------------

PERCEPTION_SYSTEM = f"""{SCENARIO_CONTEXT}

You are the PERCEPTION module. Your job:
1. Clean noisy observation data (clamp negatives to 0, ensure infected+recovered+deceased <= population)
2. Detect anomalies (infection spikes, contradictions between signals and data)
3. Identify salient changes from the previous turn

Respond with ONLY a valid JSON object matching this schema:
{{
  "salient_changes": ["list of change descriptions"],
  "flagged_anomalies": ["list of anomaly codes like SPIKE:r0 or CONTRADICTION:hospital"],
  "cleaned_observation": {{"turn": N, "regions": [...], "resources": {{...}}}}
}}"""


def perception_user(payload: dict[str, Any]) -> str:
    obs = payload.get("observation", {})
    prev = payload.get("previous_clean")
    text = f"Current observation:\n{_summarize_obs(obs)}"
    if prev:
        text += f"\n\nPrevious clean state (for change detection):\n{json.dumps(prev, default=str)[:500]}"
    return text


WORLD_MODELER_SYSTEM = f"""{SCENARIO_CONTEXT}

You are the WORLD MODELER. Your job:
1. Estimate hidden variables: true_infected_multiplier (how many real cases per reported), spread_rate, resource_depletion_rate
2. Generate 3 forecast trajectories for the next 5 turns: optimistic, baseline, pessimistic
3. Assess your confidence (0.1 to 1.0)

Respond with ONLY a valid JSON object:
{{
  "hidden_var_estimates": {{"true_infected_multiplier": 2.0, "spread_rate": 1.1, "resource_depletion_rate": 0.02}},
  "forecast_trajectories": [
    {{"label": "optimistic", "values": [100, 90, 80, 70, 60]}},
    {{"label": "baseline", "values": [100, 110, 121, 133, 146]}},
    {{"label": "pessimistic", "values": [100, 140, 196, 274, 384]}}
  ],
  "confidence": 0.7
}}"""


def world_modeler_user(payload: dict[str, Any]) -> str:
    clean = payload.get("clean_state", {})
    prev_inf = payload.get("prev_infected")
    text = f"Cleaned state:\n{json.dumps(clean, default=str)[:800]}"
    if prev_inf is not None:
        text += f"\n\nPrevious total infected: {prev_inf}"
    return text


PLANNER_SYSTEM = f"""{SCENARIO_CONTEXT}

You are the PLANNER. Your job:
1. Generate 1-3 candidate actions based on the current belief state
2. Each candidate has: action (dict with "kind" and params), rationale, expected_effect, confidence (0-1)
3. Rank by confidence descending
4. Always include at least one candidate

Valid action examples:
- {{"kind": "deploy_resource", "resource": "medical", "region_id": "r0", "amount": 10}}
- {{"kind": "restrict_movement", "region_id": "r0", "level": 1}}
- {{"kind": "request_data", "source": "telemetry"}}
- {{"kind": "noop"}}

Respond with ONLY a valid JSON object:
{{
  "candidates": [
    {{"action": {{"kind": "deploy_resource", "resource": "medical", "region_id": "r0", "amount": 10}}, "rationale": "High infection in r0", "expected_effect": "Reduce infection rate", "confidence": 0.8}},
    {{"action": {{"kind": "noop"}}, "rationale": "Wait and observe", "expected_effect": "No change", "confidence": 0.2}}
  ]
}}"""


def planner_user(payload: dict[str, Any]) -> str:
    belief = payload.get("belief_state", {})
    constraints = payload.get("constraints", [])
    text = f"Belief state:\n{json.dumps(belief, default=str)[:800]}"
    if constraints:
        text += f"\n\nActive constraints: {json.dumps(constraints, default=str)[:300]}"
    return text


CRITIC_SYSTEM = f"""{SCENARIO_CONTEXT}

You are the CRITIC. Your job:
1. Analyze the proposed action for failure modes (resource exhaustion, timing failures, cascade risks)
2. Check for policy violations
3. Compute a risk score (0.0 = safe, 1.0 = very risky)
4. Suggest amendments

Respond with ONLY a valid JSON object:
{{
  "failure_modes": ["RESOURCE_EXHAUSTION", "LOW_CONFIDENCE_IRREVERSIBLE"],
  "policy_violations": ["UNNECESSARY_ESCALATION"],
  "recommended_amendments": ["AMEND: deploy to highest-infection region instead"],
  "risk_score": 0.4
}}"""


def critic_user(payload: dict[str, Any]) -> str:
    candidate = payload.get("candidate", {})
    belief = payload.get("belief_state", {})
    text = f"Candidate action:\n{json.dumps(candidate, default=str)[:500]}"
    text += f"\n\nBelief state:\n{json.dumps(belief, default=str)[:500]}"
    return text


EXECUTIVE_SYSTEM = f"""{SCENARIO_CONTEXT}

You are the EXECUTIVE. You decide what to do next based on all artifacts produced so far.

Decisions:
- "act": Commit to an action. Set target_action to the action dict (e.g. {{"kind": "deploy_resource", ...}})
- "call": Request another role. Set target_role to one of: world_modeler, planner, critic
- "wait": Do nothing this iteration
- "escalate": Call external authority
- "stop": End deliberation

Budget rules: If remaining budget <= 2, you MUST act. If remaining > 6, you may call more roles.

Respond with ONLY a valid JSON object:
{{
  "decision": "act",
  "target_action": {{"kind": "deploy_resource", "resource": "medical", "region_id": "r0", "amount": 10}},
  "target_role": null,
  "reasoning": "Low budget, best plan has acceptable risk"
}}"""


def executive_user(payload: dict[str, Any]) -> str:
    artifacts = payload.get("artifacts", [])
    budget = payload.get("budget_status", {})
    text = f"Budget: remaining={budget.get('remaining', '?')}, spent={budget.get('spent', '?')}, total={budget.get('total', '?')}"
    text += f"\n\nArtifacts produced this turn ({len(artifacts)}):"
    for a in artifacts[-5:]:  # last 5 to keep prompt short
        text += f"\n{json.dumps(a, default=str)[:200]}"
    return text


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

ROLE_PROMPTS: dict[str, tuple[str, Any]] = {
    "perception": (PERCEPTION_SYSTEM, perception_user),
    "world_modeler": (WORLD_MODELER_SYSTEM, world_modeler_user),
    "planner": (PLANNER_SYSTEM, planner_user),
    "critic": (CRITIC_SYSTEM, critic_user),
    "executive": (EXECUTIVE_SYSTEM, executive_user),
}
