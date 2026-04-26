"""SingleLLMAgent — one model maps observation directly to an outer action."""

from __future__ import annotations

import json
import logging

from ..models import NoOp, Observation, OuterAction

_log = logging.getLogger(__name__)

_POLICY_SYSTEM = (
    "You are managing a disease outbreak across multiple regions. "
    "Each region has population, infected, recovered, and deceased counts. "
    "You have medical, personnel, and funding resources that decay over time.\n\n"
    "Given the current observation, choose exactly one action.\n\n"
    "Valid action kinds: deploy_resource, restrict_movement, request_data, "
    "public_communication, escalate, reallocate_budget, noop.\n\n"
    "Respond with ONLY valid JSON: {\"kind\": \"...\", ...action_params}"
)


def _summarize_obs(obs: Observation) -> str:
    """Render observation as concise text for the LLM prompt."""
    parts = [f"Turn: {obs.turn}"]
    for r in obs.regions[:5]:
        parts.append(
            f"  {r.region_id}: pop={r.population} inf={r.infected} "
            f"rec={r.recovered} dec={r.deceased} restricted={r.restricted}"
        )
    if len(obs.regions) > 5:
        parts.append(f"  (+{len(obs.regions) - 5} more regions)")
    parts.append(
        f"Resources: medical={obs.resources.medical} "
        f"personnel={obs.resources.personnel} funding={obs.resources.funding}"
    )
    parts.append(
        f"Telemetry: infected={obs.telemetry.total_infected} "
        f"recovered={obs.telemetry.total_recovered} deceased={obs.telemetry.total_deceased}"
    )
    for sig in obs.stakeholder_signals[:3]:
        parts.append(f"Signal: {sig.source} urgency={sig.urgency:.2f} '{sig.message}'")
    return "\n".join(parts)


class SingleLLMAgent:
    """Direct obs->action policy via a single LLM call. No deliberation."""

    def __init__(self, provider: object) -> None:
        """provider must have a .complete(system, user) -> dict method."""
        self._provider = provider
        self._turn_count = 0

    def act(self, observation: Observation) -> OuterAction:
        if observation.done:
            return NoOp()

        user_prompt = _summarize_obs(observation)
        try:
            raw = self._provider.complete(_POLICY_SYSTEM, user_prompt)
            from pydantic import TypeAdapter
            from ..models import ActionUnion
            adapter = TypeAdapter(ActionUnion)
            action = adapter.validate_python(raw)
            self._turn_count += 1
            return action
        except Exception as exc:
            _log.warning("SingleLLMAgent failed (%s), returning NoOp", exc)
            self._turn_count += 1
            return NoOp()

    def reset(self) -> None:
        self._turn_count = 0
