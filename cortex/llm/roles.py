"""Generic LLM-backed role — one class for all 5 Cortex roles."""

from __future__ import annotations

import logging
from typing import Any

from protocols.role import RoleProtocol
from schemas.artifact import (
    Artifact,
    BeliefState,
    CleanState,
    Critique,
    ExecutiveDecision,
    Plan,
    RoleInput,
)

from cortex.llm.prompts import ROLE_PROMPTS
from cortex.llm.provider import HuggingFaceProvider

_log = logging.getLogger(__name__)

# Map role names to their artifact types
_ARTIFACT_TYPES: dict[str, type] = {
    "perception": CleanState,
    "world_modeler": BeliefState,
    "planner": Plan,
    "critic": Critique,
    "executive": ExecutiveDecision,
}

_ROLE_COSTS: dict[str, int] = {
    "perception": 1,
    "world_modeler": 2,
    "planner": 2,
    "critic": 2,
    "executive": 1,
}


class LLMRole:
    """Generic LLM-backed role implementing RoleProtocol.

    Uses HuggingFaceProvider to call the model, parses JSON output
    into the expected Pydantic artifact type. Falls back to a heuristic
    role on any failure.
    """

    def __init__(
        self,
        name: str,
        provider: HuggingFaceProvider,
        fallback: RoleProtocol,
    ) -> None:
        if name not in ROLE_PROMPTS:
            raise ValueError(f"Unknown role: {name}. Valid: {list(ROLE_PROMPTS.keys())}")
        self._name = name
        self._provider = provider
        self._fallback = fallback
        self._system_prompt, self._build_user = ROLE_PROMPTS[name]
        self._artifact_type = _ARTIFACT_TYPES[name]
        self._cost = _ROLE_COSTS[name]

    @property
    def role_name(self) -> str:
        return self._name

    @property
    def cost(self) -> int:
        return self._cost

    def invoke(self, role_input: RoleInput) -> Artifact:
        user_prompt = self._build_user(role_input.payload)
        try:
            raw = self._provider.complete(self._system_prompt, user_prompt)
            artifact = self._artifact_type.model_validate(raw)
            _log.debug("LLM %s produced valid %s", self._name, type(artifact).__name__)
            return artifact
        except Exception as exc:
            self._provider.total_fallbacks += 1
            _log.warning(
                "LLM %s failed (%s: %s), using heuristic fallback",
                self._name, type(exc).__name__, exc,
            )
            return self._fallback.invoke(role_input)
