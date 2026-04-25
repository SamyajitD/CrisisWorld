"""Typed client for connecting to a remote CrisisWorld server."""

from __future__ import annotations

from typing import Any

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import ActionUnion, CrisisState, Observation


class CrisisWorldClient(EnvClient[ActionUnion, Observation, CrisisState]):
    """Typed client for remote CrisisWorld env."""

    def _step_payload(self, action: ActionUnion) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Observation]:
        obs = Observation.model_validate(payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> CrisisState:
        return CrisisState.model_validate(payload)
