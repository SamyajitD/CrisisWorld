"""EnvProtocol — contract for outbreak simulation environments."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..models import ActionUnion, CrisisState, EnvironmentMetadata, Observation


@runtime_checkable
class EnvProtocol(Protocol):
    """Stateful outbreak simulator with reset/step/state/close lifecycle."""

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation: ...

    def step(
        self,
        action: ActionUnion,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation: ...

    @property
    def state(self) -> CrisisState: ...

    def get_metadata(self) -> EnvironmentMetadata: ...

    def close(self) -> None: ...
