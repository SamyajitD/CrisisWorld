"""EnvProtocol — contract for outbreak simulation environments."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas.action import OuterAction
from src.schemas.observation import Observation
from src.schemas.state import StepResult


@runtime_checkable
class EnvProtocol(Protocol):
    """Stateful outbreak simulator with reset/step/close lifecycle."""

    def reset(self, seed: int) -> Observation: ...

    def step(self, action: OuterAction) -> StepResult: ...

    def close(self) -> None: ...
