"""AgentProtocol — contract for decision-making agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas.action import OuterAction
from src.schemas.observation import Observation


@runtime_checkable
class AgentProtocol(Protocol):
    """Policy that selects actions from observations."""

    def act(self, observation: Observation) -> OuterAction: ...

    def reset(self) -> None: ...
