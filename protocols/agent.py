"""AgentProtocol — contract for decision-making agents."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from models import ActionUnion, Observation


@runtime_checkable
class AgentProtocol(Protocol):
    """Policy that selects actions from observations."""

    def act(self, observation: Observation) -> ActionUnion: ...

    def reset(self) -> None: ...
