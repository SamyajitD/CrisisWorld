"""RoleProtocol — contract for Cortex deliberation roles."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.schemas.artifact import Artifact, RoleInput


@runtime_checkable
class RoleProtocol(Protocol):
    """Single deliberation role with typed input/output and budget cost."""

    @property
    def role_name(self) -> str: ...

    @property
    def cost(self) -> int: ...

    def invoke(self, role_input: RoleInput) -> Artifact: ...
