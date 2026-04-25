"""MemoryProtocol — contract for episode memory storage."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas.artifact import Artifact
from ..schemas.episode import MemoryDigest


@runtime_checkable
class MemoryProtocol(Protocol):
    """Append-only keyed artifact store for episode memory."""

    def store(self, key: str, artifact: Artifact) -> None: ...

    def retrieve(self, key: str) -> list[Artifact]: ...

    def digest(self) -> MemoryDigest: ...

    def clear(self) -> None: ...
