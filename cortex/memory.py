"""EpisodeMemory — implements MemoryProtocol for keyed artifact storage."""

from __future__ import annotations

from ..schemas.artifact import Artifact
from ..schemas.episode import MemoryDigest


class EpisodeMemory:
    """Append-only keyed artifact store for episode memory."""

    def __init__(self) -> None:
        self._store: dict[str, list[Artifact]] = {}

    def store(self, key: str, artifact: Artifact) -> None:
        if not key:
            raise ValueError("key must be non-empty")
        self._store.setdefault(key, []).append(artifact)

    def retrieve(self, key: str) -> list[Artifact]:
        return list(self._store.get(key, []))

    def digest(self) -> MemoryDigest:
        total = sum(len(v) for v in self._store.values())
        keys = tuple(sorted(self._store.keys()))
        summary = {k: len(v) for k, v in self._store.items()}
        return MemoryDigest(
            num_entries=total,
            keys=keys,
            summary=summary,
        )

    def clear(self) -> None:
        self._store = {}

    def reset(self) -> None:
        self.clear()


class NullMemory:
    """No-op memory that satisfies MemoryProtocol but discards all stores."""

    def store(self, key: str, artifact: object) -> None:
        pass

    def retrieve(self, key: str) -> list:
        return []

    def digest(self) -> MemoryDigest:
        return MemoryDigest()

    def clear(self) -> None:
        pass

    def reset(self) -> None:
        pass
