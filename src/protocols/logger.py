"""LoggerProtocol — contract for structured episode logging."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.schemas.episode import LogEvent


@runtime_checkable
class LoggerProtocol(Protocol):
    """Buffers and persists structured log events."""

    def record(self, event: LogEvent) -> None: ...

    def save(self, path: Path) -> Path: ...

    def flush(self) -> None: ...
