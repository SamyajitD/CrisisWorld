"""Episode schemas for tracing and results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LogEvent(BaseModel):
    """Single structured log event."""

    model_config = ConfigDict(frozen=True)

    kind: str
    turn: int = Field(ge=0)
    data: dict[str, Any] = {}


class TurnRecord(BaseModel):
    """Complete record of one turn."""

    model_config = ConfigDict(frozen=True)

    turn: int = Field(ge=0)
    observation: dict[str, Any] | None = None
    action: dict[str, Any] | None = None
    reward: dict[str, Any] | None = None
    budget_snapshot: dict[str, Any] | None = None
    artifacts: tuple[dict[str, Any], ...] = ()
    events: tuple[LogEvent, ...] = ()


class EpisodeTrace(BaseModel):
    """Full episode trace (sequence of TurnRecords)."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    turns: tuple[TurnRecord, ...] = ()
    seed: int
    condition: str = ""
    metadata: dict[str, Any] = {}


class EpisodeResult(BaseModel):
    """Summary result of one completed episode."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    seed: int = Field(ge=0)
    condition: str
    total_turns: int = Field(ge=0)
    total_reward: float
    termination_reason: str
    metrics: dict[str, Any] = {}


class MemoryDigest(BaseModel):
    """Summarized view of episode memory state."""

    model_config = ConfigDict(frozen=True)

    num_entries: int = Field(default=0, ge=0)
    keys: tuple[str, ...] = ()
    summary: dict[str, Any] = {}
