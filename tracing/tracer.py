"""EpisodeTracer — append-only event buffer with finalization."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from schemas.episode import EpisodeTrace, LogEvent, TurnRecord


class TracerFinalizedError(Exception):
    """Raised when record() is called after finalization."""


class EpisodeTracer:
    """Buffers LogEvents and produces an EpisodeTrace on finalization.

    Implements LoggerProtocol.
    """

    def __init__(self, episode_id: str, seed: int = 0) -> None:
        self._episode_id = episode_id
        self._seed = seed
        self._events: list[LogEvent] = []
        self._finalized = False
        self._cached_trace: EpisodeTrace | None = None

    def set_episode(self, episode_id: str, seed: int = 0) -> None:
        """Set episode metadata. Must be called before any recording."""
        self._episode_id = episode_id
        self._seed = seed
        self._events.clear()
        self._finalized = False
        self._cached_trace = None

    def record(self, event: LogEvent) -> None:
        """Append a LogEvent. Raises TracerFinalizedError after finalize()."""
        if self._finalized:
            raise TracerFinalizedError(self._episode_id)
        self._events.append(event)

    def finalize(self) -> EpisodeTrace:
        """Close trace and return EpisodeTrace. Idempotent."""
        if self._cached_trace is not None:
            return self._cached_trace
        self._finalized = True
        turns = self._group_events_by_turn()
        self._cached_trace = EpisodeTrace(
            episode_id=self._episode_id,
            turns=tuple(turns),
            seed=self._seed,
        )
        return self._cached_trace

    def save(self, path: Path) -> Path:
        """Finalize and write trace to disk. Returns path."""
        from tracing.serializer import save_trace

        trace = self.finalize()
        return save_trace(trace, path)

    def flush(self) -> None:
        """Clear buffer without writing. No-op if finalized."""
        if not self._finalized:
            self._events.clear()

    def _group_events_by_turn(self) -> list[TurnRecord]:
        """Group buffered events by turn number into TurnRecords."""
        by_turn: dict[int, list[LogEvent]] = defaultdict(list)
        for ev in self._events:
            by_turn[ev.turn].append(ev)

        records: list[TurnRecord] = []
        for turn_num in sorted(by_turn):
            evts = by_turn[turn_num]
            obs = next((e.data for e in evts if e.kind == "observation"), None)
            act = next((e.data for e in evts if e.kind == "action"), None)
            rew = next((e.data for e in evts if e.kind == "reward"), None)
            bud = next(
                (e.data for e in evts if e.kind == "budget_snapshot"), None
            )
            arts = tuple(
                e.data for e in evts if e.kind.startswith("artifact")
            )
            records.append(
                TurnRecord(
                    turn=turn_num,
                    observation=obs,
                    action=act,
                    reward=rew,
                    budget_snapshot=bud,
                    artifacts=arts,
                    events=tuple(evts),
                )
            )
        return records
