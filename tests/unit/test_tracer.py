"""Unit tests for src/logging/tracer.py — EpisodeTracer."""

from __future__ import annotations

import pytest

from schemas.episode import EpisodeTrace, LogEvent


class TestEpisodeTracer:
    """Tests for EpisodeTracer record/finalize/save lifecycle."""

    def test_record_appends_events_in_order(self) -> None:
        from tracing.tracer import EpisodeTracer

        tracer = EpisodeTracer("ep-001")
        events = [
            LogEvent(kind="observation", turn=0, data={"a": 1}),
            LogEvent(kind="action", turn=0, data={"b": 2}),
            LogEvent(kind="reward", turn=0, data={"c": 3}),
            LogEvent(kind="observation", turn=1, data={"d": 4}),
            LogEvent(kind="action", turn=1, data={"e": 5}),
        ]
        for ev in events:
            tracer.record(ev)

        trace = tracer.finalize()
        all_events = [ev for tr in trace.turns for ev in tr.events]
        assert len(all_events) == 5
        assert [ev.kind for ev in all_events] == [
            "observation",
            "action",
            "reward",
            "observation",
            "action",
        ]

    def test_finalize_returns_episode_trace(self) -> None:
        from tracing.tracer import EpisodeTracer

        tracer = EpisodeTracer("ep-002")
        tracer.record(LogEvent(kind="observation", turn=0, data={}))
        trace = tracer.finalize()
        assert isinstance(trace, EpisodeTrace)
        assert trace.episode_id == "ep-002"

    def test_finalize_is_idempotent(self) -> None:
        from tracing.tracer import EpisodeTracer

        tracer = EpisodeTracer("ep-003")
        tracer.record(LogEvent(kind="observation", turn=0, data={}))
        trace1 = tracer.finalize()
        trace2 = tracer.finalize()
        assert trace1 is trace2

    def test_record_after_finalize_raises(self) -> None:
        from tracing.tracer import EpisodeTracer, TracerFinalizedError

        tracer = EpisodeTracer("ep-004")
        tracer.finalize()
        with pytest.raises(TracerFinalizedError, match="ep-004"):
            tracer.record(LogEvent(kind="observation", turn=0, data={}))

    def test_finalize_empty_trace(self) -> None:
        from tracing.tracer import EpisodeTracer

        tracer = EpisodeTracer("ep-005")
        trace = tracer.finalize()
        assert isinstance(trace, EpisodeTrace)
        assert trace.turns == ()

    def test_large_event_count(self) -> None:
        from tracing.tracer import EpisodeTracer

        tracer = EpisodeTracer("ep-006")
        for i in range(10_000):
            tracer.record(LogEvent(kind="tick", turn=i, data={"i": i}))

        trace = tracer.finalize()
        total_events = sum(len(tr.events) for tr in trace.turns)
        assert total_events == 10_000

    def test_save_delegates_to_serializer(self, tmp_path: object) -> None:
        from pathlib import Path

        from tracing.serializer import load_trace
        from tracing.tracer import EpisodeTracer

        p = Path(str(tmp_path)) / "trace.json"
        tracer = EpisodeTracer("ep-007")
        tracer.record(LogEvent(kind="observation", turn=0, data={"x": 1}))
        result = tracer.save(p)
        assert result == p
        assert p.exists()
        loaded = load_trace(p)
        assert loaded.episode_id == "ep-007"

    def test_save_finalizes_implicitly(self, tmp_path: object) -> None:
        from pathlib import Path

        from tracing.tracer import EpisodeTracer, TracerFinalizedError

        p = Path(str(tmp_path)) / "trace.json"
        tracer = EpisodeTracer("ep-008")
        tracer.record(LogEvent(kind="observation", turn=0, data={}))
        tracer.save(p)
        with pytest.raises(TracerFinalizedError):
            tracer.record(LogEvent(kind="action", turn=0, data={}))
