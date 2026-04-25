"""Unit tests for src/logging/serializer.py — atomic trace I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from tracing.serializer import load_trace, save_trace
from schemas.episode import EpisodeTrace, LogEvent, TurnRecord


def _make_trace(episode_id: str = "ep-ser") -> EpisodeTrace:
    """Build a minimal valid EpisodeTrace for tests."""
    return EpisodeTrace(
        episode_id=episode_id,
        turns=(
            TurnRecord(
                turn=0,
                observation={"region": "north"},
                action={"kind": "noop"},
                events=(LogEvent(kind="observation", turn=0, data={}),),
            ),
        ),
        seed=42,
        condition="test",
    )


class TestSerializer:
    def test_round_trip(self, tmp_path: Path) -> None:
        trace = _make_trace()
        p = tmp_path / "trace.json"
        save_trace(trace, p)
        loaded = load_trace(p)
        assert loaded == trace

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        trace = _make_trace()
        deep = tmp_path / "a" / "b" / "c" / "trace.json"
        save_trace(trace, deep)
        assert deep.exists()
        loaded = load_trace(deep)
        assert loaded.episode_id == trace.episode_id

    def test_atomic_write_no_tmp_remains(self, tmp_path: Path) -> None:
        trace = _make_trace()
        p = tmp_path / "trace.json"
        save_trace(trace, p)
        tmp_file = p.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_trace(tmp_path / "nonexistent.json")

    def test_load_corrupted_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not valid json!!", encoding="utf-8")
        with pytest.raises(Exception):
            load_trace(p)

    def test_save_empty_trace(self, tmp_path: Path) -> None:
        trace = EpisodeTrace(episode_id="empty", turns=(), seed=0)
        p = tmp_path / "empty.json"
        save_trace(trace, p)
        loaded = load_trace(p)
        assert loaded == trace
        assert loaded.turns == ()

    def test_unicode_preservation(self, tmp_path: Path) -> None:
        trace = EpisodeTrace(
            episode_id="日本語テスト",
            turns=(
                TurnRecord(
                    turn=0,
                    observation={"msg": "感染者数🦠"},
                    events=(),
                ),
            ),
            seed=7,
            metadata={"emoji": "🔬"},
        )
        p = tmp_path / "unicode.json"
        save_trace(trace, p)
        loaded = load_trace(p)
        assert loaded.episode_id == "日本語テスト"
        assert loaded.turns[0].observation == {"msg": "感染者数🦠"}
        assert loaded.metadata == {"emoji": "🔬"}
