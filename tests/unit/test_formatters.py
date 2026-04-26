"""Unit tests for src/logging/formatters.py — human-readable rendering."""

from __future__ import annotations

from CrisisWorld.schemas.episode import EpisodeTrace, LogEvent, TurnRecord


def _trace_with_turns() -> EpisodeTrace:
    """Build a trace with 3 turns for formatting tests."""
    return EpisodeTrace(
        episode_id="ep-fmt-001",
        turns=(
            TurnRecord(
                turn=0,
                observation={
                    "region_north": {"infected": 120},
                    "region_south": {"infected": 45},
                },
                action={
                    "kind": "deploy_resource",
                    "region_id": "north",
                    "amount": 50,
                },
                reward={
                    "total": -0.32,
                    "outcome": 0.1,
                    "timeliness": -0.2,
                    "safety": 0.0,
                },
                budget_snapshot={"total": 30, "spent": 12, "remaining": 18},
                artifacts=(
                    {"role": "perception", "anomalies": 2},
                    {"role": "planner", "candidates": 3},
                ),
                events=(LogEvent(kind="observation", turn=0, data={}),),
            ),
            TurnRecord(
                turn=1,
                observation={"region_north": {"infected": 90}},
                action={"kind": "restrict_movement", "region_id": "north"},
                reward={"total": 0.15},
                budget_snapshot={"total": 30, "spent": 16, "remaining": 14},
                artifacts=(),
                events=(LogEvent(kind="observation", turn=1, data={}),),
            ),
            TurnRecord(
                turn=2,
                observation={"region_north": {"infected": 50}},
                action={"kind": "noop"},
                reward={"total": 0.40},
                budget_snapshot={"total": 30, "spent": 20, "remaining": 10},
                artifacts=(),
                events=(LogEvent(kind="observation", turn=2, data={}),),
            ),
        ),
        seed=42,
        condition="test",
    )


class TestFormatters:
    def test_human_readable_structure(self) -> None:
        from CrisisWorld.tracing.formatters import render_human_readable

        trace = _trace_with_turns()
        output = render_human_readable(trace)
        assert "ep-fmt-001" in output
        assert "Turn 0" in output
        assert "Turn 1" in output
        assert "Turn 2" in output
        assert "Final Summary" in output

    def test_render_turn_contains_key_fields(self) -> None:
        from CrisisWorld.tracing.formatters import render_turn

        record = _trace_with_turns().turns[0]
        output = render_turn(record)
        assert "Turn 0" in output
        assert "deploy_resource" in output
        assert "-0.32" in output
        assert "18" in output  # remaining budget

    def test_empty_trace_renders(self) -> None:
        from CrisisWorld.tracing.formatters import render_human_readable

        trace = EpisodeTrace(episode_id="empty", turns=(), seed=0)
        output = render_human_readable(trace)
        assert "Turns: 0" in output
        assert "Turn 0" not in output

    def test_turn_without_artifacts_omits_roles(self) -> None:
        from CrisisWorld.tracing.formatters import render_turn

        record = TurnRecord(
            turn=5,
            observation={"x": 1},
            action={"kind": "noop"},
            reward={"total": 0.0},
            budget_snapshot={"total": 10, "spent": 0, "remaining": 10},
            artifacts=(),
            events=(),
        )
        output = render_turn(record)
        assert "Roles:" not in output

    def test_truncates_long_values(self) -> None:
        from CrisisWorld.tracing.formatters import MAX_LINE_WIDTH, render_turn

        record = TurnRecord(
            turn=0,
            observation={"long_field": "x" * 500},
            action={"kind": "noop"},
            reward={"total": 0.0},
            budget_snapshot=None,
            artifacts=(),
            events=(),
        )
        output = render_turn(record)
        for line in output.splitlines():
            assert len(line) <= MAX_LINE_WIDTH + 10
        assert "..." in output

    def test_special_characters_escaped(self) -> None:
        from CrisisWorld.tracing.formatters import render_turn

        record = TurnRecord(
            turn=0,
            observation={"msg": "line1\nline2\ttab"},
            action={"kind": "noop"},
            reward={"total": 0.0},
            budget_snapshot=None,
            artifacts=(),
            events=(),
        )
        output = render_turn(record)
        # Raw newlines in field values should be escaped
        lines = output.splitlines()
        for line in lines:
            if "line1" in line:
                assert "\\n" in line or "line2" not in line

    def test_none_optional_fields(self) -> None:
        from CrisisWorld.tracing.formatters import render_turn

        record = TurnRecord(
            turn=0,
            observation=None,
            action=None,
            reward=None,
            budget_snapshot=None,
            artifacts=(),
            events=(),
        )
        output = render_turn(record)
        assert "N/A" in output
