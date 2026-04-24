"""Human-readable trace rendering for episode traces."""

from __future__ import annotations

from typing import Any

from src.schemas.episode import EpisodeTrace, TurnRecord

MAX_LINE_WIDTH = 100

_SEPARATOR = "=" * 40


def render_human_readable(trace: EpisodeTrace) -> str:
    """Render full episode trace as human-readable text."""
    parts: list[str] = []
    parts.append(_render_header(trace))

    for record in trace.turns:
        parts.append(render_turn(record))

    parts.append(_render_footer(trace))
    return "\n".join(parts)


def render_turn(record: TurnRecord) -> str:
    """Render a single turn as human-readable text."""
    lines: list[str] = []
    lines.append(f"--- Turn {record.turn} ---")

    # Observation
    if record.observation is not None:
        obs_text = _summarize_observation(record.observation)
        lines.append(f"Obs: {_truncate(obs_text, MAX_LINE_WIDTH - 5)}")
    else:
        lines.append("Obs: N/A")

    # Artifacts / Roles
    if record.artifacts:
        lines.append("Roles:")
        for art in record.artifacts:
            art_text = _summarize_artifact(art)
            lines.append(f"  {_truncate(art_text, MAX_LINE_WIDTH - 4)}")

    # Action
    if record.action is not None:
        act_text = _format_action(record.action)
        lines.append(f"Action: {_truncate(act_text, MAX_LINE_WIDTH - 8)}")
    else:
        lines.append("Action: N/A")

    # Reward
    if record.reward is not None:
        rew_text = _format_reward(record.reward)
        lines.append(f"Reward: {_truncate(rew_text, MAX_LINE_WIDTH - 8)}")
    else:
        lines.append("Reward: N/A")

    # Budget
    if record.budget_snapshot is not None:
        bud = record.budget_snapshot
        remaining = bud.get("remaining", "?")
        total = bud.get("total", "?")
        lines.append(f"Budget: {remaining}/{total} remaining")
    else:
        lines.append("Budget: N/A")

    return "\n".join(lines)


def _render_header(trace: EpisodeTrace) -> str:
    """Render episode header block."""
    total_reward = _compute_total_reward(trace)
    return (
        f"{_SEPARATOR}\n"
        f"Episode: {trace.episode_id}\n"
        f"Turns: {len(trace.turns)} | Total Reward: {total_reward}\n"
        f"{_SEPARATOR}"
    )


def _render_footer(trace: EpisodeTrace) -> str:
    """Render episode footer / final summary."""
    total_reward = _compute_total_reward(trace)
    return (
        f"{_SEPARATOR}\n"
        f"Final Summary\n"
        f"  Total Reward: {total_reward}\n"
        f"  Turns Played: {len(trace.turns)}\n"
        f"{_SEPARATOR}"
    )


def _compute_total_reward(trace: EpisodeTrace) -> str:
    """Sum reward totals across turns. Returns formatted string."""
    total = 0.0
    has_any = False
    for tr in trace.turns:
        if tr.reward is not None and "total" in tr.reward:
            total += tr.reward["total"]
            has_any = True
    return f"{total:.2f}" if has_any else "N/A"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, adding '...' if trimmed."""
    text = _escape_special(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _escape_special(text: str) -> str:
    """Replace raw newlines and tabs with escaped representations."""
    return text.replace("\n", "\\n").replace("\t", "\\t")


def _summarize_observation(obs: dict[str, Any]) -> str:
    """Summarize observation dict into a compact string."""
    parts: list[str] = []
    items = list(obs.items())
    shown = items[:3]
    for key, val in shown:
        parts.append(f"{key}={val}")
    text = " | ".join(parts)
    if len(items) > 3:
        text += f" (+{len(items) - 3} more)"
    return f"[{text}]"


def _summarize_artifact(art: dict[str, Any]) -> str:
    """Summarize a single artifact dict."""
    role = art.get("role", "unknown")
    details = {k: v for k, v in art.items() if k != "role"}
    if details:
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        return f"{role}: {detail_str}"
    return str(role)


def _format_action(action: dict[str, Any]) -> str:
    """Format an action dict into a compact string."""
    kind = action.get("kind", "unknown")
    params = {k: v for k, v in action.items() if k != "kind"}
    if params:
        param_str = ", ".join(f"{v}" for v in params.values())
        return f"{kind}({param_str})"
    return kind


def _format_reward(reward: dict[str, Any]) -> str:
    """Format reward dict into a compact string."""
    total = reward.get("total", "?")
    components = {k: v for k, v in reward.items() if k != "total"}
    if components:
        comp_str = ", ".join(f"{k}={v}" for k, v in components.items())
        return f"{total} ({comp_str})"
    return str(total)
