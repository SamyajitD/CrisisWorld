"""Export SFT training datasets from CrisisWorld episode traces.

Two dataset types:
1. Router SFT: Executive decision traces -> chat JSONL
2. Single-policy SFT: Observation -> outer action pairs -> chat JSONL
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SCENARIO_CONTEXT = (
    "You are managing a disease outbreak across multiple regions. "
    "Each region has population, infected, recovered, and deceased counts. "
    "You have medical, personnel, and funding resources."
)


def _load_trace(path: Path) -> dict[str, Any]:
    """Load a trace JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def _chat_row(system: str, user: str, assistant: str) -> dict[str, Any]:
    """Build one HF chat-format training row."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


# ---------------------------------------------------------------------------
# Router SFT (Executive decisions)
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = (
    f"{SCENARIO_CONTEXT}\n\n"
    "You are the EXECUTIVE. Given the artifacts produced so far and the remaining budget, "
    "decide: act (commit to an action), call (invoke another role: world_modeler, planner, critic), "
    "wait, escalate, or stop.\n\n"
    "Respond with ONLY valid JSON: "
    '{"decision": "...", "target_action": {...} or null, "target_role": "..." or null, "reasoning": "..."}'
)


def export_router_sft(traces_dir: Path, output: Path) -> int:
    """Convert deliberation traces to Executive router SFT JSONL.

    Returns number of training examples exported.
    """
    traces_dir = Path(traces_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as f:
        for trace_file in sorted(traces_dir.glob("*.json")):
            try:
                trace = _load_trace(trace_file)
            except Exception as exc:
                _log.warning("Skipping %s: %s", trace_file.name, exc)
                continue

            for turn in trace.get("turns", []):
                events = turn.get("events", [])
                artifacts_so_far: list[dict] = []
                budget_snapshot = turn.get("budget_snapshot", {})

                for event in events:
                    kind = event.get("kind", "")
                    data = event.get("data", {})

                    if kind == "artifact" and data.get("role") == "executive":
                        # This executive decision is a training example
                        user_text = (
                            f"Budget: {json.dumps(budget_snapshot)}\n\n"
                            f"Artifacts ({len(artifacts_so_far)}):\n"
                            + "\n".join(json.dumps(a)[:300] for a in artifacts_so_far[-5:])
                        )
                        # Extract the executive decision fields
                        decision_data = {
                            k: data.get(k)
                            for k in ("decision", "target_action", "target_role", "reasoning")
                            if k in data
                        }
                        if decision_data.get("decision"):
                            row = _chat_row(ROUTER_SYSTEM, user_text, json.dumps(decision_data))
                            f.write(json.dumps(row) + "\n")
                            count += 1

                    if kind == "artifact":
                        artifacts_so_far.append(data)

    _log.info("Exported %d router SFT examples to %s", count, output)
    return count


# ---------------------------------------------------------------------------
# Single-policy SFT (Observation -> outer action)
# ---------------------------------------------------------------------------

POLICY_SYSTEM = (
    f"{SCENARIO_CONTEXT}\n\n"
    "Given the current observation, choose exactly one action.\n\n"
    "Valid action kinds: deploy_resource, restrict_movement, request_data, "
    "public_communication, escalate, reallocate_budget, noop.\n\n"
    'Respond with ONLY valid JSON: {"kind": "...", ...action_params}'
)


def _summarize_obs_from_turn(turn: dict[str, Any]) -> str:
    """Build observation summary from a TurnRecord dict."""
    obs = turn.get("observation", {})
    if not obs:
        return "No observation data available."

    parts = [f"Turn: {turn.get('turn', '?')}"]
    if isinstance(obs, dict):
        for key in ("regions", "resources", "telemetry"):
            val = obs.get(key)
            if val:
                parts.append(f"{key}: {json.dumps(val, default=str)[:300]}")
    return "\n".join(parts)


def export_policy_sft(traces_dir: Path, output: Path) -> int:
    """Convert episode traces to single-policy SFT JSONL.

    Each (observation, action) pair becomes one training example.
    Returns number of training examples exported.
    """
    traces_dir = Path(traces_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as f:
        for trace_file in sorted(traces_dir.glob("*.json")):
            try:
                trace = _load_trace(trace_file)
            except Exception as exc:
                _log.warning("Skipping %s: %s", trace_file.name, exc)
                continue

            for turn in trace.get("turns", []):
                action = turn.get("action")
                if not action or not isinstance(action, dict):
                    continue

                obs_text = _summarize_obs_from_turn(turn)
                action_json = json.dumps(action)

                row = _chat_row(POLICY_SYSTEM, obs_text, action_json)
                f.write(json.dumps(row) + "\n")
                count += 1

    _log.info("Exported %d policy SFT examples to %s", count, output)
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for dataset export."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Export SFT datasets from traces")
    parser.add_argument("--task", choices=["router", "policy"], required=True)
    parser.add_argument("--traces-dir", type=Path, default=Path("traces"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.task == "router":
        n = export_router_sft(args.traces_dir, args.output)
    else:
        n = export_policy_sft(args.traces_dir, args.output)

    print(f"Exported {n} examples to {args.output}")


if __name__ == "__main__":
    main()
