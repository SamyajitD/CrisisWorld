"""Unit tests for isolated router SFT utilities."""

from __future__ import annotations

import json

from training.router_sft.export_jsonl import build_sft_record
from training.router_sft.labels import canonical_route_label, normalize_sft_target


def test_canonical_route_label_maps_call_targets() -> None:
    assert (
        canonical_route_label(
            {"decision": "call", "target_role": "planner"}
        )
        == "call_planner"
    )
    assert (
        canonical_route_label(
            {"decision": "call", "target_role": "critic"}
        )
        == "call_critic"
    )
    assert canonical_route_label(
        {"decision": "call", "target_role": "world_modeler"}
    ) == "call_world_modeler"


def test_normalize_sft_target_fills_noop_for_missing_act_target() -> None:
    normalized = normalize_sft_target(
        {
            "decision": "act",
            "target_action": None,
            "target_role": None,
            "reasoning": "fallback",
        }
    )
    assert normalized["target_action"] == {"kind": "noop"}


def test_build_sft_record_contains_assistant_json() -> None:
    raw = {
        "episode_id": "ep-1",
        "seed": 1,
        "outer_turn": 0,
        "inner_iteration": 0,
        "input_payload": {
            "budget_status": {"total": 30, "spent": 4, "remaining": 26},
            "artifacts": [{"cleaned_observation": {"regions": []}}],
        },
        "teacher_decision": {
            "decision": "call",
            "target_action": None,
            "target_role": "planner",
            "reasoning": "Need plan",
        },
        "route_label": "call_planner",
        "return_to_go": 1.5,
        "episode_total_reward": 2.0,
        "termination_reason": "contained",
    }

    record = build_sft_record(raw)
    assistant = record["messages"][-1]["content"]
    parsed = json.loads(assistant)

    assert parsed["decision"] == "call"
    assert parsed["target_role"] == "planner"
    assert record["metadata"]["route_label"] == "call_planner"
