"""Unit tests for training/ — manifest schemas and data export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.manifest import RunManifest, SFTConfig, EvalConfig
from training.data_export import export_router_sft, export_policy_sft


class TestManifest:
    def test_default_manifest(self) -> None:
        m = RunManifest(run_name="test", run_type="router_sft")
        assert m.run_type == "router_sft"
        assert m.quantization == "4bit"

    def test_sft_config_defaults(self) -> None:
        c = SFTConfig()
        assert c.learning_rate == 2e-4
        assert c.lora_r == 16

    def test_eval_config_defaults(self) -> None:
        c = EvalConfig()
        assert len(c.eval_seeds) == 5
        assert c.max_turns == 20

    def test_manifest_roundtrip(self) -> None:
        m = RunManifest(
            run_name="exp-1",
            run_type="single_policy_sft",
            base_models={"default": "mistral-7b"},
            seeds=(42, 43),
        )
        data = m.model_dump()
        m2 = RunManifest.model_validate(data)
        assert m2.run_name == "exp-1"


class TestDataExport:
    def _make_trace_file(self, tmp_path: Path, name: str = "ep-test.json") -> Path:
        """Create a minimal trace file for testing."""
        trace = {
            "episode_id": "ep-test",
            "seed": 42,
            "turns": [
                {
                    "turn": 0,
                    "observation": {"turn": 0, "regions": [{"region_id": "r0", "population": 1000, "infected": 50}]},
                    "action": {"kind": "deploy_resource", "resource": "medical", "region_id": "r0", "amount": 10},
                    "reward": {"total": 0.5},
                    "budget_snapshot": {"total": 20, "spent": 2, "remaining": 18},
                    "events": [
                        {"kind": "artifact", "turn": 0, "data": {"role": "perception", "cleaned_observation": {}}},
                        {"kind": "artifact", "turn": 0, "data": {"role": "executive", "decision": "act", "target_action": {"kind": "deploy_resource"}, "reasoning": "high infection"}},
                        {"kind": "observation", "turn": 0, "data": {"turn": 0}},
                        {"kind": "action", "turn": 0, "data": {"kind": "deploy_resource"}},
                    ],
                },
                {
                    "turn": 1,
                    "observation": {"turn": 1, "regions": [{"region_id": "r0", "population": 1000, "infected": 40}]},
                    "action": {"kind": "noop"},
                    "events": [
                        {"kind": "artifact", "turn": 1, "data": {"role": "executive", "decision": "call", "target_role": "planner", "reasoning": "need plan"}},
                        {"kind": "artifact", "turn": 1, "data": {"role": "executive", "decision": "act", "target_action": {"kind": "noop"}, "reasoning": "done"}},
                    ],
                },
            ],
        }
        path = tmp_path / name
        path.write_text(json.dumps(trace), encoding="utf-8")
        return path

    def test_router_export_produces_jsonl(self, tmp_path: Path) -> None:
        self._make_trace_file(tmp_path)
        output = tmp_path / "router.jsonl"
        count = export_router_sft(tmp_path, output)
        assert count > 0
        assert output.exists()

        # Verify JSONL format
        lines = output.read_text().strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert "messages" in row
            assert len(row["messages"]) == 3
            assert row["messages"][0]["role"] == "system"
            assert row["messages"][2]["role"] == "assistant"

    def test_router_export_extracts_decisions(self, tmp_path: Path) -> None:
        self._make_trace_file(tmp_path)
        output = tmp_path / "router.jsonl"
        count = export_router_sft(tmp_path, output)
        # Should have 3 executive decisions (1 in turn 0, 2 in turn 1)
        assert count == 3

    def test_policy_export_produces_jsonl(self, tmp_path: Path) -> None:
        self._make_trace_file(tmp_path)
        output = tmp_path / "policy.jsonl"
        count = export_policy_sft(tmp_path, output)
        assert count > 0

        lines = output.read_text().strip().split("\n")
        for line in lines:
            row = json.loads(line)
            assert "messages" in row
            # Assistant message should be valid action JSON
            action = json.loads(row["messages"][2]["content"])
            assert "kind" in action

    def test_policy_export_count_matches_turns(self, tmp_path: Path) -> None:
        self._make_trace_file(tmp_path)
        output = tmp_path / "policy.jsonl"
        count = export_policy_sft(tmp_path, output)
        assert count == 2  # 2 turns with actions

    def test_export_empty_dir(self, tmp_path: Path) -> None:
        output = tmp_path / "empty.jsonl"
        count = export_router_sft(tmp_path, output)
        assert count == 0

    def test_export_invalid_json_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("not json")
        output = tmp_path / "out.jsonl"
        count = export_router_sft(tmp_path, output)
        assert count == 0
