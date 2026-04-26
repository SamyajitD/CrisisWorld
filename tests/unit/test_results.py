"""Unit tests for training/results.py and training/hub.py schemas."""

from __future__ import annotations

import pytest

from training.results import RunResult, TrainingMetrics, EvalMetrics
from training.manifest import RunManifest


class TestResultSchemas:
    def test_training_metrics_defaults(self) -> None:
        m = TrainingMetrics()
        assert m.train_loss == 0.0
        assert m.num_examples == 0

    def test_eval_metrics_defaults(self) -> None:
        m = EvalMetrics()
        assert m.condition == ""
        assert m.fallback_count == 0

    def test_run_result_construction(self) -> None:
        r = RunResult(
            run_name="test-run",
            run_type="router_sft",
            training=TrainingMetrics(train_loss=0.5, num_examples=100),
            evaluations=(
                EvalMetrics(condition="flat-fat", mean_reward=10.0),
                EvalMetrics(condition="cortex-llm", mean_reward=12.0),
            ),
        )
        assert r.run_name == "test-run"
        assert len(r.evaluations) == 2
        assert r.evaluations[1].mean_reward == 12.0

    def test_run_result_roundtrip(self) -> None:
        r = RunResult(run_name="rt", run_type="single_policy_sft")
        data = r.model_dump()
        r2 = RunResult.model_validate(data)
        assert r2 == r

    def test_run_result_with_comparison_table(self) -> None:
        r = RunResult(
            run_name="cmp",
            run_type="cortex_eval",
            comparison_table="| Condition | Reward |\n|---|---|\n| flat | 10 |",
        )
        assert "flat" in r.comparison_table


class TestPerRoleProvider:
    def test_per_role_config_resolution(self) -> None:
        """Test the config resolution logic for per-role model assignment."""
        config = {
            "models": {
                "llama-3.1-8b": {"name": "meta-llama/Llama-3.1-8B-Instruct"},
                "qwen-7b": {"name": "Qwen/Qwen2.5-7B-Instruct"},
            },
            "default_model": "llama-3.1-8b",
            "role_models": {"planner": "qwen-7b"},
        }
        # Test the resolution logic (same as create_provider_for_role internals)
        models = config["models"]
        role_models = config.get("role_models", {})
        default_key = config["default_model"]

        # Perception should get default
        key = role_models.get("perception", default_key)
        assert models[key]["name"] == "meta-llama/Llama-3.1-8B-Instruct"

        # Planner should get override
        key = role_models.get("planner", default_key)
        assert models[key]["name"] == "Qwen/Qwen2.5-7B-Instruct"
