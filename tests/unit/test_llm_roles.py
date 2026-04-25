"""Unit tests for cortex/llm/ — provider, roles, prompts. No real API calls."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cortex.llm.provider import HuggingFaceProvider
from cortex.llm.prompts import ROLE_PROMPTS, _summarize_obs
from schemas.artifact import CleanState, BeliefState, Plan, Critique, ExecutiveDecision


# ---------------------------------------------------------------------------
# JSON extraction tests (provider._extract_json)
# ---------------------------------------------------------------------------


class TestJsonExtraction:
    def _extractor(self) -> HuggingFaceProvider:
        """Create a provider with a fake key (we only test _extract_json)."""
        with patch("cortex.llm.provider.InferenceClient"):
            return HuggingFaceProvider(api_key="fake", model="test")

    def test_simple_json(self) -> None:
        p = self._extractor()
        result = p._extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_in_code_fence(self) -> None:
        p = self._extractor()
        text = 'Here is the result:\n```json\n{"risk_score": 0.5}\n```\nDone.'
        result = p._extract_json(text)
        assert result["risk_score"] == 0.5

    def test_nested_json(self) -> None:
        p = self._extractor()
        text = '{"forecast_trajectories": [{"label": "opt", "values": [1, 2, 3]}], "confidence": 0.8}'
        result = p._extract_json(text)
        assert result["confidence"] == 0.8
        assert len(result["forecast_trajectories"]) == 1

    def test_deeply_nested(self) -> None:
        p = self._extractor()
        text = '{"a": {"b": {"c": {"d": 1}}}}'
        result = p._extract_json(text)
        assert result["a"]["b"]["c"]["d"] == 1

    def test_json_with_surrounding_text(self) -> None:
        p = self._extractor()
        text = 'The analysis shows: {"risk_score": 0.3, "failure_modes": []} and that is my answer.'
        result = p._extract_json(text)
        assert result["risk_score"] == 0.3

    def test_no_json_raises(self) -> None:
        p = self._extractor()
        with pytest.raises(ValueError, match="No JSON found"):
            p._extract_json("Just some text with no JSON at all")

    def test_unbalanced_braces_raises(self) -> None:
        p = self._extractor()
        with pytest.raises(ValueError):
            p._extract_json('{"key": "value"')

    def test_json_with_escaped_quotes(self) -> None:
        p = self._extractor()
        text = '{"message": "He said \\"hello\\"", "count": 1}'
        result = p._extract_json(text)
        assert result["count"] == 1

    def test_strings_with_braces(self) -> None:
        p = self._extractor()
        text = '{"rationale": "Deploy {medical} resources to r0", "confidence": 0.7}'
        result = p._extract_json(text)
        assert result["confidence"] == 0.7


# ---------------------------------------------------------------------------
# LLMRole tests (mocked provider)
# ---------------------------------------------------------------------------


class TestLLMRole:
    def _make_role(self, name: str, return_json: dict | Exception):
        from cortex.llm.roles import LLMRole

        provider = MagicMock(spec=HuggingFaceProvider)
        provider.total_fallbacks = 0
        if isinstance(return_json, Exception):
            provider.complete.side_effect = return_json
        else:
            provider.complete.return_value = return_json

        # Create a heuristic fallback
        fallback = MagicMock()
        fallback.invoke.return_value = CleanState()  # default fallback artifact

        role = LLMRole(name=name, provider=provider, fallback=fallback)
        return role, provider, fallback

    def test_perception_returns_clean_state(self) -> None:
        role, _, _ = self._make_role("perception", {
            "salient_changes": ["r0 infected increased"],
            "flagged_anomalies": [],
            "cleaned_observation": {"turn": 1},
        })
        from schemas.artifact import RoleInput
        result = role.invoke(RoleInput(role_name="perception", payload={"observation": {}}))
        assert isinstance(result, CleanState)

    def test_planner_returns_plan(self) -> None:
        role, _, _ = self._make_role("planner", {
            "candidates": [
                {"action": {"kind": "noop"}, "rationale": "wait", "expected_effect": "none", "confidence": 0.5},
            ],
        })
        from schemas.artifact import RoleInput
        result = role.invoke(RoleInput(role_name="planner", payload={"belief_state": {}}))
        assert isinstance(result, Plan)
        assert len(result.candidates) == 1

    def test_executive_returns_decision(self) -> None:
        role, _, _ = self._make_role("executive", {
            "decision": "act",
            "target_action": {"kind": "noop"},
            "reasoning": "budget low",
        })
        from schemas.artifact import RoleInput
        result = role.invoke(RoleInput(role_name="executive", payload={"artifacts": [], "budget_status": {}}))
        assert isinstance(result, ExecutiveDecision)
        assert result.decision == "act"

    def test_fallback_on_api_error(self) -> None:
        role, provider, fallback = self._make_role("perception", RuntimeError("API down"))
        from schemas.artifact import RoleInput
        result = role.invoke(RoleInput(role_name="perception", payload={"observation": {}}))
        fallback.invoke.assert_called_once()
        assert provider.total_fallbacks == 1

    def test_fallback_on_invalid_json(self) -> None:
        # Critic requires risk_score (0-1). Passing 5.0 will fail validation.
        role, provider, fallback = self._make_role("critic", {"risk_score": 5.0})
        from schemas.artifact import RoleInput
        result = role.invoke(RoleInput(role_name="critic", payload={"candidate": {}, "belief_state": {}}))
        fallback.invoke.assert_called_once()  # Pydantic validation fails -> fallback
        assert provider.total_fallbacks == 1

    def test_cost_matches_expected(self) -> None:
        from cortex.llm.roles import LLMRole, _ROLE_COSTS
        for name, expected_cost in _ROLE_COSTS.items():
            role, _, _ = self._make_role(name, {})
            assert role.cost == expected_cost

    def test_role_name_property(self) -> None:
        role, _, _ = self._make_role("critic", {})
        assert role.role_name == "critic"

    def test_invalid_role_name_raises(self) -> None:
        from cortex.llm.roles import LLMRole
        provider = MagicMock(spec=HuggingFaceProvider)
        fallback = MagicMock()
        with pytest.raises(ValueError, match="Unknown role"):
            LLMRole(name="nonexistent", provider=provider, fallback=fallback)

    def test_all_roles_in_protocol(self) -> None:
        from protocols.role import RoleProtocol
        for name in ROLE_PROMPTS:
            role, _, _ = self._make_role(name, {})
            assert isinstance(role, RoleProtocol)


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_all_roles_have_prompts(self) -> None:
        assert set(ROLE_PROMPTS.keys()) == {"perception", "world_modeler", "planner", "critic", "executive"}

    def test_perception_prompt_contains_observation(self) -> None:
        _, build_user = ROLE_PROMPTS["perception"]
        text = build_user({"observation": {"turn": 3, "regions": [{"region_id": "r0", "population": 1000, "infected": 50}]}})
        assert "r0" in text
        assert "1000" in text

    def test_planner_prompt_mentions_action_kinds(self) -> None:
        system, _ = ROLE_PROMPTS["planner"]
        assert "deploy_resource" in system
        assert "restrict_movement" in system
        assert "noop" in system

    def test_executive_prompt_mentions_budget(self) -> None:
        _, build_user = ROLE_PROMPTS["executive"]
        text = build_user({"artifacts": [], "budget_status": {"remaining": 5, "spent": 15, "total": 20}})
        assert "remaining=5" in text

    def test_summarize_obs_empty_regions(self) -> None:
        text = _summarize_obs({"turn": 0, "regions": []})
        assert "Turn: 0" in text

    def test_summarize_obs_handles_non_dict_regions(self) -> None:
        text = _summarize_obs({"turn": 0, "regions": "not a list"})
        assert "Turn: 0" in text


# ---------------------------------------------------------------------------
# Provider construction tests
# ---------------------------------------------------------------------------


class TestProviderConstruction:
    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="API key is required"):
            HuggingFaceProvider(api_key="")

    def test_reset_stats(self) -> None:
        with patch("cortex.llm.provider.InferenceClient"):
            p = HuggingFaceProvider(api_key="fake", model="test")
            p.total_calls = 10
            p.total_fallbacks = 3
            p.total_response_time = 25.0
            p.reset_stats()
            assert p.total_calls == 0
            assert p.total_fallbacks == 0
            assert p.total_response_time == 0.0
