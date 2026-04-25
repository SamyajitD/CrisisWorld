"""Unit tests for schemas/ — agent-side data models per CLAUDE.md spec."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# artifact.py tests
# ---------------------------------------------------------------------------


class TestArtifacts:
    def test_role_input_construction(self) -> None:
        from schemas.artifact import RoleInput

        ri = RoleInput(role_name="perception", payload={"key": "val"})
        assert ri.role_name == "perception"
        assert ri.payload == {"key": "val"}

    def test_clean_state_defaults(self) -> None:
        from schemas.artifact import CleanState

        cs = CleanState()
        assert cs.salient_changes == ()
        assert cs.flagged_anomalies == ()
        assert cs.cleaned_observation == {}

    def test_belief_state_confidence_bounds(self) -> None:
        from schemas.artifact import BeliefState

        with pytest.raises(Exception):
            BeliefState(confidence=-0.1)
        with pytest.raises(Exception):
            BeliefState(confidence=1.1)

    def test_candidate_action_confidence_bounds(self) -> None:
        from schemas.artifact import CandidateAction

        with pytest.raises(Exception):
            CandidateAction(
                action={}, rationale="r", expected_effect="e", confidence=1.5
            )

    def test_plan_empty_candidates(self) -> None:
        from schemas.artifact import Plan

        p = Plan()
        assert p.candidates == ()

    def test_critique_risk_score_bounds(self) -> None:
        from schemas.artifact import Critique

        with pytest.raises(Exception):
            Critique(risk_score=-0.1)
        with pytest.raises(Exception):
            Critique(risk_score=1.1)

    def test_executive_decision_valid_decisions(self) -> None:
        from schemas.artifact import ExecutiveDecision

        for d in ("act", "call", "wait", "escalate", "stop"):
            ed = ExecutiveDecision(decision=d, reasoning="test")
            assert ed.decision == d

    def test_executive_decision_rejects_invalid(self) -> None:
        from schemas.artifact import ExecutiveDecision

        with pytest.raises(Exception):
            ExecutiveDecision(decision="think", reasoning="test")

    def test_artifact_union_isinstance(self) -> None:
        from schemas.artifact import (
            Artifact,
            BeliefState,
            CleanState,
            Critique,
            ExecutiveDecision,
            Plan,
        )

        instances = [
            CleanState(),
            BeliefState(confidence=0.5),
            Plan(),
            Critique(risk_score=0.3),
            ExecutiveDecision(decision="act", reasoning="go"),
        ]
        for inst in instances:
            assert isinstance(inst, Artifact.__args__)  # type: ignore[attr-defined]

    def test_all_artifacts_frozen(self) -> None:
        from schemas.artifact import (
            BeliefState,
            CleanState,
            Critique,
            ExecutiveDecision,
            Plan,
        )

        for cls in (CleanState, BeliefState, Plan, Critique, ExecutiveDecision):
            assert cls.model_config.get("frozen") is True


# ---------------------------------------------------------------------------
# budget.py tests
# ---------------------------------------------------------------------------


class TestBudget:
    def test_budget_status_construction(self) -> None:
        from schemas.budget import BudgetStatus

        bs = BudgetStatus(total=20, spent=5, remaining=15)
        assert bs.total == 20
        assert bs.spent == 5
        assert bs.remaining == 15

    def test_budget_status_invariant_enforced(self) -> None:
        from schemas.budget import BudgetStatus

        with pytest.raises(ValueError, match="remaining"):
            BudgetStatus(total=20, spent=5, remaining=10)

    def test_budget_status_zero_budget(self) -> None:
        from schemas.budget import BudgetStatus

        bs = BudgetStatus(total=0, spent=0, remaining=0)
        assert bs.total == 0

    def test_budget_status_rejects_negative_total(self) -> None:
        from schemas.budget import BudgetStatus

        with pytest.raises(Exception):
            BudgetStatus(total=-1, spent=0, remaining=-1)

    def test_budget_status_frozen(self) -> None:
        from schemas.budget import BudgetStatus

        bs = BudgetStatus(total=10, spent=0, remaining=10)
        with pytest.raises(Exception):
            bs.total = 5  # type: ignore[misc]

    def test_ledger_entry_rejects_zero_cost(self) -> None:
        from schemas.budget import LedgerEntry

        with pytest.raises(Exception):
            LedgerEntry(role_name="test", cost=0, turn=0)

    def test_budget_ledger_empty_entries(self) -> None:
        from schemas.budget import BudgetLedger, BudgetStatus

        bl = BudgetLedger(
            entries=(), status=BudgetStatus(total=10, spent=0, remaining=10)
        )
        assert bl.entries == ()

    def test_budget_exhausted_error_construction(self) -> None:
        from schemas.budget import BudgetExhaustedError

        err = BudgetExhaustedError(requested=5, remaining=2)
        assert err.requested == 5
        assert err.remaining == 2

    def test_budget_exhausted_error_is_exception(self) -> None:
        from schemas.budget import BudgetExhaustedError

        assert issubclass(BudgetExhaustedError, Exception)

    def test_budget_status_serialization_roundtrip(self) -> None:
        from schemas.budget import BudgetStatus

        bs = BudgetStatus(total=20, spent=8, remaining=12)
        data = bs.model_dump()
        bs2 = BudgetStatus.model_validate(data)
        assert bs == bs2


# ---------------------------------------------------------------------------
# config.py tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_cortex_config_defaults(self) -> None:
        from schemas.config import CortexConfig

        cc = CortexConfig()
        assert cc.total_budget == 20
        assert cc.perception_cost == 1
        assert cc.max_inner_iterations == 10

    def test_cortex_config_rejects_zero_budget(self) -> None:
        from schemas.config import CortexConfig

        with pytest.raises(Exception):
            CortexConfig(total_budget=0)

    def test_experiment_config_rejects_empty_seeds(self) -> None:
        from schemas.config import ExperimentConfig

        with pytest.raises(ValueError, match="seeds"):
            ExperimentConfig(seeds=(), conditions=("a",))

    def test_experiment_config_rejects_empty_conditions(self) -> None:
        from schemas.config import ExperimentConfig

        with pytest.raises(ValueError, match="conditions"):
            ExperimentConfig(seeds=(42,), conditions=())

    def test_config_serialization_roundtrips(self) -> None:
        from schemas.config import CortexConfig, ExperimentConfig

        cc = CortexConfig()
        assert CortexConfig.model_validate(cc.model_dump()) == cc

        ec = ExperimentConfig(seeds=(42, 43), conditions=("flat-lite",))
        assert ExperimentConfig.model_validate(ec.model_dump()) == ec


# ---------------------------------------------------------------------------
# episode.py tests
# ---------------------------------------------------------------------------


class TestEpisode:
    def test_log_event_construction(self) -> None:
        from schemas.episode import LogEvent

        le = LogEvent(kind="obs", turn=0, data={"x": 1})
        assert le.kind == "obs"
        assert le.turn == 0

    def test_log_event_rejects_negative_turn(self) -> None:
        from schemas.episode import LogEvent

        with pytest.raises(Exception):
            LogEvent(kind="obs", turn=-1)

    def test_turn_record_defaults(self) -> None:
        from schemas.episode import TurnRecord

        tr = TurnRecord(turn=0)
        assert tr.observation is None
        assert tr.action is None
        assert tr.reward is None
        assert tr.budget_snapshot is None
        assert tr.artifacts == ()
        assert tr.events == ()

    def test_episode_trace_empty_turns(self) -> None:
        from schemas.episode import EpisodeTrace

        et = EpisodeTrace(episode_id="ep-1", seed=42)
        assert et.turns == ()

    def test_episode_result_rejects_negative_turns(self) -> None:
        from schemas.episode import EpisodeResult

        with pytest.raises(Exception):
            EpisodeResult(
                episode_id="x",
                seed=0,
                condition="c",
                total_turns=-1,
                total_reward=0.0,
                termination_reason="r",
            )

    def test_memory_digest_defaults(self) -> None:
        from schemas.episode import MemoryDigest

        md = MemoryDigest()
        assert md.num_entries == 0
        assert md.keys == ()

    def test_memory_digest_rejects_negative_entries(self) -> None:
        from schemas.episode import MemoryDigest

        with pytest.raises(Exception):
            MemoryDigest(num_entries=-1)

    def test_episode_trace_serialization_roundtrip(self) -> None:
        from schemas.episode import EpisodeTrace, LogEvent, TurnRecord

        trace = EpisodeTrace(
            episode_id="ep-1",
            seed=42,
            condition="test",
            turns=(
                TurnRecord(
                    turn=0,
                    events=(LogEvent(kind="obs", turn=0),),
                ),
            ),
        )
        data = trace.model_dump()
        trace2 = EpisodeTrace.model_validate(data)
        assert trace == trace2


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestCrossCutting:
    def test_all_models_have_frozen_config(self) -> None:
        from schemas.artifact import (
            BeliefState,
            CandidateAction,
            CleanState,
            Critique,
            ExecutiveDecision,
            Plan,
            RoleInput,
        )
        from schemas.budget import BudgetLedger, BudgetStatus, LedgerEntry
        from schemas.config import CortexConfig, ExperimentConfig
        from schemas.episode import (
            EpisodeResult,
            EpisodeTrace,
            LogEvent,
            MemoryDigest,
            TurnRecord,
        )

        all_models = [
            RoleInput, CleanState, BeliefState, CandidateAction, Plan,
            Critique, ExecutiveDecision,
            BudgetStatus, LedgerEntry, BudgetLedger,
            CortexConfig, ExperimentConfig,
            LogEvent, TurnRecord, EpisodeTrace, EpisodeResult, MemoryDigest,
        ]
        for cls in all_models:
            assert cls.model_config.get("frozen") is True, (
                f"{cls.__name__} is not frozen"
            )
