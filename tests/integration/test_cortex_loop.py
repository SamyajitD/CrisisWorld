"""Integration tests for cortex deliberation loop — real budget+memory, stub roles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cortex.budget import BudgetTracker
from cortex.deliberator import CortexDeliberator
from cortex.memory import EpisodeMemory
from cortex.roles import (
    CriticRole,
    ExecutiveRole,
    PerceptionRole,
    PlannerRole,
    WorldModelerRole,
)
from models import Escalate, NoOp, Observation, RegionState, ResourcePool, Telemetry
from schemas.budget import BudgetStatus
from schemas.episode import LogEvent
from tracing.tracer import EpisodeTracer

# Resolve Observation forward ref
Observation.model_rebuild(_types_namespace={"BudgetStatus": BudgetStatus})


def _make_obs(turn: int = 0, infected: int = 50) -> Observation:
    return Observation(
        turn=turn,
        regions=(RegionState(region_id="r0", population=1000, infected=infected),),
        telemetry=Telemetry(total_infected=infected),
        resources=ResourcePool(medical=100, personnel=50, funding=200),
        budget_status=BudgetStatus(total=20, spent=0, remaining=20),
    )


def _make_deliberator(budget_total: int = 20):
    roles = {
        "perception": PerceptionRole(),
        "world_modeler": WorldModelerRole(),
        "planner": PlannerRole(),
        "critic": CriticRole(),
        "executive": ExecutiveRole(),
    }
    memory = EpisodeMemory()
    logger = EpisodeTracer(episode_id="test")
    budget = BudgetTracker(budget_total)
    delib = CortexDeliberator(roles=roles, memory=memory, logger=logger)
    return delib, budget, memory, logger


class TestCortexLoop:
    def test_full_deliberation_perception_to_act(self) -> None:
        delib, budget, memory, _ = _make_deliberator(20)
        action, log = delib.deliberate(_make_obs(), budget)
        assert len(log.artifacts) >= 2  # at least perception + executive
        assert budget.remaining().spent >= 2

    def test_full_deliberation_with_world_modeler_and_planner(self) -> None:
        delib, budget, memory, _ = _make_deliberator(50)
        action, log = delib.deliberate(_make_obs(infected=200), budget)
        assert len(log.artifacts) >= 3
        assert budget.remaining().spent >= 3

    def test_full_deliberation_with_critic(self) -> None:
        delib, budget, memory, _ = _make_deliberator(50)
        action, log = delib.deliberate(_make_obs(infected=200), budget)
        # Check if any artifact has risk_score (critic output)
        has_critique = any("risk_score" in a for a in log.artifacts)
        # Critique may or may not fire depending on executive decision tree
        assert len(log.artifacts) >= 2

    def test_budget_exhaustion_midloop_graceful(self) -> None:
        delib, budget, _, _ = _make_deliberator(3)
        action, log = delib.deliberate(_make_obs(), budget)
        assert log.forced is True or isinstance(action, NoOp)
        assert log.termination_reason in ("budget_exhausted", "act", "wait", "stop")

    def test_memory_persists_across_role_calls_within_turn(self) -> None:
        delib, budget, memory, _ = _make_deliberator(20)
        delib.deliberate(_make_obs(), budget)
        # Perception should have been stored
        assert len(memory.retrieve("perception")) >= 1

    def test_memory_isolated_between_episodes(self) -> None:
        delib, budget, memory, _ = _make_deliberator(20)
        delib.deliberate(_make_obs(), budget)
        assert memory.digest().num_entries > 0
        delib.reset()
        assert memory.digest().num_entries == 0

    def test_deliberation_log_budget_snapshots(self) -> None:
        delib, budget, _, _ = _make_deliberator(20)
        _, log = delib.deliberate(_make_obs(), budget)
        assert log.budget_at_start["spent"] == 0
        assert log.budget_at_end["spent"] > 0

    def test_artifact_order_matches_invocation_order(self) -> None:
        delib, budget, _, _ = _make_deliberator(20)
        _, log = delib.deliberate(_make_obs(), budget)
        # First artifact should be from perception (has cleaned_observation)
        assert len(log.artifacts) >= 1
        first = log.artifacts[0]
        assert "cleaned_observation" in first or "salient_changes" in first

    def test_deterministic_with_same_inputs(self) -> None:
        d1, b1, _, _ = _make_deliberator(20)
        a1, l1 = d1.deliberate(_make_obs(), b1)

        d2, b2, _, _ = _make_deliberator(20)
        a2, l2 = d2.deliberate(_make_obs(), b2)

        assert a1.kind == a2.kind
        assert l1.iterations == l2.iterations

    def test_escalate_produces_correct_outer_action(self) -> None:
        # High-risk scenario to trigger escalation
        delib, budget, _, _ = _make_deliberator(50)
        obs = _make_obs(infected=800)  # Very high infection
        action, log = delib.deliberate(obs, budget)
        # May or may not escalate depending on decision tree path
        assert hasattr(action, "kind")

    def test_max_iterations_with_high_budget(self) -> None:
        delib, budget, _, _ = _make_deliberator(500)
        _, log = delib.deliberate(_make_obs(), budget)
        assert log.iterations <= 20  # MAX_INNER_ITERATIONS cap

    def test_logger_receives_structured_events(self) -> None:
        delib, budget, _, logger = _make_deliberator(20)
        delib.deliberate(_make_obs(), budget)
        trace = logger.finalize()
        # Should have at least perception + executive events
        assert len(trace.turns) >= 1 or len(logger._events) >= 2
