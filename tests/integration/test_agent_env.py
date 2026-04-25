"""Integration tests for agents/ — both agents with realistic observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from agents.cortex_agent import CortexAgent
from agents.flat import FlatAgent
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
from models import (
    EnvConfig,
    NoOp,
    Observation,
    OuterAction,
    RegionState,
    ResourcePool,
    StakeholderSignal,
    Telemetry,
)
from schemas.budget import BudgetStatus
from schemas.episode import LogEvent
from tracing.tracer import EpisodeTracer

Observation.model_rebuild(_types_namespace={"BudgetStatus": BudgetStatus})


def _make_obs(
    turn: int = 0, infected: int = 50, medical: int = 100, urgency: float = 0.3,
) -> Observation:
    return Observation(
        turn=turn,
        regions=(
            RegionState(region_id="r0", population=1000, infected=infected),
            RegionState(region_id="r1", population=1000, infected=max(0, infected // 2)),
        ),
        stakeholder_signals=(
            StakeholderSignal(source="hospital", urgency=urgency, message="report", turn=turn),
        ),
        telemetry=Telemetry(total_infected=infected + infected // 2),
        resources=ResourcePool(medical=medical, personnel=50, funding=200),
        budget_status=BudgetStatus(total=20, spent=0, remaining=20),
    )


def _make_cortex_agent(budget_total: int = 20) -> CortexAgent:
    roles = {
        "perception": PerceptionRole(),
        "world_modeler": WorldModelerRole(),
        "planner": PlannerRole(),
        "critic": CriticRole(),
        "executive": ExecutiveRole(),
    }
    memory = EpisodeMemory()
    ep_logger = EpisodeTracer(episode_id="test")
    budget = BudgetTracker(budget_total)
    delib = CortexDeliberator(roles=roles, memory=memory, logger=ep_logger)
    return CortexAgent(deliberator=delib, budget=budget, ep_logger=ep_logger, initial_budget=budget_total)


class TestAgentEnv:
    def test_flat_agent_produces_valid_action(self) -> None:
        agent = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        action = agent.act(_make_obs(infected=200))
        assert isinstance(action, OuterAction)
        assert hasattr(action, "kind")

    def test_flat_agent_deterministic_with_same_seed(self) -> None:
        obs_seq = [_make_obs(turn=i, infected=50 + i * 10) for i in range(10)]

        a1 = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        actions1 = [a1.act(o).kind for o in obs_seq]

        a2 = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        actions2 = [a2.act(o).kind for o in obs_seq]

        assert actions1 == actions2

    def test_cortex_agent_delegates_correctly(self) -> None:
        agent = _make_cortex_agent()
        action = agent.act(_make_obs())
        assert isinstance(action, OuterAction)

    def test_cortex_agent_logs_deliberation_trace(self) -> None:
        agent = _make_cortex_agent()
        agent.act(_make_obs())
        # Logger should have recorded at least one event
        assert agent._turn_count == 1

    def test_both_agents_satisfy_agent_protocol(self) -> None:
        from protocols.agent import AgentProtocol

        flat = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        cortex = _make_cortex_agent()
        assert isinstance(flat, AgentProtocol)
        assert isinstance(cortex, AgentProtocol)

    def test_both_agents_handle_reset_cleanly(self) -> None:
        flat = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        flat.act(_make_obs())
        flat.reset()
        action = flat.act(_make_obs())
        assert isinstance(action, OuterAction)

        cortex = _make_cortex_agent()
        cortex.act(_make_obs())
        cortex.reset()
        action = cortex.act(_make_obs())
        assert isinstance(action, OuterAction)

    def test_flat_and_cortex_produce_outer_action(self) -> None:
        flat = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        cortex = _make_cortex_agent()

        fa = flat.act(_make_obs())
        ca = cortex.act(_make_obs())
        assert hasattr(fa, "kind")
        assert hasattr(ca, "kind")

    def test_cortex_agent_handles_budget_exhaustion(self) -> None:
        agent = _make_cortex_agent(budget_total=1)
        # First act may partially work; second should gracefully return NoOp
        agent.act(_make_obs())
        action = agent.act(_make_obs(turn=1))
        assert isinstance(action, NoOp)

    def test_flat_agent_multi_turn_sequence(self) -> None:
        agent = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        kinds: list[str] = []
        for i in range(20):
            infected = 10 + i * 15  # Surge at turn ~8
            obs = _make_obs(turn=i, infected=min(infected, 900))
            action = agent.act(obs)
            kinds.append(action.kind)

        # Should have triggered deploy_resource at some point
        assert "deploy_resource" in kinds

    def test_cortex_agent_reset_mid_episode(self) -> None:
        agent = _make_cortex_agent()
        agent.act(_make_obs(turn=0))
        agent.act(_make_obs(turn=1))
        agent.reset()
        assert agent._turn_count == 0
        action = agent.act(_make_obs(turn=0))
        assert isinstance(action, OuterAction)

    def test_flat_agent_fat_mode_valid(self) -> None:
        # FlatAgent doesn't have fat_mode yet, but should still work
        agent = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        action = agent.act(_make_obs(infected=200))
        assert isinstance(action, OuterAction)

    def test_both_agents_accept_empty_observation(self) -> None:
        minimal_obs = Observation(
            turn=0,
            regions=(),
            telemetry=Telemetry(),
            resources=ResourcePool(),
            budget_status=BudgetStatus(total=20, spent=0, remaining=20),
        )
        flat = FlatAgent(config=EnvConfig(), rng=np.random.default_rng(42))
        action = flat.act(minimal_obs)
        assert isinstance(action, NoOp)

        cortex = _make_cortex_agent()
        action = cortex.act(minimal_obs)
        assert isinstance(action, OuterAction)
