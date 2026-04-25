"""E2E test: full cortex-agent episode from reset to termination."""

from __future__ import annotations

from agents.cortex_agent import CortexAgent
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
from models import EnvConfig
from server import CrisisWorld
from tracing.tracer import EpisodeTracer


def _make_cortex_agent(budget_total: int = 50) -> CortexAgent:
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
    deliberator = CortexDeliberator(roles=roles, memory=memory, logger=ep_logger)
    return CortexAgent(
        deliberator=deliberator,
        budget=budget,
        ep_logger=ep_logger,
        initial_budget=budget_total,
    )


class TestCortexEpisode:
    def test_cortex_episode_runs_to_completion(self) -> None:
        config = EnvConfig(max_turns=10)
        env = CrisisWorld(config=config)
        agent = _make_cortex_agent(budget_total=50)

        obs = env.reset(seed=42)
        agent.reset()
        done = False
        turns = 0

        while not done and turns < 15:
            action = agent.act(obs)
            obs = env.step(action)
            done = obs.done
            turns += 1

        assert done is True
        assert turns <= 10

    def test_cortex_episode_uses_budget(self) -> None:
        config = EnvConfig(max_turns=5)
        env = CrisisWorld(config=config)
        agent = _make_cortex_agent(budget_total=20)

        obs = env.reset(seed=42)
        agent.reset()

        for _ in range(3):
            action = agent.act(obs)
            obs = env.step(action)
            if obs.done:
                break

        # Budget should have been consumed
        assert agent._budget.remaining().spent > 0

    def test_cortex_episode_survives_budget_exhaustion(self) -> None:
        config = EnvConfig(max_turns=10)
        env = CrisisWorld(config=config)
        # Tiny budget -- will exhaust quickly
        agent = _make_cortex_agent(budget_total=3)

        obs = env.reset(seed=42)
        agent.reset()

        for _ in range(10):
            action = agent.act(obs)
            obs = env.step(action)
            if obs.done:
                break

        # Should complete without crashing
        assert True
