"""E2E test: full flat-agent episode from reset to termination."""

from __future__ import annotations

import numpy as np

from agents.flat import FlatAgent
from models import EnvConfig, NoOp
from server import CrisisWorld


class TestFlatEpisode:
    def test_flat_episode_runs_to_completion(self) -> None:
        config = EnvConfig(max_turns=20)
        env = CrisisWorld(config=config)
        agent = FlatAgent(config=config, rng=np.random.default_rng(42))

        obs = env.reset(seed=42)
        agent.reset()
        done = False
        turns = 0

        while not done and turns < 25:
            action = agent.act(obs)
            obs = env.step(action)
            done = obs.done
            turns += 1

        assert done is True
        assert turns <= 20
        assert env.state.step_count == turns

    def test_flat_episode_determinism(self) -> None:
        config = EnvConfig(max_turns=10)

        rewards1: list[float] = []
        env1 = CrisisWorld(config=config)
        a1 = FlatAgent(config=config, rng=np.random.default_rng(42))
        obs = env1.reset(seed=42)
        a1.reset()
        for _ in range(10):
            action = a1.act(obs)
            obs = env1.step(action)
            rewards1.append(obs.reward)
            if obs.done:
                break

        rewards2: list[float] = []
        env2 = CrisisWorld(config=config)
        a2 = FlatAgent(config=config, rng=np.random.default_rng(42))
        obs = env2.reset(seed=42)
        a2.reset()
        for _ in range(10):
            action = a2.act(obs)
            obs = env2.step(action)
            rewards2.append(obs.reward)
            if obs.done:
                break

        assert rewards1 == rewards2

    def test_flat_episode_produces_varied_actions(self) -> None:
        config = EnvConfig(max_turns=15)
        env = CrisisWorld(config=config)
        agent = FlatAgent(config=config, rng=np.random.default_rng(42))

        obs = env.reset(seed=42)
        agent.reset()
        action_kinds: set[str] = set()

        for _ in range(15):
            action = agent.act(obs)
            action_kinds.add(action.kind)
            obs = env.step(action)
            if obs.done:
                break

        # Should use more than just noop
        assert len(action_kinds) >= 2
