"""Integration tests for CrisisWorld environment."""

from __future__ import annotations

import pytest

from CrisisWorld.models import CrisisState, EnvConfig, NoOp


class TestCrisisWorld:
    def _make_env(self, **kwargs):
        from CrisisWorld.server.CrisisWorld_environment import CrisisWorld
        return CrisisWorld(config=EnvConfig(**kwargs))

    def test_reset_returns_valid_observation(self) -> None:
        env = self._make_env()
        obs = env.reset(seed=42)
        assert obs.turn == 0
        assert len(obs.regions) > 0
        assert obs.done is False

    def test_step_returns_observation_with_done_reward(self) -> None:
        env = self._make_env()
        env.reset(seed=42)
        obs = env.step(NoOp())
        assert isinstance(obs.done, bool)
        assert obs.reward is not None

    def test_noop_sequence_progresses(self) -> None:
        env = self._make_env(max_turns=10)
        env.reset(seed=42)
        for i in range(5):
            obs = env.step(NoOp())
            assert obs.turn == i + 1

    def test_step_before_reset_raises(self) -> None:
        env = self._make_env()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(NoOp())

    def test_step_after_close_raises(self) -> None:
        env = self._make_env()
        env.reset(seed=42)
        env.close()
        with pytest.raises(RuntimeError, match="closed"):
            env.step(NoOp())

    def test_step_after_done_raises(self) -> None:
        env = self._make_env(max_turns=2)
        env.reset(seed=42)
        env.step(NoOp())
        obs = env.step(NoOp())  # turn 2 = max_turns -> done
        assert obs.done is True
        with pytest.raises(RuntimeError, match="done"):
            env.step(NoOp())

    def test_full_episode_determinism(self) -> None:
        rewards1 = []
        env1 = self._make_env(max_turns=5)
        env1.reset(seed=42)
        for _ in range(5):
            obs = env1.step(NoOp())
            rewards1.append(obs.reward)

        rewards2 = []
        env2 = self._make_env(max_turns=5)
        env2.reset(seed=42)
        for _ in range(5):
            obs = env2.step(NoOp())
            rewards2.append(obs.reward)

        assert rewards1 == rewards2

    def test_different_seeds_different_trajectories(self) -> None:
        env1 = self._make_env(max_turns=5)
        env1.reset(seed=42)
        obs1 = env1.step(NoOp())

        env2 = self._make_env(max_turns=5)
        env2.reset(seed=99)
        obs2 = env2.step(NoOp())

        # Different seeds should produce different region states
        r1 = obs1.regions[0]
        r2 = obs2.regions[0]
        assert r1.population != r2.population or r1.infected != r2.infected

    def test_state_property_returns_crisis_state(self) -> None:
        env = self._make_env()
        env.reset(seed=42, episode_id="test-ep")
        s = env.state
        assert isinstance(s, CrisisState)
        assert s.episode_id == "test-ep"
        assert s.step_count == 0

    def test_episode_id_persisted_across_steps(self) -> None:
        env = self._make_env()
        env.reset(seed=42, episode_id="ep-persist")
        env.step(NoOp())
        assert env.state.episode_id == "ep-persist"
        assert env.state.step_count == 1

    def test_get_metadata_returns_env_metadata(self) -> None:
        env = self._make_env()
        meta = env.get_metadata()
        assert meta.name == "CrisisWorld"
        assert meta.version is not None

    def test_reset_with_episode_id(self) -> None:
        env = self._make_env()
        obs = env.reset(seed=42, episode_id="ep-1")
        assert env.state.episode_id == "ep-1"
        assert obs.metadata.get("episode_id") == "ep-1"

    def test_close_idempotent(self) -> None:
        env = self._make_env()
        env.close()
        env.close()  # should not raise
