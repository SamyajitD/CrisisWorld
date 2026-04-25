"""ExperimentRunner — multi-seed, multi-condition episode orchestration."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from .ablations import AblationCondition, build_conditions
from ..schemas.config import ExperimentConfig
from ..schemas.episode import EpisodeResult, LogEvent

_log = logging.getLogger(__name__)

logger = _log


class ExperimentResults(BaseModel):
    """Results from a full experiment run."""

    model_config = ConfigDict(frozen=False)

    conditions: dict[str, list[EpisodeResult]] = {}


class ExperimentRunner:
    """Orchestrates multi-seed episodes across ablation conditions."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[AblationCondition, int], Any],
        logger_factory: Callable[[], Any],
        config: ExperimentConfig,
    ) -> None:
        self._env_factory = env_factory
        self._agent_factory = agent_factory
        self._logger_factory = logger_factory
        self._config = config

    def run(self) -> ExperimentResults:
        """Run all conditions x seeds. Returns ExperimentResults."""
        conditions = build_conditions(self._config)
        results = ExperimentResults()

        for cond in conditions:
            episodes: list[EpisodeResult] = []
            for seed in self._config.seeds:
                env = self._env_factory()
                agent = self._agent_factory(cond, seed)
                ep_logger = self._logger_factory()
                try:
                    result = self.run_episode(
                        env=env, agent=agent, logger=ep_logger, seed=seed,
                        condition=cond.name,
                    )
                    episodes.append(result)
                except Exception:
                    logger.warning(
                        "Episode crashed: condition=%s seed=%s",
                        cond.name,
                        seed,
                        exc_info=True,
                    )
                finally:
                    try:
                        env.close()
                    except Exception:
                        logger.warning(
                            "env.close() failed: condition=%s",
                            cond.name,
                            exc_info=True,
                        )
            results.conditions[cond.name] = episodes

        return results

    def run_episode(
        self,
        env: Any,
        agent: Any,
        logger: Any,
        seed: int,
        condition: str = "",
    ) -> EpisodeResult:
        """Run a single episode. Returns EpisodeResult."""
        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        # Set episode metadata on logger via public API
        if hasattr(logger, "set_episode"):
            logger.set_episode(episode_id, seed)
        # Inject runner's logger into agent so all events go to one place
        if hasattr(agent, "set_logger"):
            agent.set_logger(logger)
        obs = env.reset(seed, episode_id=episode_id)
        agent.reset()

        total_reward = 0.0
        turn = 0
        done = False
        termination_reason = "max_turns"
        action_counts: dict[str, int] = {}

        while not done:
            action = agent.act(obs)
            action_counts[action.kind] = action_counts.get(action.kind, 0) + 1
            obs = env.step(action)

            logger.record(
                LogEvent(
                    kind="observation",
                    turn=turn,
                    data={"turn": turn},
                )
            )
            logger.record(
                LogEvent(
                    kind="action",
                    turn=turn,
                    data={"kind": action.kind},
                )
            )

            reward_total = obs.reward if obs.reward is not None else 0.0
            total_reward += reward_total

            logger.record(
                LogEvent(
                    kind="reward",
                    turn=turn,
                    data={"total": reward_total},
                )
            )

            done = obs.done
            turn += 1

            if done:
                termination_reason = obs.metadata.get(
                    "termination_reason", "done"
                )

        try:
            trace_dir = Path(self._config.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            logger.save(trace_dir / f"{episode_id}.json")
        except Exception:
            _log.warning("Failed to save trace for %s", episode_id, exc_info=True)

        return EpisodeResult(
            episode_id=episode_id,
            seed=seed,
            condition=condition,
            total_turns=turn,
            total_reward=total_reward,
            termination_reason=termination_reason,
            metrics={
                "total_cumulative_reward": total_reward,
                "action_counts": action_counts,
            },
        )
