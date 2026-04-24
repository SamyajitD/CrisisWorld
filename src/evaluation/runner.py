"""ExperimentRunner — multi-seed, multi-condition episode orchestration."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from src.evaluation.ablations import AblationCondition, build_conditions
from src.schemas.config import ExperimentConfig
from src.schemas.episode import EpisodeResult, LogEvent

logger = logging.getLogger(__name__)


class ExperimentResults(BaseModel):
    """Results from a full experiment run."""

    model_config = ConfigDict(frozen=False)

    conditions: dict[str, list[EpisodeResult]] = {}


class ExperimentRunner:
    """Orchestrates multi-seed episodes across ablation conditions."""

    def __init__(
        self,
        env_factory: Callable[[], Any],
        agent_factory: Callable[[AblationCondition], Any],
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
                agent = self._agent_factory(cond)
                ep_logger = self._logger_factory()
                try:
                    result = self.run_episode(
                        env=env, agent=agent, logger=ep_logger, seed=seed
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
    ) -> EpisodeResult:
        """Run a single episode. Returns EpisodeResult."""
        episode_id = f"ep-{uuid.uuid4().hex[:8]}"
        obs = env.reset(seed)
        agent.reset()

        total_reward = 0.0
        turn = 0
        done = False
        termination_reason = "max_turns"

        while not done:
            action = agent.act(obs)
            step_result = env.step(action)

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

            reward_total = step_result.reward.total
            total_reward += reward_total

            logger.record(
                LogEvent(
                    kind="reward",
                    turn=turn,
                    data={"total": reward_total},
                )
            )

            obs = step_result.observation
            done = step_result.done
            turn += 1

            if done:
                termination_reason = step_result.info.get(
                    "termination_reason", "done"
                )

        return EpisodeResult(
            episode_id=episode_id,
            seed=seed,
            condition="",
            total_turns=turn,
            total_reward=total_reward,
            termination_reason=termination_reason,
        )
