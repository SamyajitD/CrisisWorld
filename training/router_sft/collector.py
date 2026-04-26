"""Collect router-training data from heuristic Cortex rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .bootstrap import ensure_crisisworld_package
from .labels import canonical_route_label, normalize_sft_target

LOG = logging.getLogger(__name__)


@dataclass
class RouterExample:
    """One supervised router example captured at Executive invocation time."""

    episode_id: str
    seed: int
    outer_turn: int
    inner_iteration: int
    input_payload: dict[str, Any]
    teacher_decision: dict[str, Any]
    route_label: str
    final_action: dict[str, Any] | None = None
    immediate_reward: float | None = None
    return_to_go: float | None = None
    episode_total_reward: float | None = None
    termination_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "outer_turn": self.outer_turn,
            "inner_iteration": self.inner_iteration,
            "input_payload": self.input_payload,
            "teacher_decision": self.teacher_decision,
            "route_label": self.route_label,
            "final_action": self.final_action,
            "immediate_reward": self.immediate_reward,
            "return_to_go": self.return_to_go,
            "episode_total_reward": self.episode_total_reward,
            "termination_reason": self.termination_reason,
        }


class RecordingExecutiveRole:
    """Heuristic Executive wrapper that records training rows."""

    def __init__(self, teacher: Any) -> None:
        self._teacher = teacher
        self._episode_id = ""
        self._seed = 0
        self._outer_turn = -1
        self._inner_iteration = 0
        self._examples: list[RouterExample] = []

    @property
    def role_name(self) -> str:
        return self._teacher.role_name

    @property
    def cost(self) -> int:
        return self._teacher.cost

    def start_episode(self, episode_id: str, seed: int) -> None:
        self._episode_id = episode_id
        self._seed = seed
        self._outer_turn = -1
        self._inner_iteration = 0
        self._examples = []

    def invoke(self, role_input: Any) -> Any:
        payload = copy.deepcopy(role_input.payload)
        if _is_first_executive_call(payload.get("artifacts", [])):
            self._outer_turn += 1
            self._inner_iteration = 0
        else:
            self._inner_iteration += 1

        decision = self._teacher.invoke(role_input)
        decision_dict = normalize_sft_target(decision.model_dump())
        self._examples.append(
            RouterExample(
                episode_id=self._episode_id,
                seed=self._seed,
                outer_turn=self._outer_turn,
                inner_iteration=self._inner_iteration,
                input_payload=payload,
                teacher_decision=decision_dict,
                route_label=canonical_route_label(decision_dict),
            )
        )
        return decision

    def finalize_episode(
        self,
        rewards_by_turn: list[float],
        actions_by_turn: list[dict[str, Any]],
        termination_reason: str,
    ) -> list[dict[str, Any]]:
        suffix_returns = _suffix_sums(rewards_by_turn)
        total_reward = sum(rewards_by_turn)

        rows: list[dict[str, Any]] = []
        for example in self._examples:
            reward = 0.0
            rtg = 0.0
            action = None
            if 0 <= example.outer_turn < len(rewards_by_turn):
                reward = rewards_by_turn[example.outer_turn]
                rtg = suffix_returns[example.outer_turn]
                action = actions_by_turn[example.outer_turn]

            example.immediate_reward = reward
            example.return_to_go = rtg
            example.episode_total_reward = total_reward
            example.final_action = action
            example.termination_reason = termination_reason
            rows.append(example.to_dict())
        return rows


def collect_router_dataset(
    output_path: Path,
    *,
    num_episodes: int,
    seed_start: int,
    budget: int,
    env_config: Any,
) -> dict[str, Any]:
    """Run heuristic Cortex episodes and write one raw JSONL row per Executive call."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    route_counts: dict[str, int] = {}
    rows_written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for offset in range(num_episodes):
            seed = seed_start + offset
            episode_rows = _run_single_episode(
                seed=seed,
                budget=budget,
                env_config=env_config,
            )
            for row in episode_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1
                label = str(row["route_label"])
                route_counts[label] = route_counts.get(label, 0) + 1

    return {
        "episodes": num_episodes,
        "rows": rows_written,
        "route_counts": route_counts,
        "output_path": str(output_path),
    }


def _run_single_episode(
    *,
    seed: int,
    budget: int,
    env_config: Any,
) -> list[dict[str, Any]]:
    ensure_crisisworld_package()

    from CrisisWorld.agents.cortex_agent import CortexAgent
    from CrisisWorld.cortex.budget import BudgetTracker
    from CrisisWorld.cortex.deliberator import CortexDeliberator
    from CrisisWorld.cortex.memory import EpisodeMemory
    from CrisisWorld.cortex.roles import (
        CriticRole,
        ExecutiveRole,
        PerceptionRole,
        PlannerRole,
        WorldModelerRole,
    )
    from CrisisWorld.server import CrisisWorld as CrisisWorldEnvironment
    from CrisisWorld.tracing.tracer import EpisodeTracer

    env = CrisisWorldEnvironment(config=env_config)
    tracer = EpisodeTracer(episode_id="pending", seed=seed)
    teacher = RecordingExecutiveRole(teacher=ExecutiveRole())
    episode_id = f"router-{uuid4().hex[:12]}"
    teacher.start_episode(episode_id=episode_id, seed=seed)

    roles: dict[str, Any] = {
        "perception": PerceptionRole(),
        "world_modeler": WorldModelerRole(),
        "planner": PlannerRole(),
        "critic": CriticRole(),
        "executive": teacher,
    }
    deliberator = CortexDeliberator(
        roles=roles,
        memory=EpisodeMemory(),
        logger=tracer,
    )
    agent = CortexAgent(
        deliberator=deliberator,
        budget=BudgetTracker(budget),
        ep_logger=tracer,
        initial_budget=budget,
    )

    rewards_by_turn: list[float] = []
    actions_by_turn: list[dict[str, Any]] = []
    termination_reason = "max_turns"

    try:
        observation = env.reset(seed=seed, episode_id=episode_id)
        tracer.set_episode(episode_id, seed)
        agent.reset()

        while not observation.done:
            action = agent.act(observation)
            actions_by_turn.append(action.model_dump())
            observation = env.step(action)
            rewards_by_turn.append(float(observation.reward or 0.0))
            if observation.done:
                termination_reason = str(
                    observation.metadata.get("termination_reason", "done")
                )
    finally:
        env.close()

    return teacher.finalize_episode(
        rewards_by_turn=rewards_by_turn,
        actions_by_turn=actions_by_turn,
        termination_reason=termination_reason,
    )


def _is_first_executive_call(artifacts: list[dict[str, Any]]) -> bool:
    return (
        len(artifacts) == 1
        and isinstance(artifacts[0], dict)
        and "cleaned_observation" in artifacts[0]
    )


def _suffix_sums(values: list[float]) -> list[float]:
    suffixes = [0.0] * len(values)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running += values[index]
        suffixes[index] = running
    return suffixes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Raw JSONL output path",
    )
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--num-regions", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--initial-infected", type=int, default=10)
    parser.add_argument("--noise-level", type=float, default=0.1)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(name)s %(levelname)s: %(message)s",
    )

    ensure_crisisworld_package()
    from CrisisWorld.models import EnvConfig

    env_config = EnvConfig(
        num_regions=args.num_regions,
        max_turns=args.max_turns,
        initial_infected=args.initial_infected,
        noise_level=args.noise_level,
    )
    summary = collect_router_dataset(
        output_path=args.output,
        num_episodes=args.num_episodes,
        seed_start=args.seed_start,
        budget=args.budget,
        env_config=env_config,
    )
    LOG.info("Collection finished: %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
