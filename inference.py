"""Composition root — wires CrisisWorld + agents + evaluation.

Usage:
    python inference.py --agent flat --seed 42
    python inference.py --agent cortex --seed 42
    python inference.py --experiment configs/experiment_ablation.yaml
"""

from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
from evaluation.ablations import build_conditions
from evaluation.analysis import comparison_table, diagnostic_report
from evaluation.runner import ExperimentRunner
from models import EnvConfig
from schemas.config import CortexConfig, ExperimentConfig
from server import CrisisWorld
from tracing.tracer import EpisodeTracer

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

TRACE_DIR = Path("traces")
RESULT_DIR = Path("results")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_env(config: EnvConfig) -> CrisisWorld:
    return CrisisWorld(config=config)


def _make_flat_agent(config: EnvConfig, seed: int) -> FlatAgent:
    rng = np.random.default_rng(seed)
    return FlatAgent(config=config, rng=rng)


def _make_cortex_agent(cond: Any, cortex_config: CortexConfig) -> CortexAgent:
    """Build CortexAgent configured per the AblationCondition."""
    all_role_map = {
        "perception": PerceptionRole,
        "world_modeler": WorldModelerRole,
        "planner": PlannerRole,
        "critic": CriticRole,
        "executive": ExecutiveRole,
    }
    # Use condition's enabled_roles; always include perception + executive
    enabled = set(cond.enabled_roles) if hasattr(cond, "enabled_roles") else set(all_role_map.keys())
    enabled.add("perception")
    enabled.add("executive")
    roles = {name: cls() for name, cls in all_role_map.items() if name in enabled}

    memory = EpisodeMemory() if getattr(cond, "memory_enabled", True) else EpisodeMemory()
    episode_id = f"ep-{uuid.uuid4().hex[:8]}"
    ep_logger = EpisodeTracer(episode_id=episode_id)
    budget_total = getattr(cond, "budget", cortex_config.total_budget)
    budget = BudgetTracker(budget_total)
    deliberator = CortexDeliberator(roles=roles, memory=memory, logger=ep_logger)
    return CortexAgent(
        deliberator=deliberator,
        budget=budget,
        ep_logger=ep_logger,
        initial_budget=budget_total,
    )


def _make_logger() -> EpisodeTracer:
    episode_id = f"ep-{uuid.uuid4().hex[:8]}"
    return EpisodeTracer(episode_id=episode_id)


def _build_agent(
    cond: Any,
    env_config: EnvConfig,
    cortex_config: CortexConfig,
    seed: int,
) -> FlatAgent | CortexAgent:
    """Build the right agent based on the condition's agent_type."""
    agent_type = getattr(cond, "agent_type", "flat")
    if agent_type == "cortex":
        return _make_cortex_agent(cond, cortex_config)
    return _make_flat_agent(env_config, seed)


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------


def _load_experiment_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    env_raw = raw.get("env_config", {})
    cortex_raw = raw.get("cortex_config", {})
    return ExperimentConfig(
        seeds=tuple(raw.get("seeds", [42])),
        conditions=tuple(raw.get("conditions", [
            "flat-lite", "flat-fat",
            "cortex-lite", "cortex-full", "cortex-tuned",
        ])),
        env_config=EnvConfig(**env_raw) if env_raw else EnvConfig(),
        cortex_config=CortexConfig(**cortex_raw) if cortex_raw else CortexConfig(),
        output_dir=raw.get("output_dir", "results"),
        trace_dir=raw.get("trace_dir", "traces"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CrisisWorld + Cortex — outbreak-control experiments"
    )
    parser.add_argument(
        "--agent", choices=["flat", "cortex"], default="flat",
        help="Agent type for single-run mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Path to experiment YAML config (runs full ablation)",
    )
    args = parser.parse_args()

    if args.experiment:
        exp_config = _load_experiment_config(args.experiment)
    else:
        exp_config = ExperimentConfig(
            seeds=(args.seed,),
            conditions=(
                f"{args.agent}-lite", f"{args.agent}-fat",
            ) if args.agent == "flat" else (
                f"{args.agent}-lite", f"{args.agent}-full",
            ),
            env_config=EnvConfig(),
            cortex_config=CortexConfig(),
        )

    env_config = exp_config.env_config
    cortex_config = exp_config.cortex_config

    runner = ExperimentRunner(
        env_factory=lambda: _make_env(env_config),
        agent_factory=lambda cond: _build_agent(
            cond, env_config, cortex_config, args.seed
        ),
        logger_factory=_make_logger,
        config=exp_config,
    )

    logger.info("Running experiment: conditions=%s seeds=%s", exp_config.conditions, exp_config.seeds)
    results = runner.run()

    # Save results
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table = comparison_table(results)
    report = diagnostic_report(results)
    result_path = RESULT_DIR / "comparison.md"
    result_path.write_text(f"{table}\n\n{report}\n", encoding="utf-8")
    logger.info("Results saved to %s", result_path)

    # Print summary
    print()
    print(table)
    print()

    for cond_name, episodes in results.conditions.items():
        for ep in episodes:
            logger.info(
                "Condition=%s seed=%d turns=%d reward=%.2f reason=%s",
                cond_name, ep.seed, ep.total_turns, ep.total_reward,
                ep.termination_reason,
            )


if __name__ == "__main__":
    main()
