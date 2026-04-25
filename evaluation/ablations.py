"""Ablation condition definitions for experiments."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from schemas.config import ExperimentConfig

logger = logging.getLogger(__name__)

_ALL_ROLES = (
    "perception",
    "world_modeler",
    "planner",
    "critic",
    "executive",
)
_LITE_ROLES = ("perception", "planner", "executive")

_LOW_BUDGET = 10
_MATCHED_BUDGET = 50


class AblationCondition(BaseModel):
    """Definition of one experimental condition."""

    model_config = ConfigDict(frozen=True)

    name: str
    agent_type: str
    budget: int = Field(ge=0)
    enabled_roles: tuple[str, ...] = ()
    memory_enabled: bool = False
    critic_enabled: bool = False
    tuned_executive: bool = False
    role_backend: str = "heuristic"  # "heuristic" or "llm"


def get_matched_budget(config: ExperimentConfig) -> int:
    """Return the matched (high) budget from config."""
    return config.cortex_config.total_budget or _MATCHED_BUDGET


def get_low_budget(config: ExperimentConfig) -> int:
    """Return the low budget — fraction of matched, minimum 1."""
    matched = get_matched_budget(config)
    return max(matched // 5, 1) if matched > 0 else _LOW_BUDGET


def _make_all_conditions(
    config: ExperimentConfig,
) -> dict[str, AblationCondition]:
    """Build the full registry of all 5 conditions."""
    matched = get_matched_budget(config)
    low = get_low_budget(config)

    return {
        "flat-lite": AblationCondition(
            name="flat-lite",
            agent_type="flat",
            budget=low,
        ),
        "flat-fat": AblationCondition(
            name="flat-fat",
            agent_type="flat",
            budget=matched,
        ),
        "cortex-lite": AblationCondition(
            name="cortex-lite",
            agent_type="cortex",
            budget=low,
            enabled_roles=_LITE_ROLES,
        ),
        "cortex-full": AblationCondition(
            name="cortex-full",
            agent_type="cortex",
            budget=matched,
            enabled_roles=_ALL_ROLES,
            memory_enabled=True,
            critic_enabled=True,
        ),
        "cortex-tuned": AblationCondition(
            name="cortex-tuned",
            agent_type="cortex",
            budget=matched,
            enabled_roles=_ALL_ROLES,
            memory_enabled=True,
            critic_enabled=True,
            tuned_executive=True,
        ),
        "cortex-llm": AblationCondition(
            name="cortex-llm",
            agent_type="cortex",
            budget=matched,
            enabled_roles=_ALL_ROLES,
            memory_enabled=True,
            critic_enabled=True,
            role_backend="llm",
        ),
    }


def build_conditions(
    config: ExperimentConfig,
) -> list[AblationCondition]:
    """Build ablation conditions filtered by config.conditions."""
    all_conds = _make_all_conditions(config)
    result: list[AblationCondition] = []
    seen: set[str] = set()

    for name in config.conditions:
        if name in seen:
            continue
        seen.add(name)
        if name in all_conds:
            result.append(all_conds[name])
        else:
            logger.warning("Unknown condition '%s', skipping", name)

    return result
