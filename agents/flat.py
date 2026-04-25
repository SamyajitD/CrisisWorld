"""FlatAgent — priority-ordered heuristic policy, no deliberation."""

from __future__ import annotations

import logging as stdlib_logging
from typing import Any

import numpy as np

from models import (
    DeployResource,
    EnvConfig,
    Escalate,
    NoOp,
    Observation,
    OuterAction,
    PublicCommunication,
    ReallocateBudget,
    RegionState,
    RequestData,
    RestrictMovement,
)

logger = stdlib_logging.getLogger(__name__)


class FlatAgent:
    """Heuristic policy with 8-rule priority cascade."""

    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:
        self._config = config
        self._rng = rng
        self._turn_count = 0
        self._last_action_kind: str | None = None
        self._escalated = False
        self._data_requested = False

    def act(self, observation: Observation) -> OuterAction:
        if observation.done:
            return NoOp()

        regions = observation.regions
        resources = observation.resources
        signals = observation.stakeholder_signals

        action = self._cascade(regions, resources, signals, observation)
        self._last_action_kind = action.kind
        self._turn_count += 1
        return action

    def reset(self) -> None:
        self._turn_count = 0
        self._last_action_kind = None
        self._escalated = False
        self._data_requested = False

    def _cascade(
        self,
        regions: tuple[RegionState, ...],
        resources: Any,
        signals: Any,
        obs: Observation,
    ) -> OuterAction:
        if not regions:
            return NoOp()

        worst = _worst_region(regions)
        worst_rate = _infection_rate(worst) if worst else 0.0

        # 1. Critical infection surge
        if worst_rate > 0.15 and resources.medical > 0:
            amount = _pick_deploy_amount(resources.medical, worst)
            if amount > 0:
                return DeployResource(
                    resource="medical", region_id=worst.region_id, amount=amount
                )

        # 2. Resource depletion
        if resources.medical == 0 and resources.funding > 0:
            return ReallocateBudget(
                from_category="funding", to_category="medical",
                amount=max(1, resources.funding // 3),
            )
        if resources.personnel == 0 and resources.funding > 0:
            return ReallocateBudget(
                from_category="funding", to_category="personnel",
                amount=max(1, resources.funding // 3),
            )

        # 3. High-urgency signal -> escalate once
        if not self._escalated:
            for sig in signals:
                if sig.urgency >= 0.8:
                    self._escalated = True
                    return Escalate(agency=sig.source)

        # 4. Movement restriction
        if worst_rate > 0.08 and not worst.restricted and self._last_action_kind != "restrict_movement":
            level = 1 if worst_rate < 0.12 else 2
            return RestrictMovement(region_id=worst.region_id, level=level)

        # 5. Information gap
        if self._turn_count > 0 and not self._data_requested:
            tel = obs.telemetry
            region_infected = sum(r.infected for r in regions)
            if tel.total_infected > 0 and abs(tel.total_infected - region_infected) / max(tel.total_infected, 1) > 0.1:
                self._data_requested = True
                return RequestData(source="telemetry")

        # 6. Periodic communication
        if self._turn_count > 0 and self._turn_count % 5 == 0:
            if any(_infection_rate(r) > 0.05 for r in regions):
                return PublicCommunication(audience="public", message="situation_update")

        # 7. Proactive deployment
        if worst_rate > 0.03 and resources.medical > 0 and self._last_action_kind != "deploy_resource":
            amount = max(1, resources.medical // 4)
            return DeployResource(
                resource="medical", region_id=worst.region_id, amount=amount
            )

        # 8. Fallback
        return NoOp()


def _worst_region(regions: tuple[RegionState, ...]) -> RegionState | None:
    if not regions:
        return None
    return max(regions, key=_infection_rate)


def _infection_rate(region: RegionState) -> float:
    if region.population == 0:
        return 0.0
    return region.infected / region.population


def _pick_deploy_amount(available: int, region: RegionState) -> int:
    needed = max(1, region.infected // 10)
    return min(available // 2, needed) or min(available, 1)
