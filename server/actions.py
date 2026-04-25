"""Action validation and delayed-effect scheduling."""

from __future__ import annotations

import numpy as np

from models import (
    DeployResource,
    Escalate,
    NoOp,
    OuterAction,
    PublicCommunication,
    ReallocateBudget,
    RequestData,
    RestrictMovement,
)
from server._internal import InternalState, ScheduledEffect


def _region_exists(region_id: str, state: InternalState) -> bool:
    return any(r.region_id == region_id for r in state.regions)


def validate_and_schedule(
    action: OuterAction,
    state: InternalState,
    rng: np.random.Generator,
) -> tuple[list[str], tuple[ScheduledEffect, ...]]:
    """Validate *action* and return (errors, scheduled_effects)."""
    errors: list[str] = []
    effects: list[ScheduledEffect] = []

    if isinstance(action, DeployResource):
        if not _region_exists(action.region_id, state):
            errors.append(f"Unknown region: {action.region_id}")
        if action.amount <= 0:
            errors.append("Deploy amount must be > 0")
        if not errors:
            effects.append(ScheduledEffect(
                apply_on_turn=state.turn + 2,
                effect_type=action.kind,
                target_region=action.region_id,
                payload=action.model_dump(),
            ))

    elif isinstance(action, RestrictMovement):
        if not _region_exists(action.region_id, state):
            errors.append(f"Unknown region: {action.region_id}")
        if not errors:
            effects.append(ScheduledEffect(
                apply_on_turn=state.turn + 1,
                effect_type=action.kind,
                target_region=action.region_id,
                payload=action.model_dump(),
            ))

    elif isinstance(action, RequestData):
        if not action.source:
            errors.append("Request source must be non-empty")
        if not errors:
            effects.append(ScheduledEffect(
                apply_on_turn=state.turn + 1,
                effect_type=action.kind,
                target_region=None,
                payload=action.model_dump(),
            ))

    elif isinstance(action, PublicCommunication):
        effects.append(ScheduledEffect(
            apply_on_turn=state.turn,
            effect_type=action.kind,
            target_region=None,
            payload=action.model_dump(),
        ))

    elif isinstance(action, Escalate):
        if not action.agency:
            errors.append("Escalate agency must be non-empty")
        if not errors:
            effects.append(ScheduledEffect(
                apply_on_turn=state.turn + 2,
                effect_type=action.kind,
                target_region=None,
                payload=action.model_dump(),
            ))

    elif isinstance(action, ReallocateBudget):
        effects.append(ScheduledEffect(
            apply_on_turn=state.turn,
            effect_type=action.kind,
            target_region=None,
            payload=action.model_dump(),
        ))

    elif isinstance(action, NoOp):
        pass  # No validation, no effects

    return errors, tuple(effects)
