"""Resource pool tracking for CrisisWorld."""

from __future__ import annotations

from ..models import ResourcePool


def apply_resource_change(
    pool: ResourcePool,
    medical: int = 0,
    personnel: int = 0,
    funding: int = 0,
) -> ResourcePool:
    """Apply deltas to pool, clamping each field to >= 0."""
    return ResourcePool(
        medical=max(0, pool.medical + medical),
        personnel=max(0, pool.personnel + personnel),
        funding=max(0, pool.funding + funding),
    )


def check_sufficient(
    pool: ResourcePool,
    medical: int = 0,
    personnel: int = 0,
    funding: int = 0,
) -> list[str]:
    """Return list of violation descriptions for insufficient resources."""
    violations: list[str] = []
    if medical > pool.medical:
        violations.append(
            f"medical: need {medical}, have {pool.medical}"
        )
    if personnel > pool.personnel:
        violations.append(
            f"personnel: need {personnel}, have {pool.personnel}"
        )
    if funding > pool.funding:
        violations.append(
            f"funding: need {funding}, have {pool.funding}"
        )
    return violations


def apply_turn_decay(pool: ResourcePool) -> ResourcePool:
    """Per-turn resource decay: medical -2%, funding -1%, personnel unchanged."""
    return ResourcePool(
        medical=int(pool.medical * 0.98),
        personnel=pool.personnel,
        funding=int(pool.funding * 0.99),
    )
