"""BudgetProtocol — contract for deliberation budget accounting."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..schemas.budget import BudgetStatus


@runtime_checkable
class BudgetProtocol(Protocol):
    """Tracks and enforces deliberation compute budget."""

    def charge(self, cost: int) -> None: ...

    def remaining(self) -> BudgetStatus: ...

    def is_exhausted(self) -> bool: ...

    def reset(self, total: int) -> None: ...
