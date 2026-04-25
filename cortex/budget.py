"""BudgetTracker — implements BudgetProtocol for deliberation cost accounting."""

from __future__ import annotations

from typing import Any

from schemas.budget import BudgetExhaustedError, BudgetStatus


class BudgetTracker:
    """Tracks and enforces deliberation compute budget."""

    def __init__(self, total_budget: int) -> None:
        if total_budget <= 0:
            raise ValueError("total_budget must be > 0")
        self._total = total_budget
        self._spent = 0
        self._ledger: list[dict[str, Any]] = []

    def charge(self, cost: int) -> None:
        """Deduct cost. Raises BudgetExhaustedError if insufficient."""
        if cost < 0:
            raise ValueError("cost must be >= 0")
        if cost == 0:
            return
        remaining = self._total - self._spent
        if cost > remaining:
            raise BudgetExhaustedError(requested=cost, remaining=remaining)
        self._spent += cost
        self._ledger.append({"cost": cost, "spent_after": self._spent})

    def remaining(self) -> BudgetStatus:
        return BudgetStatus(
            total=self._total,
            spent=self._spent,
            remaining=self._total - self._spent,
        )

    def is_exhausted(self) -> bool:
        return self._total - self._spent == 0

    def reset(self, total: int) -> None:
        if total <= 0:
            raise ValueError("total must be > 0")
        self._total = total
        self._spent = 0
        self._ledger = []

    def get_ledger(self) -> list[dict[str, Any]]:
        return list(self._ledger)
