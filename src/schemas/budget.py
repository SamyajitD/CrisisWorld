"""Budget schemas for the Cortex deliberation system."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BudgetStatus(BaseModel):
    """Snapshot of budget state."""

    model_config = ConfigDict(frozen=True)

    total: int = Field(ge=0)
    spent: int = Field(ge=0)
    remaining: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_invariant(self) -> BudgetStatus:
        if self.remaining != self.total - self.spent:
            msg = (
                f"remaining ({self.remaining}) must equal "
                f"total - spent ({self.total - self.spent})"
            )
            raise ValueError(msg)
        return self


class LedgerEntry(BaseModel):
    """Single charge record in the ledger."""

    model_config = ConfigDict(frozen=True)

    role_name: str
    cost: int = Field(gt=0)
    turn: int = Field(ge=0)


class BudgetLedger(BaseModel):
    """Full budget history with entries."""

    model_config = ConfigDict(frozen=True)

    entries: tuple[LedgerEntry, ...] = ()
    status: BudgetStatus


class BudgetExhaustedError(Exception):
    """Raised when a role call would exceed remaining budget."""

    def __init__(self, requested: int, remaining: int) -> None:
        self.requested = requested
        self.remaining = remaining
        super().__init__(
            f"Budget exhausted: requested {requested}, remaining {remaining}"
        )
