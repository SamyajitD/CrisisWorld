"""Unit tests for cortex/budget.py and cortex/memory.py per CLAUDE.md spec."""

from __future__ import annotations

import pytest

from CrisisWorld.schemas.budget import BudgetExhaustedError, BudgetStatus


# ---------------------------------------------------------------------------
# BudgetTracker tests (18)
# ---------------------------------------------------------------------------


class TestBudgetTracker:
    def _make(self, total: int = 10):
        from CrisisWorld.cortex.budget import BudgetTracker
        return BudgetTracker(total)

    def test_initial_state_reflects_total(self) -> None:
        b = self._make(10)
        s = b.remaining()
        assert s.total == 10 and s.spent == 0 and s.remaining == 10
        assert not b.is_exhausted()

    def test_charge_deducts_from_remaining(self) -> None:
        b = self._make(10)
        b.charge(3)
        assert b.remaining().remaining == 7

    def test_charge_multiple_times_accumulates(self) -> None:
        b = self._make(10)
        b.charge(2); b.charge(3); b.charge(1)
        assert b.remaining().spent == 6

    def test_charge_exact_remaining_exhausts(self) -> None:
        b = self._make(5)
        b.charge(5)
        assert b.is_exhausted()

    def test_charge_exceeding_remaining_raises(self) -> None:
        b = self._make(5)
        with pytest.raises(BudgetExhaustedError):
            b.charge(6)
        assert b.remaining().spent == 0

    def test_charge_exceeding_by_one_raises(self) -> None:
        b = self._make(5)
        b.charge(3)
        with pytest.raises(BudgetExhaustedError):
            b.charge(3)
        assert b.remaining().spent == 3

    def test_charge_zero_is_noop(self) -> None:
        b = self._make(10)
        b.charge(0)
        assert b.remaining().spent == 0
        assert len(b.get_ledger()) == 0

    def test_charge_negative_raises_value_error(self) -> None:
        b = self._make(10)
        with pytest.raises(ValueError):
            b.charge(-1)

    def test_is_exhausted_false_when_remaining(self) -> None:
        b = self._make(5)
        b.charge(3)
        assert not b.is_exhausted()

    def test_is_exhausted_true_at_zero(self) -> None:
        b = self._make(5)
        b.charge(5)
        assert b.is_exhausted()

    def test_reset_restores_full_budget(self) -> None:
        b = self._make(10)
        b.charge(3)
        b.reset(10)
        assert b.remaining().remaining == 10 and b.remaining().spent == 0

    def test_reset_clears_ledger(self) -> None:
        b = self._make(10)
        b.charge(2)
        b.reset(10)
        assert len(b.get_ledger()) == 0

    def test_double_reset_is_safe(self) -> None:
        b = self._make(10)
        b.reset(10)
        b.reset(7)
        assert b.remaining().total == 7

    def test_reset_with_zero_raises_value_error(self) -> None:
        b = self._make(10)
        with pytest.raises(ValueError):
            b.reset(0)

    def test_reset_with_negative_raises_value_error(self) -> None:
        b = self._make(10)
        with pytest.raises(ValueError):
            b.reset(-5)

    def test_ledger_records_each_charge(self) -> None:
        b = self._make(10)
        b.charge(2); b.charge(3)
        ledger = b.get_ledger()
        assert len(ledger) == 2
        assert ledger[0]["cost"] == 2
        assert ledger[1]["cost"] == 3

    def test_charge_after_exhaustion_raises(self) -> None:
        b = self._make(5)
        b.charge(5)
        with pytest.raises(BudgetExhaustedError):
            b.charge(1)

    def test_remaining_returns_budget_status_type(self) -> None:
        b = self._make(10)
        assert isinstance(b.remaining(), BudgetStatus)


# ---------------------------------------------------------------------------
# EpisodeMemory tests (12)
# ---------------------------------------------------------------------------


class TestEpisodeMemory:
    def _make(self):
        from CrisisWorld.cortex.memory import EpisodeMemory
        return EpisodeMemory()

    def _artifact(self):
        from CrisisWorld.schemas.artifact import CleanState
        return CleanState()

    def test_store_and_retrieve_single(self) -> None:
        m = self._make()
        a = self._artifact()
        m.store("perception", a)
        assert m.retrieve("perception") == [a]

    def test_retrieve_nonexistent_key_returns_empty(self) -> None:
        m = self._make()
        assert m.retrieve("missing") == []

    def test_store_same_key_multiple_times(self) -> None:
        m = self._make()
        for _ in range(5):
            m.store("p", self._artifact())
        assert len(m.retrieve("p")) == 5

    def test_store_same_key_100_times(self) -> None:
        m = self._make()
        for _ in range(100):
            m.store("bulk", self._artifact())
        assert len(m.retrieve("bulk")) == 100

    def test_store_different_keys_isolated(self) -> None:
        m = self._make()
        m.store("a", self._artifact())
        m.store("b", self._artifact())
        assert len(m.retrieve("a")) == 1
        assert len(m.retrieve("b")) == 1

    def test_digest_on_empty_memory(self) -> None:
        m = self._make()
        d = m.digest()
        assert d.num_entries == 0 and d.keys == ()

    def test_digest_reflects_stored_counts(self) -> None:
        m = self._make()
        m.store("a", self._artifact())
        m.store("a", self._artifact())
        m.store("b", self._artifact())
        d = m.digest()
        assert d.num_entries == 3
        assert set(d.keys) == {"a", "b"}

    def test_reset_clears_all(self) -> None:
        m = self._make()
        m.store("a", self._artifact())
        m.reset()
        assert m.retrieve("a") == []
        assert m.digest().num_entries == 0

    def test_store_after_reset(self) -> None:
        m = self._make()
        m.store("a", self._artifact())
        m.reset()
        m.store("b", self._artifact())
        assert m.retrieve("a") == []
        assert len(m.retrieve("b")) == 1

    def test_retrieve_returns_copy(self) -> None:
        m = self._make()
        m.store("a", self._artifact())
        result = m.retrieve("a")
        result.clear()
        assert len(m.retrieve("a")) == 1

    def test_empty_key_raises_value_error(self) -> None:
        m = self._make()
        with pytest.raises(ValueError):
            m.store("", self._artifact())

    def test_digest_keys_are_sorted(self) -> None:
        m = self._make()
        m.store("c", self._artifact())
        m.store("a", self._artifact())
        m.store("b", self._artifact())
        assert m.digest().keys == ("a", "b", "c")
