"""Unit tests for server/resources.py — resource pool operations."""

from __future__ import annotations

from models import ResourcePool


class TestResources:
    def test_apply_positive_delta(self) -> None:
        from server.resources import apply_resource_change

        pool = ResourcePool(medical=100, personnel=50, funding=200)
        result = apply_resource_change(pool, medical=10, personnel=5, funding=20)
        assert result.medical == 110
        assert result.personnel == 55
        assert result.funding == 220

    def test_negative_clamp(self) -> None:
        from server.resources import apply_resource_change

        pool = ResourcePool(medical=5, personnel=0, funding=10)
        result = apply_resource_change(pool, medical=-100, personnel=-50, funding=-20)
        assert result.medical == 0
        assert result.personnel == 0
        assert result.funding == 0

    def test_check_sufficient_passes(self) -> None:
        from server.resources import check_sufficient

        pool = ResourcePool(medical=100, personnel=50, funding=200)
        violations = check_sufficient(pool, medical=10)
        assert violations == []

    def test_check_sufficient_fails(self) -> None:
        from server.resources import check_sufficient

        pool = ResourcePool(medical=5)
        violations = check_sufficient(pool, medical=10)
        assert "medical" in violations[0]

    def test_turn_decay(self) -> None:
        from server.resources import apply_turn_decay

        pool = ResourcePool(medical=100, personnel=100, funding=100)
        decayed = apply_turn_decay(pool)
        assert decayed.medical == 98  # -2%
        assert decayed.funding == 99  # -1%
        assert decayed.personnel == 100  # unchanged

    def test_zero_pool_stays_zero(self) -> None:
        from server.resources import apply_resource_change

        pool = ResourcePool()
        result = apply_resource_change(pool, medical=-10, personnel=-10, funding=-10)
        assert result.medical == 0
        assert result.personnel == 0
        assert result.funding == 0
