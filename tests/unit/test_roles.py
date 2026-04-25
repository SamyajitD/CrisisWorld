"""Unit tests for src/cortex/roles/ — all 5 Cortex specialist roles."""

from __future__ import annotations

from typing import Any

from schemas.artifact import (
    BeliefState,
    CandidateAction,
    CleanState,
    Critique,
    ExecutiveDecision,
    Plan,
    RoleInput,
)
from schemas.budget import BudgetStatus
from schemas.episode import MemoryDigest
from models import (
    IncidentReport,
    Observation,
    StakeholderSignal,
    Telemetry,
)
from models import Constraint, RegionState, ResourcePool

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _obs(
    turn: int = 0,
    regions: tuple[RegionState, ...] | None = None,
    signals: tuple[StakeholderSignal, ...] = (),
    incidents: tuple[IncidentReport, ...] = (),
    telemetry: Telemetry | None = None,
    resources: ResourcePool | None = None,
    constraints: tuple[Constraint, ...] = (),
) -> dict[str, Any]:
    """Build an observation dict suitable for RoleInput payload."""
    obs = Observation(
        turn=turn,
        regions=regions or (
            RegionState(
                region_id="north", population=1000, infected=50
            ),
        ),
        stakeholder_signals=signals,
        incidents=incidents,
        telemetry=telemetry or Telemetry(total_infected=50),
        resources=resources or ResourcePool(
            medical=100, personnel=50, funding=200
        ),
        active_constraints=constraints,
        budget_status=BudgetStatus(
            total=20, spent=0, remaining=20
        ),
    )
    return obs.model_dump()


def _clean(
    regions: dict[str, Any] | None = None,
    anomalies: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build a CleanState dict for payloads."""
    cs = CleanState(
        cleaned_observation=regions or {"north": {"infected": 50}},
        flagged_anomalies=anomalies,
    )
    return cs.model_dump()


def _belief(
    confidence: float = 0.7,
    trajectories: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    bs = BeliefState(
        hidden_var_estimates={
            "true_infected_multiplier": 2.0,
            "spread_rate": 1.1,
        },
        forecast_trajectories=trajectories or (
            {"label": "optimistic", "values": [40, 35, 30, 25, 20]},
            {"label": "baseline", "values": [50, 55, 60, 65, 70]},
            {"label": "pessimistic", "values": [60, 75, 90, 110, 130]},
        ),
        confidence=confidence,
    )
    return bs.model_dump()


def _memory_digest(
    num_entries: int = 0,
    keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    return MemoryDigest(
        num_entries=num_entries, keys=keys
    ).model_dump()


# ---------------------------------------------------------------------------
# Perception Tests (6)
# ---------------------------------------------------------------------------


class TestPerception:
    def test_returns_clean_state_schema(self) -> None:
        from cortex.roles.perception import PerceptionRole

        role = PerceptionRole()
        result = role.invoke(
            RoleInput(
                role_name="perception",
                payload={"observation": _obs()},
            )
        )
        assert isinstance(result, CleanState)
        assert isinstance(result.cleaned_observation, dict)
        assert isinstance(result.flagged_anomalies, tuple)

    def test_clamps_negative_values(self) -> None:
        from cortex.roles.perception import PerceptionRole

        role = PerceptionRole()
        # Build obs with a region whose infected is technically
        # impossible — we pass raw dict to bypass Pydantic validation
        obs_data = _obs(
            regions=(
                RegionState(
                    region_id="bad",
                    population=1000,
                    infected=0,
                    recovered=0,
                    deceased=0,
                ),
            )
        )
        # Manually inject a negative value into the raw data
        obs_data["regions"][0]["infected"] = -10
        result = role.invoke(
            RoleInput(
                role_name="perception",
                payload={"observation": obs_data},
            )
        )
        assert isinstance(result, CleanState)
        # Should flag IMPOSSIBLE
        assert any(
            "IMPOSSIBLE" in a for a in result.flagged_anomalies
        )

    def test_detects_spike_anomaly(self) -> None:
        from cortex.roles.perception import PerceptionRole

        role = PerceptionRole()
        prev = _clean(
            regions={"north": {"infected": 20, "population": 1000}}
        )
        obs_data = _obs(
            regions=(
                RegionState(
                    region_id="north",
                    population=1000,
                    infected=100,
                ),
            )
        )
        result = role.invoke(
            RoleInput(
                role_name="perception",
                payload={
                    "observation": obs_data,
                    "previous_clean": prev,
                },
            )
        )
        assert any("SPIKE" in a for a in result.flagged_anomalies)

    def test_detects_contradiction(self) -> None:
        from cortex.roles.perception import PerceptionRole

        role = PerceptionRole()
        obs_data = _obs(
            regions=(
                RegionState(
                    region_id="north",
                    population=10000,
                    infected=5,
                ),
            ),
            signals=(
                StakeholderSignal(
                    source="hospital",
                    urgency=0.9,
                    message="crisis",
                    turn=0,
                ),
            ),
        )
        result = role.invoke(
            RoleInput(
                role_name="perception",
                payload={"observation": obs_data},
            )
        )
        assert any(
            "CONTRADICTION" in a for a in result.flagged_anomalies
        )

    def test_handles_empty_regions(self) -> None:
        from cortex.roles.perception import PerceptionRole

        role = PerceptionRole()
        obs_data = _obs(regions=())
        result = role.invoke(
            RoleInput(
                role_name="perception",
                payload={"observation": obs_data},
            )
        )
        assert isinstance(result, CleanState)

    def test_filters_stale_reports(self) -> None:
        from cortex.roles.perception import PerceptionRole

        role = PerceptionRole()
        obs_data = _obs(
            turn=10,
            incidents=(
                IncidentReport(
                    region_id="north",
                    severity=0.5,
                    reported_turn=2,
                ),
                IncidentReport(
                    region_id="south",
                    severity=0.8,
                    reported_turn=9,
                ),
            ),
        )
        result = role.invoke(
            RoleInput(
                role_name="perception",
                payload={"observation": obs_data},
            )
        )
        cleaned = result.cleaned_observation
        incidents = cleaned.get("incidents", [])
        # Only the recent report (turn 9) should remain
        assert len(incidents) <= 1
        if incidents:
            assert incidents[0].get("reported_turn", 0) >= 7


# ---------------------------------------------------------------------------
# World Modeler Tests (5)
# ---------------------------------------------------------------------------


class TestWorldModeler:
    def test_returns_belief_state_schema(self) -> None:
        from cortex.roles.world_modeler import WorldModelerRole

        role = WorldModelerRole()
        result = role.invoke(
            RoleInput(
                role_name="world_modeler",
                payload={
                    "clean_state": _clean(),
                    "memory_digest": _memory_digest(),
                },
            )
        )
        assert isinstance(result, BeliefState)
        assert len(result.forecast_trajectories) == 3
        assert 0.1 <= result.confidence <= 1.0

    def test_uses_defaults_without_memory(self) -> None:
        from cortex.roles.world_modeler import WorldModelerRole

        role = WorldModelerRole()
        result = role.invoke(
            RoleInput(
                role_name="world_modeler",
                payload={
                    "clean_state": _clean(),
                    "memory_digest": _memory_digest(
                        num_entries=0
                    ),
                },
            )
        )
        assert (
            result.hidden_var_estimates.get("spread_rate", 0)
            == 1.1
        )

    def test_forecast_trajectory_structure(self) -> None:
        from cortex.roles.world_modeler import WorldModelerRole

        role = WorldModelerRole()
        result = role.invoke(
            RoleInput(
                role_name="world_modeler",
                payload={
                    "clean_state": _clean(),
                    "memory_digest": _memory_digest(),
                },
            )
        )
        for traj in result.forecast_trajectories:
            assert "values" in traj
            assert len(traj["values"]) == 5

    def test_confidence_decreases_with_anomalies(self) -> None:
        from cortex.roles.world_modeler import WorldModelerRole

        role = WorldModelerRole()
        r_clean = role.invoke(
            RoleInput(
                role_name="world_modeler",
                payload={
                    "clean_state": _clean(anomalies=()),
                    "memory_digest": _memory_digest(),
                },
            )
        )
        r_anom = role.invoke(
            RoleInput(
                role_name="world_modeler",
                payload={
                    "clean_state": _clean(
                        anomalies=(
                            "SPIKE:north",
                            "CONTRADICTION:src",
                            "GAP:detail",
                        )
                    ),
                    "memory_digest": _memory_digest(),
                },
            )
        )
        assert r_anom.confidence <= r_clean.confidence
        assert r_anom.confidence >= 0.1

    def test_zero_infection_produces_zero_forecast(self) -> None:
        from cortex.roles.world_modeler import WorldModelerRole

        role = WorldModelerRole()
        result = role.invoke(
            RoleInput(
                role_name="world_modeler",
                payload={
                    "clean_state": _clean(
                        regions={"north": {"infected": 0}}
                    ),
                    "memory_digest": _memory_digest(),
                },
            )
        )
        for traj in result.forecast_trajectories:
            assert all(v == 0 for v in traj["values"])


# ---------------------------------------------------------------------------
# Planner Tests (6)
# ---------------------------------------------------------------------------


class TestPlanner:
    def test_returns_plan_schema(self) -> None:
        from cortex.roles.planner import PlannerRole

        role = PlannerRole()
        result = role.invoke(
            RoleInput(
                role_name="planner",
                payload={
                    "belief_state": _belief(),
                    "goals": ["contain outbreak"],
                    "constraints": [],
                    "memory_digest": _memory_digest(),
                },
            )
        )
        assert isinstance(result, Plan)
        assert len(result.candidates) >= 1

    def test_respects_max_candidates(self) -> None:
        from cortex.roles.planner import PlannerRole

        role = PlannerRole()
        result = role.invoke(
            RoleInput(
                role_name="planner",
                payload={
                    "belief_state": _belief(),
                    "goals": ["contain outbreak"],
                    "constraints": [],
                    "memory_digest": _memory_digest(),
                    "max_candidates": 2,
                },
            )
        )
        assert len(result.candidates) <= 2

    def test_always_returns_at_least_one_candidate(self) -> None:
        from cortex.roles.planner import PlannerRole

        role = PlannerRole()
        # All actions constrained
        result = role.invoke(
            RoleInput(
                role_name="planner",
                payload={
                    "belief_state": _belief(),
                    "goals": [],
                    "constraints": [
                        {"name": "no_deploy", "active": True},
                        {"name": "no_restrict", "active": True},
                        {"name": "no_request", "active": True},
                        {"name": "no_escalate", "active": True},
                    ],
                    "memory_digest": _memory_digest(),
                },
            )
        )
        assert len(result.candidates) >= 1
        # Fallback should be NoOp
        assert result.candidates[-1].action.get("kind") == "noop"

    def test_excludes_constrained_actions(self) -> None:
        from cortex.roles.planner import PlannerRole

        role = PlannerRole()
        result = role.invoke(
            RoleInput(
                role_name="planner",
                payload={
                    "belief_state": _belief(),
                    "goals": ["contain"],
                    "constraints": [
                        {"name": "no_deploy", "active": True}
                    ],
                    "memory_digest": _memory_digest(),
                },
            )
        )
        for c in result.candidates:
            assert c.action.get("kind") != "deploy_resource"

    def test_ranks_candidates_by_confidence(self) -> None:
        from cortex.roles.planner import PlannerRole

        role = PlannerRole()
        result = role.invoke(
            RoleInput(
                role_name="planner",
                payload={
                    "belief_state": _belief(),
                    "goals": ["contain"],
                    "constraints": [],
                    "memory_digest": _memory_digest(),
                },
            )
        )
        confs = [c.confidence for c in result.candidates]
        assert confs == sorted(confs, reverse=True)

    def test_prioritizes_request_data_when_belief_empty(
        self,
    ) -> None:
        from cortex.roles.planner import PlannerRole

        role = PlannerRole()
        empty_belief = BeliefState(
            confidence=0.2,
            forecast_trajectories=(),
        ).model_dump()
        result = role.invoke(
            RoleInput(
                role_name="planner",
                payload={
                    "belief_state": empty_belief,
                    "goals": [],
                    "constraints": [],
                    "memory_digest": _memory_digest(),
                },
            )
        )
        top = result.candidates[0]
        assert top.action.get("kind") == "request_data"


# ---------------------------------------------------------------------------
# Critic Tests (5)
# ---------------------------------------------------------------------------


class TestCritic:
    def test_returns_critique_schema(self) -> None:
        from cortex.roles.critic import CriticRole

        role = CriticRole()
        candidate = CandidateAction(
            action={"kind": "deploy_resource", "amount": 50},
            rationale="deploy",
            expected_effect="reduce infected",
            confidence=0.7,
        ).model_dump()
        result = role.invoke(
            RoleInput(
                role_name="critic",
                payload={
                    "candidate": candidate,
                    "belief_state": _belief(),
                    "constraints": [],
                },
            )
        )
        assert isinstance(result, Critique)
        assert 0.0 <= result.risk_score <= 1.0

    def test_detects_resource_exhaustion(self) -> None:
        from cortex.roles.critic import CriticRole

        role = CriticRole()
        candidate = CandidateAction(
            action={
                "kind": "deploy_resource",
                "amount": 95,
                "total_available": 100,
            },
            rationale="deploy most resources",
            expected_effect="overwhelming response",
            confidence=0.6,
        ).model_dump()
        result = role.invoke(
            RoleInput(
                role_name="critic",
                payload={
                    "candidate": candidate,
                    "belief_state": _belief(),
                    "constraints": [],
                },
            )
        )
        assert any(
            "RESOURCE_EXHAUSTION" in fm
            for fm in result.failure_modes
        )

    def test_flags_low_confidence_irreversible(self) -> None:
        from cortex.roles.critic import CriticRole

        role = CriticRole()
        candidate = CandidateAction(
            action={
                "kind": "restrict_movement",
                "level": 3,
            },
            rationale="max restriction",
            expected_effect="stop spread",
            confidence=0.3,
        ).model_dump()
        result = role.invoke(
            RoleInput(
                role_name="critic",
                payload={
                    "candidate": candidate,
                    "belief_state": _belief(confidence=0.3),
                    "constraints": [],
                },
            )
        )
        assert any(
            "LOW_CONFIDENCE_IRREVERSIBLE" in fm
            for fm in result.failure_modes
        )

    def test_risk_score_increases_with_failures(self) -> None:
        from cortex.roles.critic import CriticRole

        role = CriticRole()
        # Clean candidate
        clean = CandidateAction(
            action={"kind": "noop"},
            rationale="wait",
            expected_effect="none",
            confidence=0.9,
        ).model_dump()
        r_clean = role.invoke(
            RoleInput(
                role_name="critic",
                payload={
                    "candidate": clean,
                    "belief_state": _belief(confidence=0.9),
                    "constraints": [],
                },
            )
        )
        # Risky candidate
        risky = CandidateAction(
            action={
                "kind": "deploy_resource",
                "amount": 95,
                "total_available": 100,
            },
            rationale="risky deploy",
            expected_effect="maybe",
            confidence=0.3,
        ).model_dump()
        r_risky = role.invoke(
            RoleInput(
                role_name="critic",
                payload={
                    "candidate": risky,
                    "belief_state": _belief(confidence=0.3),
                    "constraints": [],
                },
            )
        )
        assert r_risky.risk_score > r_clean.risk_score

    def test_clean_pass_returns_low_risk(self) -> None:
        from cortex.roles.critic import CriticRole

        role = CriticRole()
        candidate = CandidateAction(
            action={
                "kind": "deploy_resource",
                "amount": 10,
                "total_available": 100,
            },
            rationale="small deploy",
            expected_effect="help",
            confidence=0.9,
        ).model_dump()
        result = role.invoke(
            RoleInput(
                role_name="critic",
                payload={
                    "candidate": candidate,
                    "belief_state": _belief(confidence=0.9),
                    "constraints": [],
                },
            )
        )
        assert result.risk_score <= 0.2


# ---------------------------------------------------------------------------
# Executive Tests (7)
# ---------------------------------------------------------------------------


class TestExecutive:
    def test_returns_executive_decision_schema(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        plan = Plan(
            candidates=(
                CandidateAction(
                    action={"kind": "noop"},
                    rationale="wait",
                    expected_effect="none",
                    confidence=0.5,
                ),
            )
        )
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [plan.model_dump()],
                    "budget_status": BudgetStatus(
                        total=20, spent=5, remaining=15
                    ).model_dump(),
                },
            )
        )
        assert isinstance(result, ExecutiveDecision)
        assert result.decision in (
            "act",
            "call",
            "wait",
            "escalate",
            "stop",
        )

    def test_acts_when_budget_zero(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        plan = Plan(
            candidates=(
                CandidateAction(
                    action={"kind": "deploy_resource"},
                    rationale="deploy",
                    expected_effect="help",
                    confidence=0.8,
                ),
            )
        )
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [plan.model_dump()],
                    "budget_status": BudgetStatus(
                        total=20, spent=20, remaining=0
                    ).model_dump(),
                },
            )
        )
        assert result.decision == "act"

    def test_returns_noop_when_budget_zero_no_plan(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [],
                    "budget_status": BudgetStatus(
                        total=20, spent=20, remaining=0
                    ).model_dump(),
                },
            )
        )
        assert result.decision == "act"
        assert result.target_action is not None
        assert result.target_action.get("kind") == "noop"

    def test_calls_world_modeler_when_anomalies(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        clean = CleanState(
            flagged_anomalies=("SPIKE:north", "GAP:detail"),
        )
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [clean.model_dump()],
                    "budget_status": BudgetStatus(
                        total=20, spent=2, remaining=18
                    ).model_dump(),
                },
            )
        )
        assert result.decision == "call"
        assert result.target_role == "world_modeler"

    def test_calls_planner_when_no_plan(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        clean = CleanState()
        belief = BeliefState(confidence=0.7)
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [
                        clean.model_dump(),
                        belief.model_dump(),
                    ],
                    "budget_status": BudgetStatus(
                        total=20, spent=4, remaining=16
                    ).model_dump(),
                },
            )
        )
        assert result.decision == "call"
        assert result.target_role == "planner"

    def test_escalates_on_high_risk(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        clean = CleanState()
        belief = BeliefState(confidence=0.4)
        plan = Plan(
            candidates=(
                CandidateAction(
                    action={"kind": "deploy_resource"},
                    rationale="deploy",
                    expected_effect="help",
                    confidence=0.4,
                ),
            )
        )
        critique = Critique(
            failure_modes=("RESOURCE_EXHAUSTION", "CASCADE"),
            risk_score=0.8,
        )
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [
                        clean.model_dump(),
                        belief.model_dump(),
                        plan.model_dump(),
                        critique.model_dump(),
                    ],
                    "budget_status": BudgetStatus(
                        total=20, spent=8, remaining=12
                    ).model_dump(),
                },
            )
        )
        assert result.decision == "escalate"

    def test_acts_on_low_risk_critique(self) -> None:
        from cortex.roles.executive import ExecutiveRole

        role = ExecutiveRole()
        clean = CleanState()
        belief = BeliefState(confidence=0.9)
        plan = Plan(
            candidates=(
                CandidateAction(
                    action={"kind": "deploy_resource"},
                    rationale="safe deploy",
                    expected_effect="help",
                    confidence=0.9,
                ),
            )
        )
        critique = Critique(risk_score=0.2)
        result = role.invoke(
            RoleInput(
                role_name="executive",
                payload={
                    "artifacts": [
                        clean.model_dump(),
                        belief.model_dump(),
                        plan.model_dump(),
                        critique.model_dump(),
                    ],
                    "budget_status": BudgetStatus(
                        total=20, spent=8, remaining=12
                    ).model_dump(),
                },
            )
        )
        assert result.decision == "act"
