"""Tests for models.py — env-contract types extending OpenEnv bases."""

from __future__ import annotations

import pytest


# --- RegionState ---


class TestRegionState:
    def test_basic_construction(self) -> None:
        from models import RegionState

        r = RegionState(region_id="north", population=1000)
        assert r.region_id == "north"
        assert r.population == 1000
        assert r.infected == 0

    def test_population_invariant_violated(self) -> None:
        from models import RegionState

        with pytest.raises(ValueError, match="exceeds population"):
            RegionState(
                region_id="x", population=100, infected=60, recovered=50
            )

    def test_frozen(self) -> None:
        from models import RegionState

        r = RegionState(region_id="a", population=100)
        with pytest.raises(Exception):
            r.infected = 5  # type: ignore[misc]


# --- ResourcePool ---


class TestResourcePool:
    def test_defaults(self) -> None:
        from models import ResourcePool

        rp = ResourcePool()
        assert rp.medical == 0
        assert rp.personnel == 0
        assert rp.funding == 0

    def test_rejects_negative(self) -> None:
        from models import ResourcePool

        with pytest.raises(Exception):
            ResourcePool(medical=-1)


# --- Constraint ---


class TestConstraint:
    def test_construction(self) -> None:
        from models import Constraint

        c = Constraint(name="travel_ban", description="No flights", active=True)
        assert c.name == "travel_ban"
        assert c.active is True


# --- Reward ---


class TestRewardComponents:
    def test_defaults(self) -> None:
        from models import RewardComponents

        rc = RewardComponents()
        assert rc.outcome == 0.0
        assert rc.inner_compute_cost == 0.0


class TestCompositeReward:
    def test_valid_weights(self) -> None:
        from models import CompositeReward, RewardComponents

        cr = CompositeReward(
            total=1.0,
            components=RewardComponents(),
            weights={"outcome": 1.0},
        )
        assert cr.total == 1.0

    def test_invalid_weight_key(self) -> None:
        from models import CompositeReward, RewardComponents

        with pytest.raises(ValueError, match="Unknown weight keys"):
            CompositeReward(
                total=0.0,
                components=RewardComponents(),
                weights={"bogus": 1.0},
            )


# --- Config ---


class TestRewardWeights:
    def test_defaults(self) -> None:
        from models import RewardWeights

        rw = RewardWeights()
        assert rw.outcome == 1.0
        assert rw.timeliness == 0.5

    def test_rejects_negative(self) -> None:
        from models import RewardWeights

        with pytest.raises(Exception):
            RewardWeights(outcome=-0.1)


class TestEnvConfig:
    def test_defaults(self) -> None:
        from models import EnvConfig

        ec = EnvConfig()
        assert ec.num_regions == 4
        assert ec.max_turns == 50

    def test_rejects_zero_regions(self) -> None:
        from models import EnvConfig

        with pytest.raises(Exception):
            EnvConfig(num_regions=0)


# --- Observation components ---


class TestStakeholderSignal:
    def test_construction(self) -> None:
        from models import StakeholderSignal

        s = StakeholderSignal(source="WHO", urgency=0.8, message="alert", turn=1)
        assert s.source == "WHO"
        assert s.urgency == 0.8


class TestIncidentReport:
    def test_construction(self) -> None:
        from models import IncidentReport

        ir = IncidentReport(region_id="east", severity=0.5, reported_turn=3)
        assert ir.region_id == "east"


class TestTelemetry:
    def test_defaults(self) -> None:
        from models import Telemetry

        t = Telemetry()
        assert t.total_infected == 0
        assert t.data_staleness == 0


# --- Observation (OpenEnv extension) ---


class TestObservation:
    @staticmethod
    def _resolve_observation() -> type:
        from models import Observation
        from schemas.budget import BudgetStatus

        Observation.model_rebuild(
            _types_namespace={"BudgetStatus": BudgetStatus}
        )
        return Observation

    def _make_obs(self, **overrides: object) -> object:
        from models import RegionState, ResourcePool, Telemetry
        from schemas.budget import BudgetStatus

        Observation = self._resolve_observation()
        defaults: dict = dict(
            turn=0,
            regions=(RegionState(region_id="a", population=100),),
            telemetry=Telemetry(),
            resources=ResourcePool(),
            budget_status=BudgetStatus(total=20, spent=0, remaining=20),
        )
        defaults.update(overrides)
        return Observation(**defaults)

    def test_basic_construction(self) -> None:
        obs = self._make_obs()
        assert obs.turn == 0  # type: ignore[attr-defined]

    def test_done_defaults_false(self) -> None:
        obs = self._make_obs()
        assert obs.done is False  # type: ignore[attr-defined]

    def test_reward_defaults_none(self) -> None:
        obs = self._make_obs()
        assert obs.reward is None  # type: ignore[attr-defined]

    def test_metadata_defaults_empty(self) -> None:
        obs = self._make_obs()
        assert obs.metadata == {}  # type: ignore[attr-defined]

    def test_done_can_be_set(self) -> None:
        obs = self._make_obs(done=True, reward=5.0)
        assert obs.done is True  # type: ignore[attr-defined]
        assert obs.reward == 5.0  # type: ignore[attr-defined]

    def test_metadata_can_be_set(self) -> None:
        obs = self._make_obs(metadata={"key": "val"})
        assert obs.metadata == {"key": "val"}  # type: ignore[attr-defined]


# --- Actions (OpenEnv extension) ---


class TestActions:
    def test_noop(self) -> None:
        from models import NoOp

        a = NoOp()
        assert a.kind == "noop"

    def test_deploy_resource(self) -> None:
        from models import DeployResource

        a = DeployResource(resource="medical", region_id="north", amount=10)
        assert a.kind == "deploy_resource"

    def test_action_metadata_defaults_empty(self) -> None:
        from models import NoOp

        a = NoOp()
        assert a.metadata == {}

    def test_action_metadata_can_be_set(self) -> None:
        from models import NoOp

        a = NoOp(metadata={"trace": True})
        assert a.metadata == {"trace": True}

    def test_reallocate_budget_categories_differ(self) -> None:
        from models import ReallocateBudget

        with pytest.raises(ValueError, match="must differ"):
            ReallocateBudget(
                from_category="medical", to_category="medical", amount=5
            )

    def test_action_union_discriminator(self) -> None:
        from pydantic import TypeAdapter

        from models import ActionUnion, NoOp

        ta = TypeAdapter(ActionUnion)
        result = ta.validate_python({"kind": "noop"})
        assert isinstance(result, NoOp)


# --- CrisisState (OpenEnv extension, NOT frozen) ---


class TestCrisisState:
    def test_basic_construction(self) -> None:
        from models import CrisisState

        cs = CrisisState()
        assert cs.episode_id is None
        assert cs.step_count == 0

    def test_episode_id_and_step_count(self) -> None:
        from models import CrisisState

        cs = CrisisState(episode_id="ep-001", step_count=5)
        assert cs.episode_id == "ep-001"
        assert cs.step_count == 5

    def test_step_count_mutable(self) -> None:
        from models import CrisisState

        cs = CrisisState(step_count=0)
        cs.step_count = 1
        assert cs.step_count == 1

    def test_extra_fields_allowed(self) -> None:
        from models import CrisisState, RegionState, ResourcePool

        cs = CrisisState(
            regions=(RegionState(region_id="a", population=100),),
            resources=ResourcePool(medical=10),
        )
        assert len(cs.regions) == 1


# --- EnvironmentMetadata ---


class TestEnvironmentMetadata:
    def test_construction(self) -> None:
        from models import EnvironmentMetadata

        em = EnvironmentMetadata(
            name="CrisisWorld",
            description="Outbreak env",
            version="0.1.0",
        )
        assert em.name == "CrisisWorld"
        assert em.version == "0.1.0"

    def test_version_is_optional(self) -> None:
        from models import EnvironmentMetadata

        em = EnvironmentMetadata(name="Test", description="desc")
        assert em.version is None

    def test_optional_fields_default_none(self) -> None:
        from models import EnvironmentMetadata

        em = EnvironmentMetadata(name="Test", description="desc")
        assert em.readme_content is None
        assert em.author is None
        assert em.documentation_url is None

    def test_full_construction(self) -> None:
        from models import EnvironmentMetadata

        em = EnvironmentMetadata(
            name="CrisisWorld",
            description="Outbreak env",
            version="0.1.0",
            readme_content="# Readme",
            author="Meta",
            documentation_url="https://example.com",
        )
        assert em.author == "Meta"
        assert em.documentation_url == "https://example.com"


# --- OpenEnv Inheritance ---


class TestOpenEnvInheritance:
    def test_observation_inherits_openenv_base(self) -> None:
        from openenv.core.env_server.types import Observation as BaseObservation

        from models import Observation

        assert issubclass(Observation, BaseObservation)

    def test_crisis_state_inherits_openenv_state(self) -> None:
        from openenv.core.env_server.types import State as BaseState

        from models import CrisisState

        assert issubclass(CrisisState, BaseState)

    def test_outer_action_inherits_openenv_action(self) -> None:
        from openenv.core.env_server.types import Action as BaseAction

        from models import OuterAction

        assert issubclass(OuterAction, BaseAction)
