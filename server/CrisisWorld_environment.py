"""CrisisWorld — OpenEnv Environment for outbreak control."""

from __future__ import annotations

import logging
import math
from typing import Any

_log = logging.getLogger(__name__)

import numpy as np
from openenv.core.env_server.interfaces import Environment

from models import (
    ActionUnion,
    CrisisState,
    EnvConfig,
    EnvironmentMetadata,
    NoOp,
    Observation,
    OuterAction,
    RewardWeights,
)
from models import BudgetStatusSnapshot
from server._internal import InternalState
from server.actions import validate_and_schedule
from server.constraints import check_constraints
from server.dynamics import advance_epi_state
from server.observations import assemble_observation
from server.regions import build_adjacency, init_regions, seed_infection
from server.resources import apply_resource_change, apply_turn_decay
from server.rewards import compute_reward
from server.scenarios import generate_scenario
from server.stakeholders import generate_signals
from server.termination import check_termination




class CrisisWorld(Environment[ActionUnion, Observation, CrisisState]):
    """Outbreak-control environment implementing OpenEnv Environment ABC."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, config: EnvConfig | None = None) -> None:
        self._config = config or EnvConfig()
        self._state: InternalState | None = None
        self._rng: np.random.Generator | None = None
        self._done = False
        self._closed = False
        self._initial_resources = None
        self._budget = BudgetStatusSnapshot(total=50, spent=0, remaining=50)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Initialize a new episode."""
        if self._closed:
            raise RuntimeError("Environment is closed")

        if seed is None:
            _log.warning("No seed provided, defaulting to 0")
        actual_seed = seed if seed is not None else 0
        self._rng = np.random.default_rng(actual_seed)

        scenario = generate_scenario(self._config, actual_seed)
        cols = max(1, int(math.ceil(math.sqrt(self._config.num_regions))))
        regions = init_regions(self._config, self._rng)
        regions = seed_infection(
            regions, scenario.origin_region, scenario.initial_infected
        )
        ids = [r.region_id for r in regions]
        adjacency = build_adjacency(ids, cols)

        self._state = InternalState(
            episode_id=episode_id or f"ep-{actual_seed}",
            turn=0,
            regions=regions,
            adjacency=adjacency,
            resources=scenario.initial_resources,
            constraints=scenario.initial_constraints,
            pending_effects=(),
            action_history=[],
            epi_params=scenario.epi_params,
            scenario=scenario,
        )
        self._initial_resources = scenario.initial_resources
        self._done = False
        self._budget = BudgetStatusSnapshot(total=50, spent=0, remaining=50)

        signals = generate_signals(regions, 0, self._rng)
        return assemble_observation(
            regions=regions,
            resources=self._state.resources,
            constraints=self._state.constraints,
            signals=signals,
            budget_status=self._budget,
            turn=0,
            rng=self._rng,
            done=False,
            reward=None,
            metadata={
                "episode_id": self._state.episode_id,
                "step_count": 0,
            },
            noise_scale=self._state.epi_params.noise_scale,
        )

    def step(
        self,
        action: ActionUnion,
        timeout_s: float | None = None,
        **kwargs: Any,
    ) -> Observation:
        """Advance one turn."""
        if self._state is None:
            raise RuntimeError("Must call reset() before step()")
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._done:
            raise RuntimeError("Episode is done")

        st = self._state
        prev_regions = st.regions

        # 1. Increment turn
        st.turn += 1

        # 2. Validate action + check constraints
        violations = check_constraints(action, st.constraints, st.regions)
        errors, effects = validate_and_schedule(action, st, self._rng)

        # 3. Append effects to pending
        st.pending_effects = st.pending_effects + effects

        # 4. Apply due effects
        due = tuple(e for e in st.pending_effects if e.apply_on_turn <= st.turn)
        remaining = tuple(e for e in st.pending_effects if e.apply_on_turn > st.turn)
        st.pending_effects = remaining
        for effect in due:
            self._apply_effect(effect, st)

        # 5. Advance epidemiological dynamics
        st.regions = advance_epi_state(
            st.regions, st.adjacency, st.epi_params, self._rng
        )

        # 6. Apply per-turn resource decay
        st.resources = apply_turn_decay(st.resources)

        # 7. Check termination
        done, reason = check_termination(
            st.regions, st.resources, self._initial_resources,
            st.turn, st.scenario.max_turns,
        )
        self._done = done

        # 8. Compute reward
        reward = compute_reward(
            prev_regions, st.regions, action, violations,
            self._config.reward_weights, st.turn, st.scenario.max_turns,
            termination_reason=reason if done else "",
        )

        # 9. Record action
        st.action_history.append(action.model_dump())

        # 10. Generate signals and assemble observation
        signals = generate_signals(st.regions, st.turn, self._rng)
        return assemble_observation(
            regions=st.regions,
            resources=st.resources,
            constraints=st.constraints,
            signals=signals,
            budget_status=self._budget,
            turn=st.turn,
            rng=self._rng,
            done=done,
            reward=reward.total,
            metadata={
                "episode_id": st.episode_id,
                "step_count": st.turn,
                "termination_reason": reason if done else "",
                "violations": violations,
            },
            prev_regions=prev_regions,
            noise_scale=st.epi_params.noise_scale,
        )

    @property
    def state(self) -> CrisisState:
        """Read-only snapshot of current state."""
        if self._state is None:
            return CrisisState()
        return CrisisState(
            episode_id=self._state.episode_id,
            step_count=self._state.turn,
            regions=self._state.regions,
            resources=self._state.resources,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CrisisWorld",
            description="Outbreak-control environment for budgeted deliberation",
            version="0.1.0",
        )

    def close(self) -> None:
        self._closed = True

    def _apply_effect(
        self, effect: Any, st: InternalState
    ) -> None:
        """Apply a scheduled effect, mutating *st* in-place."""
        if effect.effect_type == "deploy_resource":
            payload = effect.payload
            st.resources = apply_resource_change(
                st.resources,
                medical=payload.get("amount", 0) if payload.get("resource") == "medical" else 0,
                personnel=payload.get("amount", 0) if payload.get("resource") == "personnel" else 0,
                funding=payload.get("amount", 0) if payload.get("resource") == "funding" else 0,
            )
        elif effect.effect_type == "restrict_movement":
            target = effect.target_region
            st.regions = tuple(
                r.model_copy(update={"restricted": True})
                if r.region_id == target
                else r
                for r in st.regions
            )
        elif effect.effect_type == "reallocate_budget":
            payload = effect.payload
            from_cat = payload.get("from_category", "")
            to_cat = payload.get("to_category", "")
            amount = payload.get("amount", 0)
            st.resources = apply_resource_change(
                st.resources,
                **{from_cat: -amount, to_cat: amount},
            )
        # escalate, request_data, public_communication: no state changes
