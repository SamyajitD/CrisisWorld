# src/env/ -- CrisisWorld (Outer Environment)

## Behavior

This package implements `EnvProtocol` as **CrisisWorld**, a stateful outbreak-control
simulator. It maintains hidden epidemiological state, resources, stakeholders, policy
constraints, and delayed action effects. It emits partial, noisy, lagged observations
and computes composite reward.

**Hard rules:**
- Never imported directly by `agents/` or `cortex/`. Only `main.py` wires it.
- All state transitions return new state objects -- no mutation in place.
- All stochastic operations require an explicit `rng: numpy.random.Generator` parameter.
- Same seed produces an identical trajectory.
- Functions < 50 lines. Files < 400 lines (800 hard max). No `print()`.

---

## Exposed APIs

### world.py -- CrisisWorld

```python
class CrisisWorld:
    def __init__(self, config: EnvConfig) -> None: ...
    def reset(self, seed: int) -> Observation: ...
    def step(self, action: OuterAction) -> StepResult: ...
    def close(self) -> None: ...
```

### dynamics.py -- `advance_epi_state(regions, adjacency, params, rng) -> tuple[RegionState, ...]`
### regions.py -- `init_regions(config, rng)`, `build_adjacency(ids, cols)`, `seed_infection(regions, origin, count)`
### resources.py -- `apply_resource_change(pool, delta)`, `check_sufficient(pool, cost)`, `apply_turn_decay(pool)`
### stakeholders.py -- `generate_signals(regions, turn, rng) -> tuple[StakeholderSignal, ...]`
### constraints.py -- `check_constraints(action, active) -> list[str]`
### actions.py -- `validate_and_schedule(action, state, rng) -> tuple[list[str], tuple[ScheduledEffect, ...]]`
### observations.py -- `assemble_observation(state, rng, turn) -> Observation`
### rewards.py -- `compute_reward(prev, curr, action, violations, weights, turn, max_turns) -> CompositeReward`
### scenarios.py -- `generate_scenario(config, seed) -> ScenarioParams`
### termination.py -- `check_termination(state, turn, max_turns) -> tuple[bool, str]`

---

## Internal Data Structures (not exported)

```
InternalState:    turn, regions, adjacency, resources, constraints, pending_effects, action_history, epi_params, scenario
ScheduledEffect:  apply_on_turn, effect_type, target_region, payload
EpiParams:        beta (0.15-0.45), gamma (0.05-0.15), mu (0.005-0.03), inter_region_spread (0.01-0.10), noise_scale (0.01-0.05)
ScenarioParams:   origin_region, initial_infected, epi_params, initial_resources, initial_constraints, max_turns
```

---

## Module Implementation Plans

### dynamics.py -- SIR Model (discrete-time stochastic)

**Intra-region (per region):**
```
S = population - infected - recovered - deceased
new_infections = binomial(S, 1 - exp(-beta * infected / population)) + noise
new_recoveries = binomial(infected, gamma)
new_deaths = binomial(infected, mu)
```
If restricted: effective beta *= 0.5. Clamp all to non-negative. Ensure I+R+D <= population.

**Inter-region spread:** For adjacent pairs, `spillover = binomial(r.infected, inter_region_spread)`. If restricted, outgoing spillover *= 0.1. Cap at susceptible count.

**Edge cases:** zero population -> unchanged. negative from noise -> clamp to 0. all regions zero infected -> no-op. spillover to zero susceptible -> discard.

### regions.py -- Grid Management

Default 3x3 grid (configurable). 4-connected adjacency (up/down/left/right).
Population per region: `rng.integers(50_000, 200_000)`. Origin gets initial infected.

**Edge cases:** 1x1 grid -> empty adjacency. Origin not found -> ValueError. Initial infected > population -> clamp.

### resources.py -- Pool Operations

`apply_resource_change`: `max(0, pool.field + delta.field)` per field. Never negative.
`apply_turn_decay`: medical -2%, funding -1%, personnel unchanged.
**Edge cases:** negative delta beyond pool -> clamp to 0. All zero -> stays zero.

### stakeholders.py -- Signal Generation

4 sources (hospital, media, government, public). Urgency correlates with state, plus Gaussian noise.
Lag: hospital=0, media=0-1, government=1, public=1-2. Clamp urgency to [0, 1].
**Edge cases:** turn 0 -> use current state for lagged signals. No infected -> near-zero urgency.

### constraints.py -- Policy Rules

5 default constraints: no_restrict_low_infection, resource_equity, escalation_requires_threshold, max_restriction_duration, budget_floor. `check_constraints` returns violation descriptions.
**Edge cases:** multiple violations -> return all. NoOp -> never violates. Unknown region -> skip.

### actions.py -- Validation + Delayed Effects

| Action | Delay | Key validation |
|--------|-------|---------------|
| DeployResource | 2 turns | Region exists, sufficient resources, amount > 0 |
| RestrictMovement | 1 turn | Region exists, level in {1,2,3} |
| RequestData | 1 turn | Valid source |
| PublicCommunication | 0 turns | Non-empty message |
| Escalate | 2 turns | Valid agency |
| ReallocateBudget | 0 turns | Valid categories, amount available |
| NoOp | 0 turns | Always valid |

**Edge cases:** nonexistent region -> validation failure. Duplicate restriction -> update level.

### observations.py -- Partial Observability

| Data | Visibility |
|------|-----------|
| Exact infected/recovered/deceased | HIDDEN (noisy: observed = true + N(0, 0.2*true)) |
| Population, resources, constraints | REVEALED |
| Telemetry aggregates | REVEALED but lagged 1 turn |
| True epi parameters | HIDDEN |

**Edge cases:** turn 0 -> telemetry zeros. Noise makes negative -> clamp to 0. Noise exceeds population -> clamp.

### rewards.py -- `R = α·outcome + β·timeliness - γ·compute - δ·violations + ε·comms`

```
R_outcome = 1.0 - 2.0 * mortality_rate - infection_rate  (clamp [-1, 1])
R_timeliness = -delta_infected/total_pop if growing, else 0
C_compute = 0.0 (set by evaluation layer)
R_violations = len(violations) * 0.1
R_comms = 0.2 if PublicCommunication + no violations, 0.05 if with violations, else 0.0
Terminal: contained +2.0, catastrophic -3.0, timed out -1.0
```

**Edge cases:** zero population -> R_outcome=0.0. All recovered -> R_outcome=1.0.

### scenarios.py -- Deterministic generation from seed

Independent RNG from seed. Samples: origin region, initial_infected (10-500), all epi params, initial resources.

### termination.py -- End Conditions (checked in order)

1. **Contained**: all regions < 1% infected AND turn >= 5 -> "contained"
2. **Catastrophic mortality**: deceased > 5% total pop -> "catastrophic_mortality"
3. **Catastrophic spread**: infected > 60% total pop -> "catastrophic_spread"
4. **Resource collapse**: all resources < 5% initial -> "resource_collapse"
5. **Max turns**: turn >= max_turns -> "max_turns"

First match wins. **Edge cases:** turn 0 not contained (min turn 5). Exactly 1% -> NOT contained.

---

## Step Pipeline (world.py)

```
1. Guard checks (initialized, not closed, not done)
2. Increment turn
3. Validate action + check constraints -> violations, effects
4. Append effects to pending
5. Apply due effects (apply_on_turn == current_turn)
6. Advance epidemiological dynamics
7. Apply per-turn resource decay
8. Update constraint activation
9. Compute reward(prev_state, new_state, action, violations, weights)
10. Check termination
11. Assemble observation
12. Return StepResult
```

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `EnvProtocol` | `src/protocols/env.py` | CrisisWorld implements this |
| `Observation`, `StakeholderSignal`, `Telemetry`, `IncidentReport` | `src/schemas/observation.py` | Observation assembly |
| `OuterAction` + variants | `src/schemas/action.py` | Action validation |
| `CompositeReward`, `RewardComponents` | `src/schemas/reward.py` | Reward computation |
| `RegionState`, `ResourcePool`, `Constraint`, `StepResult` | `src/schemas/state.py` | State management |
| `EnvConfig` | `src/schemas/config.py` | Configuration |
| `numpy` | Third-party | RNG and stochastic operations |

---

## Test Plan

### tests/unit/test_dynamics.py

```
test_single_region_sir_basic                       -- known params, bounds check, S+I+R+D==pop
test_zero_infected_no_change                       -- all counts unchanged
test_restriction_halves_beta                       -- restricted produces fewer infections
test_inter_region_spread                           -- adjacent region gains infections
test_restriction_blocks_spillover                  -- restricted outgoing *= 0.1
test_determinism_same_seed                         -- two runs identical
test_negative_clamp                                -- pathological noise -> all counts >= 0
test_zero_population_region_unchanged              -- pop=0 -> unchanged
```

### tests/unit/test_regions.py

```
test_init_regions_count                            -- 3x3 -> 9 regions
test_init_regions_all_healthy                      -- all infected=0
test_build_adjacency_corners                       -- r0 has 2 neighbors
test_build_adjacency_edge                          -- r1 has 3 neighbors
test_build_adjacency_center                        -- r4 has 4 neighbors
test_seed_infection                                -- origin gets infected, others 0
test_seed_infection_clamp                           -- count > pop -> clamped
test_single_region_grid                            -- 1x1 -> 1 region, empty adjacency
test_determinism_same_seed                         -- identical populations
```

### tests/unit/test_resources.py

```
test_apply_positive_delta                          -- correct addition
test_negative_clamp                                -- excess deduction -> 0
test_check_sufficient_passes                       -- empty violation list
test_check_sufficient_fails                        -- "medical" in violations
test_turn_decay                                    -- medical ~98, funding ~99, personnel 100
test_zero_pool_stays_zero                          -- 0 + negative -> 0
```

### tests/unit/test_rewards.py

```
test_perfect_outcome                               -- 0 infected/deceased -> R_outcome=1.0
test_high_mortality_penalty                        -- 10% mortality -> R_outcome < 0
test_timeliness_growing_infection                  -- delta > 0 -> negative
test_timeliness_shrinking_infection                -- delta < 0 -> 0.0
test_violations_stacking                           -- 3 violations -> 0.3
test_comms_reward_clean                            -- PublicComm + 0 violations -> 0.2
test_comms_reward_with_violations                  -- PublicComm + violations -> 0.05
test_composite_weighted_sum                        -- known weights, verify formula
test_zero_population_no_crash                      -- R_outcome=0.0, no division by zero
test_terminal_bonus_contained                      -- +2.0 bonus
```

### tests/unit/test_termination.py

```
test_contained                                     -- all < 1%, turn >= 5 -> (True, "contained")
test_not_contained_below_min_turn                  -- turn=2 -> (False, "")
test_catastrophic_mortality                        -- deceased > 5% -> (True, "catastrophic_mortality")
test_catastrophic_spread                           -- infected > 60% -> (True, "catastrophic_spread")
test_resource_collapse                             -- all < 5% initial -> (True, "resource_collapse")
test_max_turns                                     -- turn == max -> (True, "max_turns")
test_priority_order                                -- contained AND max_turns -> "contained"
```

### tests/integration/test_env_step.py

```
test_reset_returns_valid_observation               -- turn=0, non-empty regions, valid resources
test_step_returns_step_result                      -- obs, reward, done fields present
test_noop_sequence_progresses                      -- 10 NoOps, turn increments, infection changes
test_step_before_reset_raises                      -- RuntimeError
test_step_after_close_raises                       -- RuntimeError
test_step_after_done_raises                        -- RuntimeError
test_full_episode_determinism                      -- seed=42 twice -> identical reward sequences
test_deploy_resource_delayed                       -- resources arrive on turn+2
test_constraint_violation_penalized                -- violation in info, lower reward
test_different_seeds_different_trajectories         -- seed=42 vs 99 -> different obs
test_observation_noise                             -- 10 resets, variance > 0
test_containment_terminates                        -- low beta -> "contained" or "max_turns"
```

---

## Implementation Order

1. `_internal.py` (InternalState, ScheduledEffect, EpiParams, ScenarioParams)
2. `regions.py` + test_regions.py
3. `resources.py` + test_resources.py
4. `dynamics.py` + test_dynamics.py
5. `scenarios.py`
6. `constraints.py`
7. `actions.py`
8. `stakeholders.py`
9. `observations.py`
10. `rewards.py` + test_rewards.py
11. `termination.py` + test_termination.py
12. `world.py` + test_env_step.py
