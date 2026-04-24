# src/agents/ -- Agent Implementations

## Behavior

Two agents satisfying `AgentProtocol`:

- **FlatAgent** -- Priority-ordered heuristic policy. No deliberation, no Cortex roles,
  no budget consumption. Used in Flat-lite and Flat-fat ablation conditions.
- **CortexAgent** -- Delegates to `CortexDeliberator` via constructor injection
  (protocol-typed, never concrete class). Used in Cortex-lite and Cortex-full.

Both expose identical interfaces. The experiment runner swaps them transparently.

**Hard rules:**
- Never imports from `src/env/` directly.
- Never imports concrete classes from `src/cortex/`. Protocol types only.
- All wiring in `main.py`. All stochastic decisions use injected RNG. No `print()`.

---

## Exposed APIs

### flat.py -- FlatAgent

```python
class FlatAgent:
    def __init__(self, config: EnvConfig, rng: Generator) -> None: ...
    def act(self, observation: Observation) -> OuterAction: ...
    def reset(self) -> None: ...
```

### cortex_agent.py -- CortexAgent

```python
class CortexAgent:
    def __init__(self, deliberator, budget: BudgetProtocol, logger: LoggerProtocol) -> None: ...
    def act(self, observation: Observation) -> OuterAction: ...
    def reset(self) -> None: ...
```

---

## Implementation Plan

### FlatAgent -- Heuristic Policy

**Per-episode state:** `_turn_count`, `_action_history`, `_last_action_kind`, `_escalated`, `_data_requested_this_episode`

**Priority cascade (first matching rule fires):**

1. **Critical infection surge**: any region infected/population > 0.15 AND medical > 0
   -> `DeployResource(region=worst, amount=min(medical//2, needed))`

2. **Resource depletion**: medical == 0 OR personnel == 0
   -> `ReallocateBudget(from="funding", to=depleted, amount=funding//3)`

3. **High-urgency signal**: any signal urgency >= 0.8 AND not `_escalated`
   -> `Escalate(agency=signal.source)`. Set `_escalated = True`.

4. **Movement restriction**: infection rate > 0.08 AND not restricted AND last != restrict
   -> `RestrictMovement(region=target, level=1 if rate<0.12 else 2)`

5. **Information gap**: turn > 0 AND telemetry/region mismatch > 10% AND source not queried
   -> `RequestData(source="telemetry")`

6. **Public communication**: turn % 5 == 0 AND turn > 0 AND any region rate > 0.05
   -> `PublicCommunication(audience="public", message="situation_update")`

7. **Proactive deployment**: any region rate > 0.03 AND medical > 0 AND last != deploy
   -> `DeployResource(region=highest_rate, amount=medical//4)`

8. **Fallback**: `NoOp(kind="noop")`

**Helpers:** `_worst_region(regions)`, `_compute_infection_rate(region)` (safe: pop==0 -> 0.0), `_pick_deploy_amount(available, needed)`

**Flat-lite vs Flat-fat:**
Same class, differentiated by `config.fat_mode: bool` (default False).
- Flat-fat: Two-pass evaluation with RNG-jittered thresholds (+/- 10%). If passes disagree, pick higher-scoring candidate via heuristic score (infection rate reduction estimate).
- Uses matched compute budget without deliberation roles.

**Edge cases:**
1. Zero regions -> all region rules skip, fallback NoOp
2. All resources depleted -> rules 1,2,7 skip (amount=0), falls to comms/escalation/NoOp
3. Population zero in region -> `_compute_infection_rate` returns 0.0, never selected as worst
4. First turn -> `_last_action_kind` is None (no repeat guard), comms rule skips (turn>0)
5. RNG determinism -> same seed + same obs sequence = identical actions

### CortexAgent -- Delegation Wrapper

**act(observation)**:
1. If `budget.is_exhausted()`: return NoOp (skip deliberation).
2. Try: `action, delib_log = deliberator.deliberate(observation, budget)`
3. Catch `BudgetExhaustedError`: log warning, return NoOp.
4. Record `LogEvent(kind="deliberation", turn=_turn_count, data={...})` via logger.
5. If logger raises: catch, `logging.warning(...)`, continue.
6. Increment `_turn_count`, return action.

**reset()**:
1. `deliberator.reset()`
2. `budget.reset(initial_budget)` (stored from construction)
3. `_turn_count = 0`

**Edge cases:**
1. BudgetExhaustedError -> NoOp, log event
2. Budget already exhausted at act() start -> skip deliberation, NoOp
3. Logger failure -> catch, warn, still return action
4. Reset mid-episode -> valid, clears all state
5. Deliberator returns NoOp -> pass through unchanged
6. Non-BudgetExhaustedError -> propagates (not caught)
7. act() after reset() with no new obs -> valid, no stale state

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `AgentProtocol` | `src/protocols/agent.py` | Both implement this |
| `BudgetProtocol` | `src/protocols/budget.py` | CortexAgent budget |
| `LoggerProtocol` | `src/protocols/logger.py` | CortexAgent logging |
| `Observation`, `StakeholderSignal` | `src/schemas/observation.py` | Input |
| `OuterAction` + all variants | `src/schemas/action.py` | Output |
| `EnvConfig` | `src/schemas/config.py` | FlatAgent config |
| `RegionState`, `ResourcePool` | `src/schemas/state.py` | FlatAgent reads obs |
| `BudgetStatus`, `BudgetExhaustedError` | `src/schemas/budget.py` | CortexAgent error handling |
| `LogEvent` | `src/schemas/episode.py` | CortexAgent logging |
| `numpy.random.Generator` | numpy | FlatAgent RNG |

---

## Test Plan

### tests/unit/test_flat_agent.py

```
test_act_returns_outer_action_subclass             -- valid OuterAction, non-empty kind
test_determinism_same_seed                         -- two agents seed=42, identical sequences
test_critical_surge_triggers_deploy                -- rate > 0.15, medical > 0 -> DeployResource
test_resource_depletion_triggers_reallocation      -- medical=0 -> ReallocateBudget
test_zero_regions_returns_noop                     -- regions=() -> NoOp
test_all_resources_zero_falls_through              -- no deploy/realloc actions
test_reset_clears_state                            -- turn=0, history empty
test_fat_mode_two_pass                             -- fat_mode=True, valid action returned
test_periodic_communication                        -- turn 5 -> PublicCommunication
test_escalation_fires_once                         -- high urgency turns 0-2 -> Escalate once
test_population_zero_safe                          -- no ZeroDivisionError
test_determinism_different_seed                    -- seeds 42 vs 99 may differ
```

### tests/unit/test_cortex_agent.py

```
test_act_delegates_to_deliberator                  -- deliberate called once with obs+budget
test_act_returns_deliberator_action                -- exact action instance passed through
test_act_logs_deliberation                         -- logger.record called, kind="deliberation"
test_budget_exhausted_returns_noop                 -- BudgetExhaustedError -> NoOp
test_budget_already_exhausted_skips_deliberation   -- is_exhausted -> deliberate not called
test_logger_failure_does_not_crash                 -- RuntimeError in logger -> action still returned
test_reset_cascades                                -- deliberator.reset + budget.reset called
test_reset_clears_turn_count                       -- _turn_count=0 after reset
test_unexpected_exception_propagates               -- ValueError propagates
test_turn_count_increments                         -- 3 acts -> _turn_count=3
```

### tests/integration/test_agent_env.py

```
test_flat_agent_produces_valid_action              -- realistic obs -> valid OuterAction
test_flat_agent_deterministic_with_same_seed       -- 10 obs, identical action lists
test_cortex_agent_delegates_correctly              -- stub deliberator receives correct args
test_cortex_agent_logs_deliberation_trace          -- stub logger receives LogEvent
test_both_agents_satisfy_agent_protocol            -- isinstance check or structural verification
test_both_agents_handle_reset_cleanly              -- act, reset, act -> no exception
test_flat_and_cortex_produce_outer_action          -- both return OuterAction with kind field
test_cortex_agent_handles_budget_exhaustion        -- BudgetExhaustedError -> NoOp, no crash
test_flat_agent_multi_turn_sequence                -- 20 turns, surge at 8 -> DeployResource appears
test_cortex_agent_reset_mid_episode                -- reset clears state, fresh deliberation
test_flat_agent_fat_mode_valid                     -- fat_mode=True -> valid actions
test_both_agents_accept_empty_observation          -- minimal obs -> valid OuterAction
```

---

## Implementation Order

1. FlatAgent skeleton (NoOp fallback only)
2. FlatAgent helpers (`_worst_region`, `_compute_infection_rate`, `_pick_deploy_amount`)
3. FlatAgent 8-rule cascade
4. FlatAgent fat_mode two-pass
5. tests/unit/test_flat_agent.py (RED->GREEN)
6. CortexAgent (delegation + error handling + logging)
7. tests/unit/test_cortex_agent.py (RED->GREEN)
8. tests/integration/test_agent_env.py
9. `__init__.py` exports

## File Size Targets

| File | Estimated | Hard Max |
|------|-----------|----------|
| flat.py | 180-220 | 400 |
| cortex_agent.py | 70-90 | 400 |
| __init__.py | 5 | 10 |

## Invariants

1. Both satisfy AgentProtocol
2. FlatAgent.act() never calls BudgetProtocol
3. CortexAgent.act() always calls deliberate() unless budget exhausted
4. FlatAgent deterministic for same seed + observation sequence
5. CortexAgent catches BudgetExhaustedError, returns NoOp
6. CortexAgent catches logger failures, continues
7. Neither imports from src/env/ or concrete src/cortex/
8. _compute_infection_rate() never raises ZeroDivisionError
