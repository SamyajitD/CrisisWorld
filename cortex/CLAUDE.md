# cortex/ -- Cortex Deliberation System

## Behavior

This package implements the structured deliberation loop with five specialist roles.
`CortexDeliberator` orchestrates the inner loop: Perception runs automatically each turn,
then the Executive decides whether to invoke more roles or commit to an action.

- `budget.py` implements `BudgetProtocol` -- every role call goes through `charge()`.
- `memory.py` implements `MemoryProtocol` -- keyed, append-only store for current episode.
- Individual roles live in `roles/` (documented in `roles/CLAUDE.md`).

`CortexDeliberator` operates within an OpenEnv episode lifecycle. It receives
`episode_id` for trace correlation (from `Observation.metadata["episode_id"]`
or passed through by the agent). All artifacts are logged with the episode's
`episode_id` and current `step_count`.

**Hard rules:**
- Never imports from `server/` or `agents/`.
- Every role invocation must go through budget accounting.
- All artifacts are typed Pydantic models -- no free-text reasoning.
- Artifacts are append-only per turn -- never overwrite a previous artifact.

---

## Exposed APIs

### deliberator.py -- CortexDeliberator

```python
class CortexDeliberator:
    def __init__(self, roles: dict[str, RoleProtocol], memory: MemoryProtocol, logger: LoggerProtocol) -> None: ...
    def deliberate(self, observation: Observation, budget: BudgetProtocol) -> tuple[OuterAction, DeliberationLog]: ...
    def reset(self) -> None: ...
```

### budget.py -- BudgetTracker

```python
class BudgetTracker:
    def __init__(self, total_budget: int) -> None: ...
    def charge(self, cost: int) -> None: ...
    def remaining(self) -> BudgetStatus: ...
    def is_exhausted(self) -> bool: ...
    def reset(self, total_budget: int) -> None: ...
    def get_ledger(self) -> list[dict]: ...
```

### memory.py -- EpisodeMemory

```python
class EpisodeMemory:
    def __init__(self) -> None: ...
    def store(self, key: str, artifact: Artifact) -> None: ...
    def retrieve(self, key: str) -> list[Artifact]: ...
    def digest(self) -> MemoryDigest: ...
    def reset(self) -> None: ...
```

---

## Implementation Plan

### Phase 1: budget.py -- BudgetTracker

**Internal state:** `_total: int`, `_spent: int`, `_ledger: list[dict]`

**charge(cost)**:
1. cost < 0 -> ValueError. cost == 0 -> no-op (no ledger entry).
2. If `_total - _spent - cost < 0` -> raise `BudgetExhaustedError(requested=cost, remaining=_total-_spent)`. No deduction.
3. Otherwise: `_spent += cost`, append ledger entry.

**remaining()**: Return `BudgetStatus(total=_total, spent=_spent, remaining=_total-_spent)`.
**is_exhausted()**: Return `_total - _spent == 0`.
**reset(total)**: total <= 0 -> ValueError. Set `_total=total, _spent=0, _ledger=[]`.
**get_ledger()**: Return shallow copy of `_ledger`.

**Edge cases:**
1. charge(0) -> no-op, no ledger entry
2. charge(-1) -> ValueError
3. charge(exactly_remaining) -> succeeds, is_exhausted() True
4. charge(exceeds_by_1) -> BudgetExhaustedError, spent unchanged
5. double-reset -> safe, replaces state
6. reset(0) -> ValueError
7. charge after exhaustion -> BudgetExhaustedError for any cost > 0

### Phase 2: memory.py -- EpisodeMemory

**Internal state:** `_store: dict[str, list[Artifact]]`

**store(key, artifact)**: key empty -> ValueError. Append to `_store[key]`.
**retrieve(key)**: Return `list(_store.get(key, []))` (copy).
**digest()**: Returns `MemoryDigest(num_entries=total_count, keys=sorted_keys, summary={key: count})`.
**reset()**: `_store = {}`.

**Edge cases:**
1. retrieve nonexistent key -> `[]`
2. store same key 100 times -> list of 100
3. digest on empty -> `MemoryDigest(num_entries=0, keys=(), summary={})`
4. store after reset -> works normally
5. empty key -> ValueError
6. retrieve returns copy -> mutation doesn't affect internal state
7. store preserves insertion order

### Phase 3: deliberator.py -- CortexDeliberator

**Constants:** `MAX_INNER_ITERATIONS = 20`, `PERCEPTION_COST = 1`, `EXECUTIVE_COST = 1`

**DeliberationLog** (frozen dataclass, internal):
```
artifacts: tuple[Artifact, ...], iterations: int, budget_at_start: BudgetStatus,
budget_at_end: BudgetStatus, termination_reason: str, forced: bool
```

**__init__(roles, memory, logger)**: Validate "perception" and "executive" in roles (else ValueError).

**deliberate(observation, budget)**:
```
1. budget_at_start = budget.remaining()
2. Run Perception: budget.charge(1), invoke, append artifact, store in memory
   -> If BudgetExhaustedError: return (NoOp, log with forced=True, reason="budget_exhausted")
3. Executive loop (max MAX_INNER_ITERATIONS):
   a. budget.charge(1), invoke Executive
   b. If "act" -> return (target_action, log)
   c. If "wait"/"stop" -> return (NoOp, log)
   d. If "escalate" -> return (Escalate, log)
   e. If "call" -> validate role name, budget.charge(role_cost), invoke role
      -> If BudgetExhaustedError during dispatch: force_act()
      -> If invalid role name: log warning, skip, continue loop
4. If max iterations reached -> force_act(reason="max_iterations")
```

**force_act()**: Try one final Executive call. If budget insufficient or Executive doesn't return "act", return NoOp.

**reset()**: Clear `_turn_artifacts`, call `_memory.reset()`. Do NOT reset budget (owned by caller).

**Edge cases:**
1. Budget exhausted before Perception -> return (NoOp, log forced=True, iterations=0)
2. No roles beyond perception+executive -> Executive can only act/wait/stop
3. Executive always says "call" -> MAX_INNER_ITERATIONS cap
4. Executive returns invalid role name -> skip, continue loop
5. Empty observation -> Perception handles it
6. Budget exhausted mid-loop -> catch BudgetExhaustedError, force_act
7. Budget covers Perception+Executive only -> dispatch raises, force_act
8. Role invoke raises non-budget exception -> let propagate

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `RoleProtocol` | `protocols.role` | Each role injected as this type |
| `BudgetProtocol` | `protocols.budget` | BudgetTracker implements this |
| `MemoryProtocol` | `protocols.memory` | EpisodeMemory implements this |
| `LoggerProtocol` | `protocols.logger` | Injected for turn-level logging |
| `Observation` | `models` | Input to deliberation |
| `OuterAction`, `NoOp`, `Escalate` | `models` | Output + fallback actions |
| `Artifact`, `CleanState`, `BeliefState`, `Plan`, `Critique`, `ExecutiveDecision`, `RoleInput` | `schemas.artifact` | Role I/O types |
| `BudgetStatus`, `BudgetExhaustedError` | `schemas.budget` | Budget tracking |
| `MemoryDigest`, `LogEvent` | `schemas.episode` | Memory + logging |

---

## Test Plan

### tests/unit/test_budget.py

```
test_initial_state_reflects_total                  -- remaining == total, spent == 0, not exhausted
test_charge_deducts_from_remaining                 -- charge 3 from 10 -> remaining 7
test_charge_multiple_times_accumulates             -- 2+3+1 from 10 -> spent 6
test_charge_exact_remaining_exhausts               -- charge 5 from 5 -> exhausted
test_charge_exceeding_remaining_raises             -- charge 6 from 5 -> BudgetExhaustedError, spent unchanged
test_charge_exceeding_by_one_raises                -- 3+3 from 5 -> second raises, spent=3
test_charge_zero_is_noop                           -- charge 0 -> spent=0, no ledger entry
test_charge_negative_raises_value_error            -- charge -1 -> ValueError
test_is_exhausted_false_when_remaining             -- 3 from 5 -> False
test_is_exhausted_true_at_zero                     -- 5 from 5 -> True
test_reset_restores_full_budget                    -- charge 3, reset 10 -> remaining=10, spent=0
test_reset_clears_ledger                           -- charge 2, reset -> ledger empty
test_double_reset_is_safe                          -- reset(10), reset(7) -> total=7
test_reset_with_zero_raises_value_error            -- ValueError
test_reset_with_negative_raises_value_error        -- ValueError
test_ledger_records_each_charge                    -- 2 charges -> 2 entries with correct costs
test_charge_after_exhaustion_raises                -- exhaust, then charge 1 -> BudgetExhaustedError
test_remaining_returns_budget_status_type          -- isinstance BudgetStatus
```

### tests/unit/test_memory.py

```
test_store_and_retrieve_single                     -- 1 artifact retrievable
test_retrieve_nonexistent_key_returns_empty        -- returns []
test_store_same_key_multiple_times                 -- 5 artifacts in order
test_store_same_key_100_times                      -- 100 artifacts
test_store_different_keys_isolated                 -- keys don't bleed
test_digest_on_empty_memory                        -- num_entries=0
test_digest_reflects_stored_counts                 -- correct totals and keys
test_reset_clears_all                              -- all keys empty
test_store_after_reset                             -- only post-reset artifacts
test_retrieve_returns_copy                         -- mutation doesn't affect internal
test_empty_key_raises_value_error                  -- ValueError
test_digest_keys_are_sorted                        -- alphabetical
```

### tests/unit/test_deliberator.py (mock roles)

```
test_perception_runs_first                         -- perception.invoke called before executive
test_executive_act_returns_action                  -- act -> returns target_action
test_executive_wait_returns_noop                   -- wait -> NoOp
test_executive_stop_returns_noop                   -- stop -> NoOp
test_executive_escalate_returns_escalate_action    -- escalate -> Escalate
test_call_dispatches_to_requested_role             -- call world_modeler -> invoked once
test_multiple_role_calls_before_act                -- 3 roles invoked in sequence
test_budget_charged_for_perception                 -- spent >= 1
test_budget_charged_for_executive                  -- act immediately -> spent=2
test_budget_charged_for_dispatched_role            -- planner call -> correct total
test_budget_exhaustion_forces_act                  -- forced=True, reason="budget_exhausted"
test_budget_too_low_for_perception                 -- budget=0 -> NoOp, iterations=0
test_max_iterations_prevents_infinite_loop         -- always "call" -> terminates at 20
test_invalid_role_name_skips_dispatch              -- no exception, executive retried
test_artifacts_are_append_only                     -- log.artifacts length matches invocations
test_memory_stores_perception_output               -- memory.retrieve("perception") non-empty
test_memory_stores_dispatched_role_output           -- memory.retrieve("planner") non-empty
test_logger_records_every_artifact                 -- logger.record call count matches
test_reset_clears_turn_artifacts                   -- second deliberation has fresh artifacts
test_reset_clears_memory                           -- memory empty after reset
test_missing_perception_role_raises                -- ValueError in __init__
test_missing_executive_role_raises                 -- ValueError in __init__
```

### tests/integration/test_cortex_loop.py (stub roles, real budget+memory)

```
test_full_deliberation_perception_to_act           -- 2 artifacts, budget spent=2
test_full_deliberation_with_world_modeler_and_planner -- 6 artifacts, budget=8
test_full_deliberation_with_critic                 -- Critique present
test_budget_exhaustion_midloop_graceful            -- forced=True, NoOp returned
test_memory_persists_across_role_calls_within_turn -- all roles stored
test_memory_isolated_between_episodes              -- reset clears first episode
test_deliberation_log_budget_snapshots             -- start.spent=0, end.spent=total
test_artifact_order_matches_invocation_order       -- exact sequence
test_deterministic_with_same_inputs                -- two runs identical
test_escalate_produces_correct_outer_action        -- Escalate returned
test_max_iterations_with_high_budget               -- terminates, correct artifact count
test_logger_receives_structured_events             -- all events are LogEvent
```

---

## Implementation Order

1. `budget.py` + tests/unit/test_budget.py (RED->GREEN)
2. `memory.py` + tests/unit/test_memory.py (RED->GREEN)
3. `deliberator.py` + tests/unit/test_deliberator.py (RED->GREEN)
4. tests/integration/test_cortex_loop.py

## File Size Targets

| File | Estimated | Hard Max |
|------|-----------|----------|
| budget.py | 60-80 | 150 |
| memory.py | 50-70 | 150 |
| deliberator.py | 150-200 | 400 |
| __init__.py | 5-10 | 20 |
