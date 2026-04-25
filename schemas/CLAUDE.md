# schemas/ -- Agent-Side Data Layer

## Behavior

This directory contains **only** frozen Pydantic v2 models and one domain exception
for the **agent side** of the system: Cortex artifacts, budget accounting, experiment
configuration, and episode tracing.

No behavior. No side effects. No business logic.

**Environment-contract types** (Observation, Action, Reward, RegionState, etc.) live
in root `models.py`, which extends OpenEnv base types. This directory does NOT
contain those types.

All models use `model_config = ConfigDict(frozen=True)`. Mutation is forbidden.
State transitions must use `model_copy(update={...})` to produce new instances.

**Imports allowed:** `pydantic`, Python stdlib, `models` (for re-export in `__init__.py` only).
Nothing from `server/`, `cortex/`, `agents/`, `evaluation/`, or `logging/`.

---

## Exposed APIs

### artifact.py

| Model | Purpose |
|-------|---------|
| `RoleInput` | Typed input envelope for a role invocation |
| `CleanState` | Perception output |
| `BeliefState` | World Modeler output |
| `CandidateAction` | Single candidate within a Plan |
| `Plan` | Planner output: list of candidates |
| `Critique` | Critic output |
| `ExecutiveDecision` | Executive output |
| `Artifact` | Union type of all five role outputs |

### budget.py

| Model | Purpose |
|-------|---------|
| `BudgetStatus` | Snapshot of budget state |
| `BudgetLedger` | Full budget history with entries |
| `LedgerEntry` | Single charge record in the ledger |
| `BudgetExhaustedError` | Exception raised when budget is exceeded |

### config.py

| Model | Purpose |
|-------|---------|
| `CortexConfig` | Cortex deliberation configuration |
| `ExperimentConfig` | Full experiment specification |

> `RewardWeights` and `EnvConfig` have moved to root `models.py`.

### episode.py

| Model | Purpose |
|-------|---------|
| `LogEvent` | Single structured log event |
| `TurnRecord` | Complete record of one turn |
| `EpisodeTrace` | Full episode trace (sequence of TurnRecords) |
| `EpisodeResult` | Summary result of one completed episode |
| `MemoryDigest` | Summarized view of episode memory state |

### `__init__.py`

Re-exports all agent-side models above, plus env-contract types from `models`
for convenience:

```python
# Agent-side (canonical)
from schemas.artifact import CleanState, BeliefState, Plan, Critique, ...
from schemas.budget import BudgetStatus, BudgetLedger, ...
from schemas.config import CortexConfig, ExperimentConfig
from schemas.episode import EpisodeTrace, EpisodeResult, ...

# Env-contract re-exports (canonical source: models.py)
from models import (
    Observation, OuterAction, ActionUnion, RegionState, ResourcePool,
    Constraint, CompositeReward, RewardComponents, RewardWeights, EnvConfig,
    CrisisState, EnvironmentMetadata, BudgetStatus as _BudgetStatus,
)
```

---

## Implementation Plan -- Per-Module Field Specifications

### artifact.py

```
RoleInput:          role_name: str, payload: dict[str, object]
CleanState:         salient_changes/flagged_anomalies: tuple[str, ...] = (), cleaned_observation: dict = {}
BeliefState:        hidden_var_estimates: dict = {}, forecast_trajectories: tuple[dict, ...] = (), confidence: float (ge=0, le=1)
CandidateAction:    action: dict, rationale: str, expected_effect: str, confidence: float (ge=0, le=1)
Plan:               candidates: tuple[CandidateAction, ...] = ()
Critique:           failure_modes/policy_violations/recommended_amendments: tuple[str, ...] = (), risk_score: float (ge=0, le=1)
ExecutiveDecision:  decision: Literal["act","call","wait","escalate","stop"], target_action: dict|None, target_role: str|None, reasoning: str
Artifact = CleanState | BeliefState | Plan | Critique | ExecutiveDecision
```

### budget.py

```
BudgetStatus:   total: int (ge=0), spent: int (ge=0), remaining: int (ge=0)
                model_validator: remaining == total - spent
LedgerEntry:    role_name: str, cost: int (gt=0), turn: int (ge=0)
BudgetLedger:   entries: tuple[LedgerEntry, ...] = (), status: BudgetStatus
BudgetExhaustedError(Exception): stores requested: int, remaining: int
```

### config.py

```
CortexConfig:     total_budget=20(gt=0), perception/world_modeler/planner/critic/executive cost (gt=0), max_inner_iterations=10(gt=0), memory_enabled=True, critic_enabled=True
ExperimentConfig: seeds: tuple[int,...], conditions: tuple[str,...], env/cortex config, output/trace dirs
                  model_validators: seeds and conditions must be non-empty
```

> `RewardWeights` and `EnvConfig` specs are in root `models.py`.

### episode.py

```
LogEvent:       kind: str, turn: int (ge=0), data: dict = {}
TurnRecord:     turn: int (ge=0), observation/action/reward/budget_snapshot: dict|None, artifacts: tuple[dict,...] = (), events: tuple[LogEvent,...] = ()
EpisodeTrace:   episode_id: str, turns: tuple[TurnRecord,...] = (), seed: int, condition: str = "", metadata: dict = {}
EpisodeResult:  episode_id: str, seed: int (ge=0), condition: str, total_turns: int (ge=0), total_reward: float, termination_reason: str, metrics: dict = {}
MemoryDigest:   num_entries: int (ge=0), keys: tuple[str,...] = (), summary: dict = {}
```

---

## Edge Cases and Design Decisions

- **Empty collections**: All tuple fields default to `()`. Code must handle gracefully.
- **Frozen mutation**: Assignment raises `ValidationError`. Use `model_copy(update={...})`.
- **BudgetStatus invariant**: `remaining == total - spent`, enforced by model_validator.
- **Artifact union**: Plain `Union` alias, discrimination by `isinstance` at runtime.
- **BudgetExhaustedError**: Plain Exception subclass with `requested` and `remaining` attrs.
- **Import ordering** (no cycles): budget -> config -> artifact -> episode -> `__init__.py`

---

## Usage Example

```python
from models import RegionState          # Env-contract type (canonical from models.py)
from schemas.budget import BudgetStatus  # Agent-side type (canonical from schemas/)

region = RegionState(
    region_id="north", population=100000,
    infected=50, recovered=10, deceased=2, restricted=False,
)
# region.infected = 100  # FAILS -- frozen, raises ValidationError
updated = region.model_copy(update={"infected": 100})  # Correct

budget = BudgetStatus(total=20, spent=3, remaining=17)
```

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `pydantic` | Third-party | All model definitions |
| `BudgetStatus` | `schemas.budget` | Referenced by `models.Observation.budget_status` |
| Env-contract types | `models` | Re-exported by `__init__.py` for convenience |

Within `schemas/`, modules MAY import from each other (e.g., `config` imports
`BudgetStatus` type reference). No circular imports exist in the reduced scope.

---

## Test Plan -- tests/unit/test_schemas.py

### artifact.py tests

```
test_role_input_construction                       -- fields match
test_clean_state_defaults                          -- all empty
test_belief_state_confidence_bounds                -- -0.1 and 1.1 raise
test_candidate_action_confidence_bounds            -- 1.5 raises
test_plan_empty_candidates                         -- candidates == ()
test_critique_risk_score_bounds                    -- -0.1 and 1.1 raise
test_executive_decision_valid_decisions            -- all 5 literals accepted
test_executive_decision_rejects_invalid            -- "think" raises
test_artifact_union_isinstance                     -- each type passes isinstance
test_all_artifacts_frozen                          -- all 5 types, assignment raises
```

### budget.py tests

```
test_budget_status_construction                    -- fields match
test_budget_status_invariant_enforced              -- wrong remaining raises
test_budget_status_zero_budget                     -- 0/0/0 OK
test_budget_status_rejects_negative_total          -- total=-1 raises
test_budget_status_frozen                          -- assignment raises
test_ledger_entry_rejects_zero_cost                -- cost=0 raises
test_budget_ledger_empty_entries                   -- entries=() OK
test_budget_exhausted_error_construction            -- requested/remaining attrs accessible
test_budget_exhausted_error_is_exception           -- issubclass(Exception)
test_budget_status_serialization_roundtrip          -- dump/reconstruct
```

### config.py tests

```
test_cortex_config_defaults                        -- total_budget=20, etc.
test_cortex_config_rejects_zero_budget             -- total_budget=0 raises
test_experiment_config_rejects_empty_seeds         -- seeds=() raises
test_experiment_config_rejects_empty_conditions    -- conditions=() raises
test_config_serialization_roundtrips               -- cortex + experiment dump/reconstruct
```

### episode.py tests

```
test_log_event_construction                        -- fields match
test_log_event_rejects_negative_turn               -- turn=-1 raises
test_turn_record_defaults                          -- optional fields default
test_episode_trace_empty_turns                     -- turns=() OK
test_episode_result_rejects_negative_turns         -- total_turns=-1 raises
test_memory_digest_defaults                        -- num_entries=0, keys=()
test_memory_digest_rejects_negative_entries        -- num_entries=-1 raises
test_episode_trace_serialization_roundtrip          -- dump/reconstruct
```

### Cross-cutting tests

```
test_all_models_have_frozen_config                 -- meta-test: every model's config["frozen"] is True
```

---

## Implementation Order

1. `budget.py` -> 2. `config.py` (reduced) -> 3. `artifact.py` -> 4. `episode.py` -> 5. `__init__.py`

Write tests RED before each module, then implement GREEN.
