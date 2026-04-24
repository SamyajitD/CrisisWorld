# src/schemas/ -- Data Layer

## Behavior

This directory contains **only** frozen Pydantic v2 models and one domain exception.
No behavior. No side effects. No business logic. No imports from any other `src/` package.
This is the **leaf node** of the dependency graph -- every other package depends on it.

All models use `model_config = ConfigDict(frozen=True)`. Mutation is forbidden.
State transitions must use `model_copy(update={...})` to produce new instances.

**Imports allowed:** `pydantic`, Python stdlib only. Nothing from `src/`.

---

## Exposed APIs

### observation.py

| Model | Purpose |
|-------|---------|
| `StakeholderSignal` | Noisy signal from a stakeholder source |
| `IncidentReport` | Single reported outbreak incident |
| `Telemetry` | Aggregated numeric outbreak indicators |
| `Observation` | Full observation bundle delivered to the agent each turn |

### action.py

| Model | Purpose |
|-------|---------|
| `OuterAction` | Base model with `kind` literal discriminator |
| `DeployResource` | Send resources to a region |
| `RestrictMovement` | Impose movement restrictions on a region |
| `RequestData` | Request better data from a source (costs budget) |
| `PublicCommunication` | Issue a public statement |
| `Escalate` | Escalate to a higher authority |
| `ReallocateBudget` | Shift budget between categories |
| `NoOp` | Do nothing this turn |
| `ActionUnion` | Annotated union of all seven concrete action types |

### reward.py

| Model | Purpose |
|-------|---------|
| `RewardComponents` | Individual reward terms |
| `CompositeReward` | Weighted composite result with total, components, weights |

### state.py

| Model | Purpose |
|-------|---------|
| `RegionState` | Epidemiological state of one region |
| `ResourcePool` | Available operational resources |
| `Constraint` | Active policy or legal constraint |
| `StepResult` | Return value of `env.step()` |

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
| `RewardWeights` | Weights for the composite reward function |
| `EnvConfig` | CrisisWorld configuration |
| `CortexConfig` | Cortex deliberation configuration |
| `ExperimentConfig` | Full experiment specification |

### episode.py

| Model | Purpose |
|-------|---------|
| `LogEvent` | Single structured log event |
| `TurnRecord` | Complete record of one turn |
| `EpisodeTrace` | Full episode trace (sequence of TurnRecords) |
| `EpisodeResult` | Summary result of one completed episode |
| `MemoryDigest` | Summarized view of episode memory state |

---

## Implementation Plan -- Per-Module Field Specifications

### observation.py

```
StakeholderSignal(BaseModel):
    model_config = ConfigDict(frozen=True)
    source:    str              # "hospital", "media", "government"
    urgency:   float            # 0.0-1.0, validators: ge=0.0, le=1.0
    message:   str
    turn:      int              # ge=0

IncidentReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    region_id:     str
    severity:      float        # 0.0-1.0, validators: ge=0.0, le=1.0
    reported_turn: int          # ge=0
    description:   str = ""

Telemetry(BaseModel):
    model_config = ConfigDict(frozen=True)
    total_infected:  int = 0    # ge=0
    total_recovered: int = 0    # ge=0
    total_deceased:  int = 0    # ge=0
    data_staleness:  int = 0    # ge=0

Observation(BaseModel):
    model_config = ConfigDict(frozen=True)
    turn:                int                            # ge=0
    regions:             tuple[RegionState, ...]
    stakeholder_signals: tuple[StakeholderSignal, ...] = ()
    incidents:           tuple[IncidentReport, ...] = ()
    telemetry:           Telemetry
    resources:           ResourcePool
    active_constraints:  tuple[Constraint, ...] = ()
    budget_status:       BudgetStatus
```

### action.py

All actions use `kind` as `Literal` discriminator for union dispatch.

```
OuterAction(BaseModel):     kind: str, frozen=True
DeployResource:             kind="deploy_resource", resource: str, region_id: str, amount: int (gt=0)
RestrictMovement:           kind="restrict_movement", region_id: str, level: int (ge=0, le=3)
RequestData:                kind="request_data", source: str
PublicCommunication:        kind="public_communication", audience: str, message: str (min_length=1)
Escalate:                   kind="escalate", agency: str
ReallocateBudget:           kind="reallocate_budget", from_category: str, to_category: str, amount: int (gt=0)
                            model_validator: from_category != to_category
NoOp:                       kind="noop"
ActionUnion = Annotated[..., Field(discriminator="kind")]
```

### reward.py

```
RewardComponents:  outcome, timeliness, inner_compute_cost (ge=0), safety_violations (ge=0), comms_quality
CompositeReward:   total: float, components: RewardComponents, weights: dict[str, float]
                   model_validator: weights keys subset of component names
```

### state.py

```
RegionState:   region_id: str, population: int (gt=0), infected/recovered/deceased: int (ge=0), restricted: bool
               model_validator: infected + recovered + deceased <= population
ResourcePool:  medical/personnel/funding: int (ge=0)
Constraint:    name: str, description: str = "", active: bool = True
StepResult:    observation: Observation, reward: CompositeReward, done: bool, info: dict = {}
```

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
RewardWeights:    outcome=1.0, timeliness=0.5, inner_compute_cost=0.1, safety_violations=1.0, comms_quality=0.3 (all ge=0)
EnvConfig:        num_regions=4(gt=0), max_turns=50(gt=0), initial_infected=10(ge=0), noise_level=0.1(ge=0,le=1), telemetry_lag=1(ge=0), reward_weights=RewardWeights()
CortexConfig:     total_budget=20(gt=0), perception/world_modeler/planner/critic/executive cost (gt=0), max_inner_iterations=10(gt=0), memory_enabled=True, critic_enabled=True
ExperimentConfig: seeds: tuple[int,...], conditions: tuple[str,...], env/cortex config, output/trace dirs
                  model_validators: seeds and conditions must be non-empty
```

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
- **Negative/zero values**: `RegionState.population` must be gt=0. All counts ge=0. Sum invariant enforced.
- **Frozen mutation**: Assignment raises `ValidationError`. Use `model_copy(update={...})`.
- **ActionUnion discrimination**: Uses `Field(discriminator="kind")`. Unknown kind raises `ValidationError`.
- **BudgetStatus invariant**: `remaining == total - spent`, enforced by model_validator.
- **ReallocateBudget self-transfer**: `from_category != to_category`, enforced by model_validator.
- **CompositeReward weights**: Only canonical component names allowed in keys.
- **Artifact union**: Plain `Union` alias, discrimination by `isinstance` at runtime.
- **BudgetExhaustedError**: Plain Exception subclass with `requested` and `remaining` attrs.
- **Import ordering** (no cycles): budget -> reward -> config -> artifact -> episode -> state(partial) -> observation -> state(StepResult via TYPE_CHECKING)

---

## Usage Example

```python
from src.schemas.state import RegionState

region = RegionState(
    region_id="north", population=100000,
    infected=50, recovered=10, deceased=2, restricted=False,
)
# region.infected = 100  # FAILS -- frozen, raises ValidationError
updated = region.model_copy(update={"infected": 100})  # Correct
```

---

## External Dependencies

None. Leaf node. Imports only `pydantic` and Python stdlib.
Within `schemas/`, modules MAY import from each other. Use `TYPE_CHECKING`
guards to avoid circular imports between `state.py` and `observation.py`.

---

## Test Plan -- tests/unit/test_schemas.py

### observation.py tests

```
test_stakeholder_signal_construction              -- all fields match
test_stakeholder_signal_urgency_bounds             -- -0.1 and 1.1 raise ValidationError
test_stakeholder_signal_frozen                     -- assignment raises ValidationError
test_incident_report_construction                  -- all fields match
test_incident_report_severity_bounds               -- -0.5 and 2.0 raise ValidationError
test_incident_report_default_description           -- description == ""
test_telemetry_defaults_to_zero                    -- all fields 0
test_telemetry_rejects_negative                    -- total_infected=-1 raises
test_observation_construction                      -- full observation, all fields
test_observation_empty_collections                 -- empty tuples accepted
test_observation_rejects_negative_turn             -- turn=-1 raises
test_observation_frozen                            -- assignment raises
test_observation_model_copy                        -- original unchanged, copy updated
test_observation_serialization_roundtrip            -- model_dump -> reconstruct == original
```

### action.py tests

```
test_noop_construction                             -- kind == "noop"
test_deploy_resource_construction                  -- all fields, kind correct
test_deploy_resource_rejects_zero_amount           -- amount=0 raises
test_restrict_movement_level_bounds                -- -1 and 4 raise
test_public_communication_rejects_empty_message    -- message="" raises
test_reallocate_budget_rejects_self_transfer       -- same from/to raises
test_reallocate_budget_rejects_zero_amount         -- amount=0 raises
test_action_union_discriminator                    -- TypeAdapter validates kind="noop" -> NoOp
test_action_union_rejects_unknown_kind             -- kind="unknown" raises
test_all_actions_frozen                            -- all 7 types, assignment raises
test_deploy_resource_serialization_roundtrip        -- dump/reconstruct == original
```

### reward.py tests

```
test_reward_components_defaults                    -- all 0.0
test_reward_components_rejects_negative_cost       -- inner_compute_cost=-1 raises
test_composite_reward_rejects_unknown_weight_key   -- bogus key raises
test_composite_reward_accepts_valid_keys           -- all 5 keys accepted
test_composite_reward_serialization_roundtrip       -- dump/reconstruct
```

### state.py tests

```
test_region_state_construction                     -- all fields match
test_region_state_defaults                         -- infected=0, restricted=False
test_region_state_rejects_zero_population          -- population=0 raises
test_region_state_rejects_negative_infected        -- infected=-1 raises
test_region_state_rejects_overflow                 -- sum > population raises
test_region_state_allows_exact_population          -- sum == population OK
test_region_state_model_copy                       -- original unchanged
test_resource_pool_defaults_to_zero                -- all 0
test_resource_pool_rejects_negative                -- medical=-1 raises
test_constraint_active_default                     -- active=True
test_step_result_construction                      -- all fields
test_step_result_frozen                            -- assignment raises
test_step_result_serialization_roundtrip            -- dump/reconstruct
```

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
test_reward_weights_defaults                       -- default values match spec
test_reward_weights_rejects_negative               -- outcome=-1 raises
test_env_config_defaults                           -- num_regions=4, etc.
test_env_config_rejects_zero_regions               -- num_regions=0 raises
test_env_config_rejects_noise_above_one            -- noise_level=1.5 raises
test_cortex_config_defaults                        -- total_budget=20, etc.
test_cortex_config_rejects_zero_budget             -- total_budget=0 raises
test_experiment_config_rejects_empty_seeds         -- seeds=() raises
test_experiment_config_rejects_empty_conditions    -- conditions=() raises
test_config_serialization_roundtrips               -- env + cortex dump/reconstruct
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

1. `budget.py` -> 2. `reward.py` -> 3. `config.py` -> 4. `artifact.py` -> 5. `episode.py`
6. `state.py` (RegionState, ResourcePool, Constraint) -> 7. `observation.py`
8. `state.py` (StepResult via TYPE_CHECKING) -> 9. `__init__.py`

Write tests RED before each module, then implement GREEN.
