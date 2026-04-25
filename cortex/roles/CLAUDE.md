# cortex/roles/ -- Cortex Specialist Roles

## Behavior

Five Cortex specialist roles, each implementing `RoleProtocol`.
Every role: single measurable function, typed I/O, known budget cost, visible artifact.
Roles are **stateless** -- receive typed input, return typed artifacts.
No imports from `server/` or `agents/`.

`Observation` (from `models`) now includes `done`, `reward`, and `metadata` fields
inherited from OpenEnv. Perception should pass `metadata` through into
`cleaned_observation`. Planner-generated candidate actions carry `metadata: dict`
(from OpenEnv `Action` base) -- set to `{}` on generated candidates. Executive
can check `observation.done` directly as an early exit signal.

All share: `invoke(input: RoleInput) -> Artifact`.

---

## Exposed APIs

### perception.py -- `PerceptionRole.invoke(input) -> CleanState`
Input: `payload["observation"]`, optional `payload["previous_clean"]`. Cost: 1. Auto every turn.

### world_modeler.py -- `WorldModelerRole.invoke(input) -> BeliefState`
Input: `payload["clean_state"]`, `payload["memory_digest"]`. Cost: 2. On Executive request.

### planner.py -- `PlannerRole.invoke(input) -> Plan`
Input: `payload["belief_state"]`, `payload["goals"]`, `payload["constraints"]`, `payload["memory_digest"]`. Cost: 2.

### critic.py -- `CriticRole.invoke(input) -> Critique`
Input: `payload["candidate"]`, `payload["belief_state"]`, `payload["constraints"]`. Cost: 2.

### executive.py -- `ExecutiveRole.invoke(input) -> ExecutiveDecision`
Input: `payload["artifacts"]`, `payload["budget_status"]`, optional `payload["called_roles"]`. Cost: 1.

---

## Implementation Plans

### perception.py -- Noise Filtering + Anomaly Detection

**Three sequential passes:**

1. **Clamp pass**: For each RegionState, clamp infected/recovered/deceased to [0, population]. NaN/inf -> 0 + anomaly flag.
2. **Consistency pass**: If I+R+D > population, scale proportionally. Corrects rounding/noise.
3. **Staleness pass**: Exclude IncidentReports older than threshold (default 3 turns) from cleaned output.

**Anomaly detection:**
- `SPIKE:<region_id>` -- infected > 2x previous_clean value
- `IMPOSSIBLE:<field>` -- NaN, negative, or exceeds logical bound
- `CONTRADICTION:<source>` -- signal urgency >= 0.8 but region infection rate < 1%
- `GAP:<detail>` -- telemetry aggregates differ from sum of regions by > 5%

**CleanState output:** cleaned_observation (clamped, consistent, stale removed), salient_changes (diff from previous), flagged_anomalies.

**Helpers:** `_clamp_region()`, `_detect_spikes()`, `_detect_contradictions()`, `_diff_observations()`

**Edge cases:** all-zero obs -> no anomalies, note "no active outbreak". NaN values -> replace with 0, flag. Empty regions -> pass through. All reports stale -> `GAP:all_reports_stale`.

### world_modeler.py -- Hidden Variable Estimation + Forecasts

**Hidden variables (dict[str, float]):**
- `true_infected_multiplier`: base 2.0, adjusted by anomaly proportion
- `spread_rate`: growth rate from last 3 turns in memory. Default 1.1 if < 2 data points.
- `resource_depletion_rate`: from ResourcePool snapshots. Default 0.0.

**3 forecast trajectories (5 turns each):**
1. Optimistic: `spread_rate * 0.8^t`
2. Baseline: `spread_rate` constant
3. Pessimistic: `spread_rate * 1.2^t` (1.4^t if contradictions flagged)

**Confidence scoring:** `base(0.8) - 0.1*anomaly_count(cap 0.4) - 0.1*(history<3)`. Clamp [0.1, 1.0].

**Helpers:** `_estimate_spread_rate()`, `_project_trajectory()`, `_compute_confidence()`

**Edge cases:** no memory -> defaults. Contradictory signals -> wider pessimistic spread. Single-turn -> min confidence. All zero infected -> zero forecasts, confidence=0.8.

### planner.py -- Candidate Action Generation

**Two stages:**
1. **Enumerate**: For each region, generate applicable actions (DeployResource if resources available, RestrictMovement if not maxed, RequestData if anomalies/low confidence, Escalate if pessimistic forecast severe, NoOp always).
2. **Prune by constraints**: Remove actions violating active constraints.

**Ranking:** `score = 0.4*urgency + 0.3*resource_efficiency + 0.3*forecast_alignment`. Top N candidates (default 3, configurable via `max_candidates`).

**CandidateAction:** action, rationale (one-sentence from scores), expected_effect, confidence (the score).

**Plan always has >= 1 candidate** (NoOp as fallback if all pruned).

**Helpers:** `_enumerate_actions()`, `_score_candidate()`, `_build_rationale()`

**Edge cases:** no valid actions -> NoOp with confidence=0.1. All constrained -> NoOp. Empty belief state -> low confidence, prioritize RequestData. Resources depleted -> no DeployResource.

### critic.py -- Failure Mode Analysis

**3 failure categories:**
1. **Resource failure**: DeployResource using >80% of pool -> `RESOURCE_EXHAUSTION`. >100% -> `RESOURCE_IMPOSSIBLE`.
2. **Timing failure**: Low confidence (<0.4) + irreversible action (max restriction) -> `LOW_CONFIDENCE_IRREVERSIBLE`.
3. **Cascade failure**: Pessimistic forecast >50% infected in 3 turns + action is NoOp/RequestData -> `INACTION_CASCADE`.

**Risk score:** `0.1(base) + 0.2*failure_count(cap 0.6) + (1-confidence)*0.3`. Clamp [0, 1].

**Policy violations:** Direct constraint matches. Fairness check (targeting region with lower need than others). Unnecessary escalation (confidence > 0.7).

**recommended_amendments:** One `"AMEND: <suggestion>"` per failure/violation.

**Helpers:** `_check_resource_failure()`, `_check_timing_failure()`, `_check_cascade_failure()`, `_check_policy_violations()`, `_compute_risk_score()`

**Edge cases:** single candidate -> normal analysis. All policies violated -> risk_score=1.0. Zero risk indicators -> risk_score=base+confidence_penalty. Extremely high risk -> clamp to 1.0.

### executive.py -- Decision Tree

**Evaluated top-to-bottom, first match fires:**

1. **Budget exhausted (remaining <= 0)**: Act with best plan candidate or NoOp.
2. **Budget critical (remaining <= 2)**: Act if plan exists, else call planner.
3. **No CleanState**: Call perception (should not happen, safety net).
4. **No BeliefState + anomalies**: Call world_modeler.
5. **No Plan**: Call planner.
6. **Plan exists, no Critique**: If top candidate confidence >= 0.8 and budget <= 4 -> act. Else call critic.
7. **Plan + Critique**: risk < 0.3 -> act. risk 0.3-0.7 + budget >= 4 -> call planner (re-plan). risk >= 0.7 -> escalate.
8. **Fallback**: wait if budget > 0, else NoOp.

**Budget heuristics:** 0 -> must act. 1-2 -> converge. 3-5 -> one more role. 6+ -> full tree.

**Role selection:** Never call a role invoked 3+ times this turn. Never call perception (auto). Prefer role filling most critical info gap.

**Helpers:** `_find_artifact()`, `_find_all_artifacts()`, `_best_candidate()`, `_should_escalate()`

**Edge cases:** zero budget -> act immediately. All roles called max times -> act with best or NoOp. Conflicting Plan+Critique -> risk score is tiebreaker. Multiple Plans -> use most recent.

---

## Test Plan -- tests/unit/test_roles.py

### Perception (6 tests)

```
test_perception_returns_clean_state_schema         -- return type, valid fields
test_perception_clamps_negative_values             -- infected=-10 -> 0, IMPOSSIBLE flag
test_perception_detects_spike_anomaly              -- 4x increase -> SPIKE flag
test_perception_detects_contradiction              -- high urgency + low infection -> CONTRADICTION
test_perception_handles_empty_regions              -- no crash, valid CleanState
test_perception_filters_stale_reports              -- old reports removed, recent kept
```

### World Modeler (5 tests)

```
test_world_modeler_returns_belief_state_schema     -- return type, 3 trajectories, confidence in [0.1, 1]
test_world_modeler_uses_defaults_without_memory    -- spread_rate=1.1, depletion=0.0
test_world_modeler_forecast_trajectory_structure   -- 3 dicts, each with 5-element lists
test_world_modeler_confidence_decreases_with_anomalies -- monotonically decreasing, >= 0.1
test_world_modeler_zero_infection_produces_zero_forecast -- all trajectories zero
```

### Planner (6 tests)

```
test_planner_returns_plan_schema                   -- return type, non-empty candidates
test_planner_respects_max_candidates               -- max_candidates=2 -> len <= 2
test_planner_always_returns_at_least_one_candidate -- all constrained -> NoOp
test_planner_excludes_constrained_actions          -- blocked action absent
test_planner_ranks_candidates_by_confidence        -- sorted descending
test_planner_prioritizes_request_data_when_belief_empty -- top candidate is RequestData
```

### Critic (5 tests)

```
test_critic_returns_critique_schema                -- return type, all fields present
test_critic_detects_resource_exhaustion            -- 90% deploy -> RESOURCE_EXHAUSTION
test_critic_flags_low_confidence_irreversible      -- confidence=0.3 + max restriction -> flag
test_critic_risk_score_increases_with_failures     -- monotonically increasing
test_critic_clean_pass_returns_low_risk            -- no failures, risk <= 0.2
```

### Executive (7 tests)

```
test_executive_returns_executive_decision_schema   -- return type, valid decision
test_executive_acts_when_budget_zero               -- decision="act"
test_executive_returns_noop_when_budget_zero_no_plan -- decision="act", NoOp
test_executive_calls_world_modeler_when_anomalies  -- decision="call", target="world_modeler"
test_executive_calls_planner_when_no_plan          -- decision="call", target="planner"
test_executive_escalates_on_high_risk              -- risk=0.8 -> "escalate"
test_executive_acts_on_low_risk_critique           -- risk=0.2 -> "act"
```

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `RoleProtocol` | `protocols.role` | All roles implement this |
| `RoleInput`, `Artifact`, `CleanState`, `BeliefState`, `Plan`, `CandidateAction`, `Critique`, `ExecutiveDecision` | `schemas.artifact` | Role I/O |
| `Observation`, `StakeholderSignal` | `models` | Perception input |
| `RegionState`, `ResourcePool`, `Constraint` | `models` | Data within observations |
| `OuterAction` variants | `models` | Planner/Executive output |
| `BudgetStatus` | `schemas.budget` | Executive input |
| `MemoryDigest` | `schemas.episode` | WorldModeler + Planner input |

**Forbidden:** Nothing from `server/`, `agents/`, `evaluation/`, `tracing/`.
