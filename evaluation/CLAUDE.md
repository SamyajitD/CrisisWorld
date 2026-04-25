# evaluation/ -- Experiment Orchestration

## Behavior

Orchestrates multi-seed experiments, collects metrics, configures ablation conditions,
produces comparison tables. Receives env/agent **factories** via injection -- never
imports concrete env/agent classes. Consumed only by `inference.py`.

**OpenEnv orchestrator role:** The `ExperimentRunner` acts as the **orchestrator**
in OpenEnv's two-interface model. It manages the HTTP simulation-control interface
(`reset`/`step`/`state`). It generates a unique `episode_id` per episode and
passes it to `env.reset(seed, episode_id)`. It uses `env.state` to read
`step_count` and episode metadata between steps. `CrisisWorldClient` can be
used for remote execution against a Docker-hosted environment.

---

## Exposed APIs

### runner.py -- ExperimentRunner

```python
class ExperimentRunner:
    def __init__(self, env_factory, agent_factory, logger_factory, config: ExperimentConfig) -> None: ...
    def run(self) -> ExperimentResults: ...
    def run_episode(self, env, agent, logger, seed) -> EpisodeResult: ...
```

### metrics.py

```python
def collect_episode_metrics(trace: EpisodeTrace) -> EpisodeMetrics: ...
def aggregate_metrics(episodes: list[EpisodeMetrics]) -> AggregateMetrics: ...
def compute_confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]: ...
```

### ablations.py

```python
class AblationCondition: ...  # frozen Pydantic: name, agent_type, budget, enabled_roles, memory/critic enabled
def build_conditions(config: ExperimentConfig) -> list[AblationCondition]: ...
def get_matched_budget(config) -> int: ...
def get_low_budget(config) -> int: ...
```

### analysis.py

```python
def comparison_table(results: ExperimentResults) -> str: ...
def diagnostic_report(results: ExperimentResults) -> str: ...
def significance_summary(results: ExperimentResults) -> str: ...
```

---

## Implementation Plan

### runner.py -- ExperimentRunner

**run()**: For each condition -> for each seed:
1. Create env, agent, logger from factories
2. `run_episode(env, agent, logger, seed)`
3. Collect metrics, aggregate per condition
4. Crashed episodes: log error, skip, continue. `env.close()` in finally block.
5. Return ExperimentResults with all conditions.

**run_episode()**: `episode_id = uuid4().hex` -> `obs = env.reset(seed, episode_id=episode_id)` -> `agent.reset()` -> loop: `action = agent.act(obs)`, `obs = env.step(action)`, `logger.record(...)`, build TurnRecord, read reward from `obs.reward`, check `obs.done` -> until done -> read `env.state.step_count` for final turn count -> finalize trace -> return EpisodeResult.

**Error handling:**
| Failure | Handling |
|---------|----------|
| agent.act() raises | Caught, episode skipped, logged |
| env.step() raises | Caught, episode skipped, logged |
| logger.save() raises | Warning logged, result still returned |
| All seeds fail | aggregate returns NaN aggregate, condition still in results |
| env.close() raises | Caught in finally, warning logged |

### metrics.py -- Metric Definitions

**Primary:** total_cumulative_reward (sum), outbreak_duration (turns with infected>0), final_mortality_rate (deceased/population)

**Secondary:** resource_efficiency (deployed/available), comms_quality_score (sum of comms component), constraint_violation_count (turns with violations)

**Diagnostic:** role_call_frequency (dict[str, int]), budget_spend_rate (spent/turns), inner_loop_iterations_per_turn (mean)

**aggregate_metrics()**: mean, std, 95% CI via t-distribution. Empty list -> NaN aggregate with n=0. Single episode -> std=0, CI=(nan, nan).

**collect_episode_metrics()**: Single pass over trace.turns. Missing info keys use `.get()` with defaults.

### ablations.py -- Five Conditions

| Condition | agent_type | budget | roles | memory | critic |
|-----------|-----------|--------|-------|--------|--------|
| flat-lite | flat | low (10) | () | no | no |
| flat-fat | flat | matched (50) | () | no | no |
| cortex-lite | cortex | low (10) | perception+planner+executive | no | no |
| cortex-full | cortex | matched (50) | all 5 | yes | yes |
| cortex-tuned | cortex | matched (50) | all 5 + tuned executive | yes | yes |

"Matched compute" = same budget units. Flat gets budget but doesn't consume via roles.
`build_conditions()` filters by `config.conditions` if specified. Unknown names -> warning, skipped.

### analysis.py -- Reporting

**comparison_table()**: Markdown table. Conditions as rows, metrics as columns. Format: `mean +/- std`.
**diagnostic_report()**: Role-call frequency table (Cortex only), budget spend rate, inner-loop iterations.
**significance_summary()**: Pairwise CI overlap analysis. Non-overlapping -> "likely significant". Overlap -> "not significant". Single condition -> "Insufficient conditions".

---

## Edge Cases

**runner.py:** Episode crashes -> skip+log. All seeds fail -> NaN aggregate. env.close() raises -> caught. Empty seeds -> empty results.
**metrics.py:** Zero-turn trace -> metrics=0/NaN. Missing info keys -> defaults. Single seed -> std=0. All reward zero -> valid.
**ablations.py:** Unknown condition name -> warn+skip. Missing budget config -> defaults. Duplicate names -> deduplicate.
**analysis.py:** All NaN -> "N/A". One condition -> no pairwise. Zero std -> CI collapses.

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `EnvProtocol` | `protocols.env` | Factory return type |
| `AgentProtocol` | `protocols.agent` | Factory return type |
| `LoggerProtocol` | `protocols.logger` | Factory return type |
| `ExperimentConfig` | `schemas.config` | Configuration |
| `EpisodeResult`, `EpisodeTrace`, `TurnRecord`, `LogEvent` | `schemas.episode` | Episode data |
| `CompositeReward`, `RewardComponents` | `models` | Metric extraction |
| `BudgetStatus` | `schemas.budget` | Diagnostic metrics |
| `Observation` | `models` | Turn record (carries `done`, `reward`, `metadata`) |
| `ActionUnion` | `models` | Turn record |
| `CrisisState` | `models` | Read from `env.state` for step_count |
| `scipy.stats.t` | scipy | CI calculation |
| `uuid4` | `uuid` (stdlib) | Generate episode_id |
| `statistics`, `math`, `logging`, `pathlib` | stdlib | Aggregation, logging, paths |

> **Removed:** `StepResult` -- no longer exists. `env.step()` returns `Observation`
> directly. Read `obs.done` for termination and `obs.reward` for the composite reward.

---

## Test Plan -- tests/integration/test_episode.py

### Fixtures

```
fake_env_factory      -- FakeEnv: 5 turns, predictable obs/rewards
fake_agent_factory    -- FakeAgent: always NoOp, cortex conditions populate info
fake_logger_factory   -- FakeLogger: in-memory, save is no-op
experiment_config     -- seeds=[42,43,44], all five conditions
```

### Runner Tests

```
test_run_single_seed_single_condition              -- 1 key, 1 episode, n=1
test_run_multi_seed_produces_correct_count         -- 3 seeds -> 3 episodes, n=3
test_run_all_conditions_produces_five_results      -- 5 condition keys
test_run_episode_returns_correct_turn_count        -- 5 turns -> num_turns=5
test_episode_crash_is_logged_and_skipped           -- crashing env, 1 of 2 succeeds
test_env_close_called_even_on_crash                -- close() called per attempt
```

### Metric Tests

```
test_collect_episode_metrics_primary_values        -- known rewards -> correct sum
test_collect_episode_metrics_handles_missing_info  -- no role_calls -> empty dict
test_aggregate_metrics_mean_and_std                -- [10,20,30] -> mean=20, std=10
test_aggregate_metrics_empty_list_returns_nan      -- n=0, all NaN
```

### Ablation Tests

```
test_build_conditions_returns_five                 -- 5 conditions, correct names
test_flat_conditions_have_no_roles                 -- enabled_roles=(), memory=False
test_cortex_full_has_all_roles_and_memory          -- 5 roles, memory=True
test_matched_budget_conditions_share_budget        -- flat-fat, cortex-full, cortex-tuned same
test_low_budget_conditions_share_budget            -- flat-lite, cortex-lite same
test_build_conditions_with_subset_filter           -- conditions=["flat-fat","cortex-full"] -> 2
```

### Analysis Tests

```
test_comparison_table_format                       -- contains "|", both condition names
test_comparison_table_nan_shows_na                 -- NaN -> "N/A"
test_diagnostic_report_skips_flat                  -- no flat in role-call table
test_significance_summary_non_overlapping_ci       -- "likely significant"
test_significance_summary_single_condition         -- "Insufficient conditions"
```

---

## Implementation Order

1. `ablations.py` (no deps on other eval modules)
2. `metrics.py` (depends only on schemas)
3. `runner.py` (depends on ablations + metrics)
4. `analysis.py` (depends on metrics output types)
5. `__init__.py` (re-exports)

## File Size Targets

| File | Estimated | Hard Max |
|------|-----------|----------|
| runner.py | 120-150 | 400 |
| metrics.py | 100-130 | 400 |
| ablations.py | 80-100 | 400 |
| analysis.py | 100-130 | 400 |
| __init__.py | 10-15 | 20 |
