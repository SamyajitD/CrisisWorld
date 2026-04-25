# protocols/ -- Contract Layer

## Behavior

This directory contains **only** `typing.Protocol` abstract classes.
Zero implementation. Zero side effects. Zero business logic.
Every protocol defines the minimal interface that a concrete class in another
package must satisfy. Protocols are the sole coupling surface between packages.

Only allowed import sources: `models` (env-contract types), `schemas/` (agent-side types).

All protocols use `@runtime_checkable` so concrete implementations can be
verified with `isinstance` checks at composition-root wiring time.

---

## Module Index

| File | Protocol | Implemented by |
|------|----------|----------------|
| `env.py` | `EnvProtocol` | `server/CrisisWorld_environment.py::CrisisWorld` |
| `agent.py` | `AgentProtocol` | `agents/flat.py::FlatAgent`, `agents/cortex_agent.py::CortexAgent` |
| `role.py` | `RoleProtocol` | Each file in `cortex/roles/` |
| `budget.py` | `BudgetProtocol` | `cortex/budget.py::BudgetTracker` |
| `memory.py` | `MemoryProtocol` | `cortex/memory.py::EpisodeMemory` |
| `logger.py` | `LoggerProtocol` | `tracing/tracer.py::EpisodeTracer` |

> `EnvProtocol` is the project's structural-typing mirror of OpenEnv's
> `Environment` ABC. `CrisisWorld` satisfies BOTH `EnvProtocol` (via duck typing)
> and `Environment` (via direct subclassing).

---

## EnvProtocol -- `env.py`

```python
@runtime_checkable
class EnvProtocol(Protocol):
    def reset(self, seed: int | None = None,
              episode_id: str | None = None, **kwargs) -> Observation: ...
    def step(self, action: ActionUnion,
             timeout_s: float | None = None, **kwargs) -> Observation: ...
    @property
    def state(self) -> CrisisState: ...
    def get_metadata(self) -> EnvironmentMetadata: ...
    def close(self) -> None: ...
```

### Contracts

**reset(seed, episode_id)**: Initialize episode. Returns turn-0 Observation.
Same seed = identical trajectory.
- Preconditions: seed >= 0 or None (auto-generate). episode_id optional (auto-generate if None).
- Postconditions: All state reset. `self.state.episode_id == episode_id`. `self.state.step_count == 0`. Returned Observation has `done=False`, `reward=None`.
- Edge cases: negative seed -> ValueError. reset after close -> undefined. Double reset -> valid (re-initializes).

**step(action, timeout_s)**: Advance one turn. Returns Observation with `done` and `reward` set.
- Preconditions: reset() called. Episode not done. action is valid `ActionUnion`.
- Postconditions: `self.state.step_count` incremented. `obs.reward` contains `CompositeReward`. `obs.done=True` on termination. `obs.metadata` contains episode context.
- Edge cases: step before reset -> RuntimeError. step after done -> RuntimeError. Invalid action -> ValueError. timeout_s exceeded -> TimeoutError (future).

**state**: Read-only property returning `CrisisState` snapshot.
- Contains: `episode_id`, `step_count`, current region/resource summaries.
- Postconditions: `state.step_count` matches number of `step()` calls since last `reset()`.

**get_metadata()**: Returns `EnvironmentMetadata(name, description, version)`. Static.

**close()**: Release resources. Idempotent.
- Edge cases: double close -> no-op. close before reset -> no-op.

---

## AgentProtocol -- `agent.py`

```python
@runtime_checkable
class AgentProtocol(Protocol):
    def act(self, observation: Observation) -> ActionUnion: ...
    def reset(self) -> None: ...
```

### Contracts

**act(observation)**: Select next action from observation. Returns valid `ActionUnion`.
- Preconditions: reset() called. observation is valid. `observation.done == False`.
- Postconditions: Returns schema-valid `ActionUnion`. Deterministic for same seed+sequence.
- Edge cases: act before reset -> RuntimeError. observation with `done=True` -> should return `NoOp`.

**reset()**: Clear per-episode state. Idempotent.
- Edge cases: double reset -> no-op. reset mid-episode -> valid.

> **Two-mode operation (OpenEnv):** In training/evaluation mode, agents receive
> observations directly via `act()` calls from the experiment runner. In
> production mode, agents interact with the environment via MCP tools -- the
> protocol remains the same but invocation is through MCP, not direct calls.

---

## RoleProtocol -- `role.py`

```python
@runtime_checkable
class RoleProtocol(Protocol):
    @property
    def role_name(self) -> str: ...
    @property
    def cost(self) -> int: ...
    def invoke(self, role_input: RoleInput) -> Artifact: ...
```

### Contracts

**role_name**: Non-empty lowercase string, stable across calls.
**cost**: Positive integer (>= 1), stable for a given config.
**invoke(role_input)**: Execute one role invocation. Returns typed Artifact.
- Preconditions: role_input.role_name matches self.role_name (else ValueError). Budget charged by CALLER before invoke.
- Postconditions: Returns one frozen Artifact matching role type.
- Edge cases: mismatched role_name -> ValueError. Missing payload keys -> ValueError.

---

## BudgetProtocol -- `budget.py`

```python
@runtime_checkable
class BudgetProtocol(Protocol):
    def charge(self, cost: int) -> None: ...
    def remaining(self) -> BudgetStatus: ...
    def is_exhausted(self) -> bool: ...
    def reset(self, total: int) -> None: ...
```

### Contracts

**charge(cost)**: Deduct cost units. Atomic (all-or-nothing).
- Preconditions: cost > 0 (else ValueError). reset() called.
- Postconditions: If remaining >= cost -> deducted. If remaining < cost -> BudgetExhaustedError, no deduction.
- Edge cases: charge(0) -> no-op. charge(-1) -> ValueError. charge after exhaustion -> BudgetExhaustedError.

**remaining()**: Return frozen BudgetStatus snapshot. total == spent + remaining.
**is_exhausted()**: True iff remaining == 0.
**reset(total)**: Initialize budget. total > 0 (else ValueError). Clears ledger.

---

## MemoryProtocol -- `memory.py`

```python
@runtime_checkable
class MemoryProtocol(Protocol):
    def store(self, key: str, artifact: Artifact) -> None: ...
    def retrieve(self, key: str) -> list[Artifact]: ...
    def digest(self) -> MemoryDigest: ...
    def clear(self) -> None: ...
```

### Contracts

**store(key, artifact)**: Append artifact under key. Append-only, insertion order preserved.
- Preconditions: key non-empty (else ValueError).
- Edge cases: same key 100 times -> 100 entries. Duplicate artifacts kept.

**retrieve(key)**: Return copy of artifacts under key. Empty list if key not found (never KeyError).
**digest()**: Return summarized MemoryDigest. Valid even on empty memory.
**clear()**: Remove all entries. Idempotent.

---

## LoggerProtocol -- `logger.py`

```python
@runtime_checkable
class LoggerProtocol(Protocol):
    def record(self, event: LogEvent) -> None: ...
    def save(self, path: Path) -> Path: ...
    def flush(self) -> None: ...
```

### Contracts

**record(event)**: Buffer a LogEvent. Insertion order preserved.
- Edge cases: record after save -> buffered for next save.

**save(path)**: Write buffered events to JSON at path. Returns written path. Buffer NOT cleared after save.
- Edge cases: save with empty buffer -> writes empty JSON. Path exists -> overwrite.

**flush()**: Clear buffer without writing.
- Edge cases: flush empty buffer -> no-op.

---

## Implementation Plan

Each protocol file is ~20-30 lines. All follow this template:

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
# schema imports...

@runtime_checkable
class XProtocol(Protocol):
    """One-line docstring."""
    def method(self, ...) -> ...: ...
```

Implementation order: env.py -> agent.py -> role.py -> budget.py -> memory.py -> logger.py -> __init__.py

### __init__.py Exports

```python
from protocols.env import EnvProtocol
from protocols.agent import AgentProtocol
from protocols.role import RoleProtocol
from protocols.budget import BudgetProtocol
from protocols.memory import MemoryProtocol
from protocols.logger import LoggerProtocol

__all__ = ["EnvProtocol", "AgentProtocol", "RoleProtocol", "BudgetProtocol", "MemoryProtocol", "LoggerProtocol"]
```

---

## External Dependencies

| Import | Source | Used by |
|--------|--------|---------|
| `Observation` | `models` | EnvProtocol, AgentProtocol |
| `ActionUnion`, `OuterAction` | `models` | EnvProtocol, AgentProtocol |
| `CrisisState` | `models` | EnvProtocol (state property) |
| `EnvironmentMetadata` | `models` | EnvProtocol (get_metadata) |
| `RoleInput`, `Artifact` | `schemas.artifact` | RoleProtocol, MemoryProtocol |
| `BudgetStatus` | `schemas.budget` | BudgetProtocol |
| `MemoryDigest` | `schemas.episode` | MemoryProtocol |
| `LogEvent` | `schemas.episode` | LoggerProtocol |

---

## Test Plan

### tests/unit/test_protocols.py -- Structural Subtyping (3 per protocol)

```
test_env_protocol_structural_subtyping             -- minimal class with reset/step/state/get_metadata/close -> isinstance True
test_env_protocol_rejects_incomplete_impl          -- missing step -> isinstance False
test_env_protocol_rejects_unrelated_object         -- string/int -> isinstance False

test_agent_protocol_structural_subtyping           -- minimal class with act/reset -> isinstance True
test_agent_protocol_rejects_missing_reset          -- act only -> isinstance False
test_agent_protocol_rejects_missing_act            -- reset only -> isinstance False

test_role_protocol_structural_subtyping            -- class with role_name/cost/invoke -> isinstance True
test_role_protocol_rejects_missing_invoke          -- no invoke -> isinstance False
test_role_protocol_accepts_plain_attributes        -- role_name as attr (not @property) -> isinstance True

test_budget_protocol_structural_subtyping          -- all 4 methods -> isinstance True
test_budget_protocol_rejects_missing_charge        -- no charge -> isinstance False
test_budget_protocol_rejects_missing_reset         -- no reset -> isinstance False

test_memory_protocol_structural_subtyping          -- all 4 methods -> isinstance True
test_memory_protocol_rejects_missing_store         -- no store -> isinstance False
test_memory_protocol_rejects_missing_clear         -- no clear -> isinstance False

test_logger_protocol_structural_subtyping          -- all 3 methods -> isinstance True
test_logger_protocol_rejects_missing_save          -- no save -> isinstance False
test_logger_protocol_rejects_missing_flush         -- no flush -> isinstance False
```

### tests/conftest.py -- Mock Implementations (fixtures)

Minimal fakes satisfying each protocol for use by integration tests:

- **MockEnv**: reset stores seed + episode_id, returns hardcoded Observation (done=False, reward=None). step increments turn, done after 3 turns. state property returns CrisisState.
- **MockAgent**: act returns NoOp. reset clears call counter.
- **MockRole**: role_name="mock", cost=1, invoke returns hardcoded CleanState.
- **MockBudget**: tracks total/spent. charge deducts or raises. reset reinitializes.
- **MockMemory**: dict[str, list[Artifact]]. store appends. retrieve returns list or []. clear empties.
- **MockLogger**: list[LogEvent] buffer. record appends. save writes JSON. flush clears.

### tests/unit/test_protocol_mocks.py -- Behavioral Tests (16 tests)

```
test_mock_env_reset_determinism                    -- reset(42) twice -> identical Observation
test_mock_env_step_before_reset_raises             -- RuntimeError
test_mock_env_step_after_done_raises               -- RuntimeError
test_mock_env_close_idempotent                     -- no exception
test_mock_env_state_returns_crisis_state           -- state.episode_id, state.step_count valid
test_mock_env_episode_id_persisted                 -- reset(episode_id="ep-1"), state.episode_id == "ep-1"

test_mock_agent_act_returns_action_union           -- isinstance ActionUnion check

test_mock_budget_charge_success                    -- remaining decremented
test_mock_budget_charge_exhausted                  -- BudgetExhaustedError, no partial deduction
test_mock_budget_charge_zero_raises                -- ValueError
test_mock_budget_reset_clears_state                -- fresh after reset

test_mock_memory_store_and_retrieve                -- stored artifact retrievable
test_mock_memory_retrieve_missing_key              -- returns []
test_mock_memory_clear                             -- all keys empty after clear
test_mock_memory_append_only                       -- two stores -> two retrievals in order

test_mock_logger_record_and_save                   -- event in saved JSON
test_mock_logger_save_empty                        -- empty JSON array
test_mock_logger_flush_clears_buffer               -- save after flush -> empty
test_mock_logger_save_does_not_clear               -- two saves contain same event
```

---

## Risks and Mitigations

- **runtime_checkable false confidence**: isinstance only checks method existence, not signatures. Mitigated by mock behavioral tests and mypy static checking.
- **Protocol changes not propagated**: Adding a method triggers mypy errors in all implementations. Mock tests catch behavioral divergence.
- **Properties in RoleProtocol**: runtime_checkable checks attribute existence for properties. Plain attributes also satisfy. Tested explicitly.
- **EnvProtocol diverges from Environment ABC**: Mitigated by keeping EnvProtocol as strict superset of OpenEnv's Environment signature. CrisisWorld satisfies both.
