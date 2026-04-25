# Refactor Plan: Adopt OpenEnv Directory Structure

**Date**: 2026-04-25
**Scope**: Full project restructuring — MetaFinals root becomes an OpenEnv package
**Goal**: The project root follows the exact `openenv init` scaffold. All existing
code (env, cortex, agents, evaluation, logging) is reorganized to live within this
structure. The OpenEnv scaffold files are **non-negotiable**.

---

## OpenEnv Reference

**Repository**: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
**Package**: `openenv-core >= 0.2.1` (BSD 3-Clause, Meta Platforms)
**RFCs**: 001-abstractions, 002-env-spec, 003-mcp-support

### Mandatory Scaffold (from `openenv init`)

```
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── CrisisWorld_environment.py
│   ├── Dockerfile
│   └── requirements.txt
└── uv.lock
```

This is the **non-negotiable** skeleton. Every file listed above MUST exist at
those exact paths relative to the project root.

### Core Contract

```python
class Environment(ABC, Generic[ActT, ObsT, StateT]):
    @abstractmethod
    def reset(self, seed: Optional[int] = None,
              episode_id: Optional[str] = None, **kwargs) -> ObsT: ...
    @abstractmethod
    def step(self, action: ActT,
             timeout_s: Optional[float] = None, **kwargs) -> ObsT: ...
    @property
    @abstractmethod
    def state(self) -> StateT: ...
    # Optional: reset_async(), step_async(), get_metadata(), close()
```

### Required Base Models (`openenv.core.env_server.types`)

```python
class Action(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              arbitrary_types_allowed=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              arbitrary_types_allowed=True)
    done: bool = Field(default=False)
    reward: bool | int | float | None = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class State(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True,
                              arbitrary_types_allowed=True)
    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)
```

### Two-Interface Architecture

| Interface | Consumer | Purpose |
|-----------|----------|---------|
| **HTTP** | Orchestrator / Evaluation runner | `reset()`, `step()`, `state`, health, metrics |
| **MCP** | Agent (production mode) | Environment-defined tools only; `reset`/`step`/`state` NEVER exposed as tools |

---

## Current vs Target Directory Structure

### Current

```
MetaFinals/
├── src/
│   ├── __init__.py
│   ├── protocols/          # Contract layer
│   ├── schemas/            # ALL data models
│   ├── env/                # CrisisWorld implementation
│   ├── cortex/             # Deliberation system
│   ├── agents/             # Agent implementations
│   ├── evaluation/         # Experiment orchestration
│   └── logging/            # Episode tracing
├── tests/
├── configs/
├── inference.py
└── pyproject.toml
```

### Target (OpenEnv Scaffold at Root)

```
MetaFinals/
│
│   ── OPENENV MANDATORY FILES (non-negotiable) ──────────────────────
│
├── __init__.py                        # Package root — exports models + client
├── models.py                          # Env-contract types (Action, Observation, State)
├── client.py                          # CrisisWorldClient (EnvClient subclass)
├── openenv.yaml                       # OpenEnv environment configuration
├── pyproject.toml                     # Project config (merged: OpenEnv + project deps)
├── README.md                          # Environment + project documentation
├── uv.lock                            # uv lockfile (replaces pip)
│
├── server/                            # ── CRISISWORLD SERVER ──
│   ├── __init__.py                    # Exports: CrisisWorld
│   ├── app.py                         # create_fastapi_app(env, Action, Obs)
│   ├── CrisisWorld_environment.py     # CrisisWorld(Environment) — reset/step/state
│   ├── Dockerfile                     # Container for isolated execution
│   ├── requirements.txt               # Server-specific Python dependencies
│   ├── dynamics.py                    # Epidemiological model (from src/env/)
│   ├── regions.py                     # Grid/region state (from src/env/)
│   ├── resources.py                   # Resource pool tracking (from src/env/)
│   ├── stakeholders.py               # Signal generation (from src/env/)
│   ├── constraints.py                # Policy enforcement (from src/env/)
│   ├── actions.py                     # Action validation + delayed effects (from src/env/)
│   ├── observations.py               # Partial observation assembly (from src/env/)
│   ├── rewards.py                     # Composite reward (from src/env/)
│   ├── scenarios.py                   # Seeded scenario generation (from src/env/)
│   ├── termination.py                # Episode end conditions (from src/env/)
│   └── _internal.py                   # InternalState, ScheduledEffect, EpiParams
│
│   ── PROJECT-SPECIFIC DIRECTORIES ──────────────────────────────────
│
├── protocols/                         # Contract layer (updated signatures)
│   ├── __init__.py
│   ├── env.py                         # EnvProtocol (mirrors OpenEnv Environment ABC)
│   ├── agent.py                       # AgentProtocol (+ MCP mode docs)
│   ├── role.py                        # RoleProtocol
│   ├── budget.py                      # BudgetProtocol
│   ├── memory.py                      # MemoryProtocol
│   └── logger.py                      # LoggerProtocol
│
├── schemas/                           # Agent-side data models ONLY
│   ├── __init__.py                    # Re-exports env models from models.py
│   ├── artifact.py                    # CleanState, BeliefState, Plan, Critique, ...
│   ├── budget.py                      # BudgetStatus, BudgetLedger, BudgetExhaustedError
│   ├── config.py                      # CortexConfig, ExperimentConfig
│   └── episode.py                     # LogEvent, TurnRecord, EpisodeTrace, EpisodeResult
│
├── cortex/                            # Deliberation system
│   ├── __init__.py
│   ├── deliberator.py
│   ├── budget.py
│   ├── memory.py
│   └── roles/
│       ├── __init__.py
│       ├── perception.py
│       ├── world_modeler.py
│       ├── planner.py
│       ├── critic.py
│       └── executive.py
│
├── agents/                            # Agent implementations
│   ├── __init__.py
│   ├── flat.py
│   └── cortex_agent.py
│
├── evaluation/                        # Experiment orchestration
│   ├── __init__.py
│   ├── runner.py
│   ├── metrics.py
│   ├── ablations.py
│   └── analysis.py
│
├── logging/                           # Episode tracing
│   ├── __init__.py
│   ├── tracer.py
│   ├── serializer.py
│   └── formatters.py
│
├── tests/
│   ├── unit/
│   │   ├── test_models.py             # Tests for models.py (env-contract types)
│   │   ├── test_dynamics.py
│   │   ├── test_regions.py
│   │   ├── test_resources.py
│   │   ├── test_rewards.py
│   │   ├── test_budget.py
│   │   ├── test_roles.py
│   │   ├── test_schemas.py            # Agent-side schemas only
│   │   ├── test_tracer.py
│   │   ├── test_serializer.py
│   │   └── test_formatters.py
│   ├── integration/
│   │   ├── test_env_step.py
│   │   ├── test_cortex_loop.py
│   │   ├── test_agent_env.py
│   │   └── test_episode.py
│   └── e2e/
│       ├── test_flat_episode.py
│       └── test_cortex_episode.py
│
├── configs/
│   ├── env_default.yaml
│   ├── cortex_default.yaml
│   ├── experiment_ablation.yaml
│   └── reward_weights.yaml
│
├── inference.py                            # Composition root
├── project.md                         # Project specification
├── traces/                            # Git-ignored
└── results/                           # Git-ignored
```

### Key Structural Decisions

1. **No `src/` wrapper** — The `src/` directory is eliminated. `protocols/`, `schemas/`,
   `cortex/`, `agents/`, `evaluation/`, `logging/` move to the project root alongside the
   OpenEnv scaffold files.

2. **No `envs/` nesting** — The project root IS the OpenEnv package. No intermediate
   `envs/crisis_world/` directory.

3. **`server/` contains ALL env implementation** — CrisisWorld internals (dynamics, regions,
   etc.) live inside `server/` alongside the mandatory OpenEnv files.

4. **Environment file naming** — `CrisisWorld_environment.py` (not `crisis_world.py` or
   `world.py`). This follows the OpenEnv naming convention.

5. **`uv` replaces pip** — `uv.lock` at root; use `uv` for dependency management.

6. **`openenv.yaml`** — OpenEnv configuration file at root.

7. **`server/requirements.txt`** — Server-specific deps for Docker builds.

---

## File Migration Map

### OpenEnv Mandatory Files (NEW)

| File | Purpose | Status |
|------|---------|--------|
| `__init__.py` | Root package init — exports models + client | NEW (replaces `src/__init__.py`) |
| `models.py` | All env-contract types | NEW (consolidates schemas) |
| `client.py` | `CrisisWorldClient(EnvClient)` | NEW |
| `openenv.yaml` | OpenEnv environment config | NEW |
| `README.md` | Environment documentation | NEW (or update existing) |
| `uv.lock` | Dependency lock file | NEW (generated by `uv`) |
| `server/__init__.py` | Server package init | NEW |
| `server/app.py` | `create_fastapi_app()` | NEW |
| `server/CrisisWorld_environment.py` | `CrisisWorld(Environment)` | NEW (from `src/env/world.py`) |
| `server/Dockerfile` | Container isolation | NEW |
| `server/requirements.txt` | Server Python deps | NEW |

### Files that MOVE into `server/`

| Current Path | New Path | Notes |
|-------------|----------|-------|
| `src/env/world.py` | `server/CrisisWorld_environment.py` | Rename + subclass `Environment` |
| `src/env/dynamics.py` | `server/dynamics.py` | Update imports |
| `src/env/regions.py` | `server/regions.py` | Update imports |
| `src/env/resources.py` | `server/resources.py` | Update imports |
| `src/env/stakeholders.py` | `server/stakeholders.py` | Update imports |
| `src/env/constraints.py` | `server/constraints.py` | Update imports |
| `src/env/actions.py` | `server/actions.py` | Update imports |
| `src/env/observations.py` | `server/observations.py` | Update imports |
| `src/env/rewards.py` | `server/rewards.py` | Update imports |
| `src/env/scenarios.py` | `server/scenarios.py` | Update imports |
| `src/env/termination.py` | `server/termination.py` | Update imports |

### Files that MOVE to project root (out of `src/`)

| Current Path | New Path | Notes |
|-------------|----------|-------|
| `src/protocols/` | `protocols/` | Same internal structure |
| `src/schemas/` | `schemas/` | Reduced to agent-side only |
| `src/cortex/` | `cortex/` | Same internal structure |
| `src/agents/` | `agents/` | Same internal structure |
| `src/evaluation/` | `evaluation/` | Same internal structure |
| `src/logging/` | `logging/` | Same internal structure |

### Schema Files that MERGE into root `models.py`

| Current Path | What Moves |
|-------------|------------|
| `src/schemas/observation.py` | `StakeholderSignal`, `IncidentReport`, `Telemetry`, `Observation` |
| `src/schemas/action.py` | `OuterAction`, all action variants, `ActionUnion` |
| `src/schemas/reward.py` | `RewardComponents`, `CompositeReward` |
| `src/schemas/state.py` | `RegionState`, `ResourcePool`, `Constraint` |
| `src/schemas/config.py` (partial) | `RewardWeights`, `EnvConfig` |

### Schema Files that STAY in `schemas/` (Agent-Side)

| File | Models |
|------|--------|
| `artifact.py` | `CleanState`, `BeliefState`, `Plan`, `Critique`, `ExecutiveDecision` |
| `budget.py` | `BudgetStatus`, `BudgetLedger`, `LedgerEntry`, `BudgetExhaustedError` |
| `config.py` (reduced) | `CortexConfig`, `ExperimentConfig` |
| `episode.py` | `LogEvent`, `TurnRecord`, `EpisodeTrace`, `EpisodeResult`, `MemoryDigest` |

### Files DELETED

| Path | Reason |
|------|--------|
| `src/` (entire directory) | Contents moved to root; wrapper eliminated |
| `src/env/` | Replaced by `server/` |
| `src/schemas/observation.py` | Merged into `models.py` |
| `src/schemas/action.py` | Merged into `models.py` |
| `src/schemas/reward.py` | Merged into `models.py` |
| `src/schemas/state.py` | Merged into `models.py` |

---

## New Dependency Graph

```
models.py                             ← LEAF (extends openenv.core base types)
    ↑
server/*                              ← imports models.py only (never from schemas/)
    ↑
client.py                             ← imports models.py only
    ↑
schemas/                              ← imports from models.py for env types
    ↑
protocols/                            ← imports from models.py + schemas/
    ↑
cortex/                               ← imports protocols/ + schemas/ + models.py
agents/                               ← imports protocols/ + schemas/ + models.py
evaluation/                           ← imports protocols/ + schemas/ + models.py
logging/                              ← imports protocols/ + schemas/
    ↑
inference.py                               ← wires everything
```

**Forbidden imports:**
- `server/` NEVER imports from `schemas/`, `cortex/`, `agents/`, `evaluation/`, `logging/`
- `cortex/` NEVER imports from `server/` (only from `models.py`)
- `agents/` NEVER imports from `server/` (only from `models.py`)
- `models.py` imports NOTHING from the project (only `openenv.core` + `pydantic`)

---

## `models.py` — Design

Root-level file consolidating all environment-contract types. Extends OpenEnv bases.

### Contents

```python
"""CrisisWorld environment models — Action, Observation, State types.

Extends OpenEnv base models. This is the leaf node of the project's
dependency graph — no imports from any other project module.
"""
from openenv.core.env_server.types import (
    Action as BaseAction,
    Observation as BaseObservation,
    State as BaseState,
)

# --- Observation components ---
class StakeholderSignal(BaseModel): ...
class IncidentReport(BaseModel): ...
class Telemetry(BaseModel): ...

# --- State components ---
class RegionState(BaseModel): ...
class ResourcePool(BaseModel): ...
class Constraint(BaseModel): ...

# --- Reward ---
class RewardWeights(BaseModel): ...
class RewardComponents(BaseModel): ...
class CompositeReward(BaseModel): ...

# --- Config ---
class EnvConfig(BaseModel): ...

# --- Observation (extends OpenEnv) ---
class Observation(BaseObservation):
    """Full observation bundle. Inherits done, reward, metadata from OpenEnv."""
    turn: int
    regions: tuple[RegionState, ...]
    stakeholder_signals: tuple[StakeholderSignal, ...] = ()
    incidents: tuple[IncidentReport, ...] = ()
    telemetry: Telemetry
    resources: ResourcePool
    active_constraints: tuple[Constraint, ...] = ()
    budget_status: ...  # from schemas/budget.py via TYPE_CHECKING
    # done: bool          ← inherited from BaseObservation
    # reward: float|None  ← inherited from BaseObservation
    # metadata: dict      ← inherited from BaseObservation

# --- Actions (extend OpenEnv) ---
class OuterAction(BaseAction):
    """Base action. Inherits metadata from OpenEnv."""
    kind: str
    # metadata: dict  ← inherited from BaseAction

class DeployResource(OuterAction): kind = "deploy_resource"; ...
class RestrictMovement(OuterAction): kind = "restrict_movement"; ...
class RequestData(OuterAction): kind = "request_data"; ...
class PublicCommunication(OuterAction): kind = "public_communication"; ...
class Escalate(OuterAction): kind = "escalate"; ...
class ReallocateBudget(OuterAction): kind = "reallocate_budget"; ...
class NoOp(OuterAction): kind = "noop"
ActionUnion = Annotated[..., Field(discriminator="kind")]

# --- State (extends OpenEnv) ---
class CrisisState(BaseState):
    """Server-side episode state. Inherits episode_id, step_count from OpenEnv."""
    # episode_id: str|None  ← inherited from BaseState
    # step_count: int        ← inherited from BaseState
    regions: tuple[RegionState, ...] = ()
    resources: ResourcePool | None = None

# --- Metadata ---
class EnvironmentMetadata(BaseModel):
    name: str
    description: str
    version: str
```

### `model_config` Resolution

- **Most models**: `ConfigDict(frozen=True, extra="forbid")` — keeps project immutability, adds OpenEnv's extra-forbid.
- **`CrisisState`**: `ConfigDict(extra="allow")` per OpenEnv `State` base — NOT frozen, server mutates `step_count`.
- **`Observation`/`OuterAction`**: Inherit `model_config` from OpenEnv bases, override with `frozen=True` where possible.

### `StepResult` Disposition

**Removed entirely.** Its fields absorbed into `Observation`:
- `done` → `Observation.done` (inherited from OpenEnv)
- `reward` → `Observation.reward` (inherited from OpenEnv)
- `observation` → IS the `Observation`
- `info` → `Observation.metadata` (inherited from OpenEnv)

---

## `server/CrisisWorld_environment.py` — Design

```python
from openenv.core.env_server.interfaces import Environment
from models import ActionUnion, Observation, CrisisState, EnvironmentMetadata

class CrisisWorld(Environment[ActionUnion, Observation, CrisisState]):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation: ...
    def step(self, action, timeout_s=None, **kwargs) -> Observation: ...

    @property
    def state(self) -> CrisisState: ...

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="CrisisWorld",
            description="Outbreak-control environment for budgeted deliberation",
            version="0.1.0",
        )

    def close(self) -> None: ...
```

**Changes from current `world.py`:**
1. Renamed to `CrisisWorld_environment.py` (OpenEnv convention).
2. Subclasses `Environment` instead of implementing `EnvProtocol` only.
3. `reset()` accepts `episode_id`, tracks it on `CrisisState`.
4. `step()` accepts `timeout_s`, returns `Observation` (with `done`+`reward`), NOT `StepResult`.
5. New `state` property returns `CrisisState(episode_id, step_count, ...)`.
6. New `get_metadata()` returns env metadata.
7. `SUPPORTS_CONCURRENT_SESSIONS = False` class attribute.

---

## `client.py` — Design

```python
from openenv.core.client import EnvClient
from models import ActionUnion, Observation, CrisisState

class CrisisWorldClient(EnvClient[ActionUnion, Observation, CrisisState]):
    """Typed client for connecting to a remote CrisisWorld server."""
    pass
```

---

## `server/app.py` — Design

```python
from openenv.core.env_server import create_fastapi_app
from models import ActionUnion, Observation
from server.CrisisWorld_environment import CrisisWorld

env = CrisisWorld()
app = create_fastapi_app(env, ActionUnion, Observation)
```

---

## `openenv.yaml` — Design

```yaml
name: crisis_world
version: "0.1.0"
description: "Outbreak-control environment for budgeted deliberation research"
entry_point: server.CrisisWorld_environment:CrisisWorld
models:
  action: models:ActionUnion
  observation: models:Observation
  state: models:CrisisState
```

---

## `server/requirements.txt`

```
openenv-core>=0.2.1,<0.3
pydantic>=2.0
numpy>=1.24
fastapi>=0.110
uvicorn[standard]>=0.29
```

---

## `pyproject.toml` — Rewrite

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "crisis-world"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openenv-core>=0.2.1,<0.3",
    "pydantic>=2.0",
    "numpy>=1.24",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
]
server = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.ruff]
target-version = "py311"
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
```

---

## Import Path Changes (Global Search-Replace)

All `src.*` prefixes are dropped. Everything is at root.

| Old Import | New Import |
|-----------|-----------|
| `from src.schemas.observation import ...` | `from models import ...` |
| `from src.schemas.action import ...` | `from models import ...` |
| `from src.schemas.reward import ...` | `from models import ...` |
| `from src.schemas.state import RegionState, ...` | `from models import RegionState, ...` |
| `from src.schemas.state import StepResult` | **REMOVED** — use `Observation` directly |
| `from src.schemas.config import RewardWeights, EnvConfig` | `from models import RewardWeights, EnvConfig` |
| `from src.schemas.config import CortexConfig, ExperimentConfig` | `from schemas.config import CortexConfig, ExperimentConfig` |
| `from src.schemas.artifact import ...` | `from schemas.artifact import ...` |
| `from src.schemas.budget import ...` | `from schemas.budget import ...` |
| `from src.schemas.episode import ...` | `from schemas.episode import ...` |
| `from src.protocols.env import EnvProtocol` | `from protocols.env import EnvProtocol` |
| `from src.protocols.agent import AgentProtocol` | `from protocols.agent import AgentProtocol` |
| `from src.protocols.* import ...` | `from protocols.* import ...` |
| `from src.env import CrisisWorld` | `from server import CrisisWorld` |
| `from src.env.world import CrisisWorld` | `from server.CrisisWorld_environment import CrisisWorld` |
| `from src.env.dynamics import ...` | `from server.dynamics import ...` |
| (all `src.env.*`) | `server.*` |
| `from src.cortex import ...` | `from cortex import ...` |
| `from src.cortex.roles import ...` | `from cortex.roles import ...` |
| `from src.agents import ...` | `from agents import ...` |
| `from src.evaluation import ...` | `from evaluation import ...` |
| `from src.logging import ...` | `from logging import ...` |

**Note on `logging/`**: The project's `logging/` package shadows Python's stdlib `logging`.
This was already the case under `src/logging/`. If it causes issues, import as
`from logging import tracer` (project) vs `import logging as stdlib_logging`.

---

## CLAUDE.md Refactoring — Detailed Per-File Specifications

### Classification

| Category | Files | Scope |
|----------|-------|-------|
| **FULL REWRITE** | `.claude/CLAUDE.md`, `schemas/CLAUDE.md`, `protocols/CLAUDE.md`, `server/CLAUDE.md` (new) | Content changes fundamentally |
| **MODERATE REWRITE** | `agents/CLAUDE.md`, `evaluation/CLAUDE.md` | Structure preserved, significant section rewrites |
| **TARGETED EDITS** | `cortex/CLAUDE.md`, `cortex/roles/CLAUDE.md`, `logging/CLAUDE.md` | Import paths + specific OpenEnv additions |

### Update Order (Dependency-Aware)

```
Phase A (Leaf nodes — no other CLAUDE.md depends on these):
  A.1: schemas/CLAUDE.md          ← scope change affects all downstream refs
  A.2: server/CLAUDE.md           ← new file, independent of others

Phase B (Contract layer — all downstream references this):
  B.1: protocols/CLAUDE.md        ← EnvProtocol rewrite; depends on A.1

Phase C (Consumers — depend on A + B being settled; parallel-safe):
  C.1: cortex/CLAUDE.md
  C.2: cortex/roles/CLAUDE.md
  C.3: agents/CLAUDE.md
  C.4: evaluation/CLAUDE.md
  C.5: logging/CLAUDE.md

Phase D (Master document — references everything):
  D.1: .claude/CLAUDE.md          ← must be LAST
```

---

### A.1 `schemas/CLAUDE.md` — FULL REWRITE

**Location**: `schemas/CLAUDE.md` (moved from `src/schemas/CLAUDE.md`)

**Scope change**: Directory is halved. Env-contract types (observation, action,
reward, state) move to root `models.py`. Only agent-side types remain.

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | `schemas/ -- Agent-Side Data Layer` |
| Behavior | Rewrite | Scope narrowed to agent-side. `__init__.py` re-exports env types from `models.py` for convenience. Imports allowed: `pydantic`, stdlib, `models` (re-export only) |
| Exposed APIs — observation.py | **DELETE** | Moved to `models.py` |
| Exposed APIs — action.py | **DELETE** | Moved to `models.py` |
| Exposed APIs — reward.py | **DELETE** | Moved to `models.py` |
| Exposed APIs — state.py | **DELETE** | `RegionState`/`ResourcePool`/`Constraint` → `models.py`. `StepResult` removed entirely |
| Exposed APIs — artifact.py | Keep | No changes |
| Exposed APIs — budget.py | Keep | No changes |
| Exposed APIs — config.py | Edit | Remove `RewardWeights`, `EnvConfig` (moved to `models.py`). Keep `CortexConfig`, `ExperimentConfig` |
| Exposed APIs — episode.py | Keep | No changes |
| All Implementation Plans | Same pattern | Delete observation/action/reward/state plans. Keep artifact/budget/config(reduced)/episode |
| Edge Cases | Edit | Remove StepResult and env-type edge cases |
| Usage Example | Rewrite | `from models import RegionState` (not `from src.schemas.state`) |
| External Dependencies | Edit | Add `models` as import source. Remove "Nothing" claim |
| Test Plan | Halved | Delete env-type tests (move to `test_models.py`). Keep artifact/budget/config/episode tests |
| Implementation Order | Rewrite | 4 files only: `budget.py` → `config.py` → `artifact.py` → `episode.py` → `__init__.py` |

**New sections:**
- `__init__.py` re-export specification (imports from `models` + re-exports)
- Cross-reference to `models.py` for env-contract types

**Estimated size**: ~200 lines (down from ~394)

---

### A.2 `server/CLAUDE.md` — NEW FILE (replaces `src/env/CLAUDE.md`)

**Location**: `server/CLAUDE.md`

**Content structure** (modeled on `src/env/CLAUDE.md`):

| Section | Content |
|---------|---------|
| Title | `server/ -- CrisisWorld Server (OpenEnv Environment)` |
| Behavior | Subclasses `Environment` ABC (not just `EnvProtocol`). Contains all env implementation. **Hard rules**: imports ONLY from `models.py`; never from `schemas/`, `cortex/`, `agents/`, etc. All state transitions return new objects. All stochastic ops require explicit `rng`. |
| OpenEnv Mandatory Files | Table listing `CrisisWorld_environment.py`, `app.py`, `Dockerfile`, `requirements.txt` with their purpose |
| Exposed APIs — `CrisisWorld_environment.py` | `CrisisWorld(Environment[ActionUnion, Observation, CrisisState])`. Methods: `reset(seed, episode_id)`, `step(action, timeout_s)`, `state` property, `get_metadata()`, `close()`. `SUPPORTS_CONCURRENT_SESSIONS = False` |
| Exposed APIs — `app.py` | `create_fastapi_app(env, ActionUnion, Observation)` |
| Exposed APIs — helper modules | Same as old `src/env/CLAUDE.md`: dynamics, regions, resources, stakeholders, constraints, actions, observations, rewards, scenarios, termination. All signatures preserved, import paths updated |
| Internal Data Structures | `InternalState` (adds `episode_id`), `ScheduledEffect`, `EpiParams`, `ScenarioParams` |
| Module Implementation Plans | All 10 helper module plans carried over from `src/env/CLAUDE.md` unchanged in logic. Only import references change from `src.schemas.*` to `models.*` |
| Step Pipeline | 12-step pipeline. Changes: step returns `Observation` (with `done`+`reward` set) not `StepResult`. Add `state.step_count` increment. Add `episode_id` tracking |
| External Dependencies | All `src/schemas/*` → `models.*`. `StepResult` removed. Add `openenv.core.env_server.interfaces.Environment`. Add `openenv.core.env_server.types.State` |
| Test Plan | Carried from old env tests. Updated imports. New tests: `test_state_property`, `test_get_metadata`, `test_episode_id_persisted`, `test_step_returns_observation_with_done_reward` |
| Implementation Order | `_internal.py` → `regions.py` → `resources.py` → `dynamics.py` → `scenarios.py` → `constraints.py` → `actions.py` → `stakeholders.py` → `observations.py` → `rewards.py` → `termination.py` → `CrisisWorld_environment.py` → `app.py` |
| Dockerfile spec | `FROM python:3.11-slim`, `COPY requirements.txt`, `RUN pip install`, `COPY server/`, `CMD uvicorn` |
| Server startup | `uvicorn server.app:app --port 8000` |

**Estimated size**: ~330 lines

---

### B.1 `protocols/CLAUDE.md` — FULL REWRITE

**Location**: `protocols/CLAUDE.md` (moved from `src/protocols/CLAUDE.md`)

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | `protocols/ -- Contract Layer` (drop `src/`) |
| Behavior | Edit | Import sources: `models` + `schemas/`. Drop `src/` references |
| Module Index — "Implemented by" | Rewrite | `src/env/world.py::CrisisWorld` → `server/CrisisWorld_environment.py::CrisisWorld`. All `src/agents/` → `agents/`. All `src/cortex/` → `cortex/`. All `src/logging/` → `logging/` |
| **EnvProtocol** | **Major rewrite** | New full contract (see below) |
| AgentProtocol | Edit | Add MCP mode documentation: "In production mode, agents receive observations via MCP tools, not direct `act()` calls. Protocol stays the same for training/evaluation" |
| RoleProtocol | Path edit | Drop `src/` |
| BudgetProtocol | Path edit | Drop `src/` |
| MemoryProtocol | Path edit | Drop `src/` |
| LoggerProtocol | Path edit | Drop `src/` |
| External Dependencies | Rewrite | All `src/schemas/*` → `models.*` (env types) or `schemas.*` (agent types). Remove `StepResult`. Add `CrisisState`, `EnvironmentMetadata` from `models` |
| Test Plan | Edit | Add tests for `state` property, updated step return type |
| Risks | Edit | Add: "EnvProtocol diverges from Environment ABC — mitigated by keeping as strict superset" |

**EnvProtocol new contract:**

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

**Contract changes:**
- `reset()`: adds `episode_id` param. Post: `self.state.episode_id == episode_id`, `self.state.step_count == 0`
- `step()`: adds `timeout_s`. Returns `Observation` (NOT `StepResult`). `obs.done` indicates termination. `obs.reward` is the composite reward. Post: `self.state.step_count` incremented
- `state`: read-only property returning `CrisisState` with `episode_id` + `step_count`
- `get_metadata()`: returns `EnvironmentMetadata(name, description, version)`

**Estimated size**: ~320 lines (up from ~291)

---

### C.1 `cortex/CLAUDE.md` — TARGETED EDITS

**Location**: `cortex/CLAUDE.md` (moved from `src/cortex/CLAUDE.md`)

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | Drop `src/` |
| Hard rules | Edit | "Never imports from `server/` or `agents/`" (was `src/env/`, `src/agents/`) |
| External Dependencies | Rewrite all rows | Env types (`Observation`, `OuterAction`, `NoOp`, `Escalate`) → `from models import ...`. Agent types (`Artifact`, `RoleInput`, `BudgetStatus`) → `from schemas.* import ...`. Protocols → `from protocols.* import ...` |
| Implementation Plan — deliberator.py | Add | "DeliberationLog includes `episode_id` for trace correlation" |

**New content**: Note about `episode_id` propagation through deliberation

**Estimated size**: ~265 lines (minimal change)

---

### C.2 `cortex/roles/CLAUDE.md` — TARGETED EDITS

**Location**: `cortex/roles/CLAUDE.md` (moved from `src/cortex/roles/CLAUDE.md`)

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | Drop `src/` |
| Hard rules | Edit | "No imports from `server/` or `agents/`" |
| Implementation Plans — perception.py | Add | "`Observation` now has `done`, `reward`, `metadata` fields — Perception passes `metadata` through into `cleaned_observation`" |
| Implementation Plans — planner.py | Add | "Generated actions carry `metadata` field (from OpenEnv `Action` base). Set `metadata={}` on candidates" |
| Implementation Plans — executive.py | Add | "`observation.done` directly available — Executive can use as early exit signal" |
| External Dependencies | Rewrite all rows | Same pattern: env types from `models`, agent types from `schemas.*` |
| Forbidden imports | Edit | "Nothing from `server/`, `agents/`, `evaluation/`, `logging/`" |

**Estimated size**: ~200 lines (minimal change)

---

### C.3 `agents/CLAUDE.md` — MODERATE REWRITE

**Location**: `agents/CLAUDE.md` (moved from `src/agents/CLAUDE.md`)

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | Drop `src/` |
| Hard rules | Edit | "Never imports from `server/`" (was `src/env/`). "Never imports concrete classes from `cortex/`" (was `src/cortex/`) |
| Exposed APIs — FlatAgent | Edit | `act()` receives `Observation` with `done`+`reward`. Use `observation.done` for early exit |
| Exposed APIs — CortexAgent | Edit | Add two-mode operation note |
| Implementation Plan — FlatAgent | Edit | `EnvConfig` from `models`. `RegionState`/`ResourcePool` from `models`. `Observation` from `models` |
| Implementation Plan — CortexAgent | Edit | Add: "If `observation.done`, return `NoOp` immediately". Mention `CrisisWorldClient` for future remote execution |
| External Dependencies | Rewrite | All `src/*` → root-level. Env types from `models`, agent types from `schemas.*` |
| Test Plan | Edit | Add: `test_cortex_agent_noop_when_done`, `test_flat_agent_uses_observation_done` |
| Invariants | Edit | "Neither imports from `server/` or concrete `cortex/`" |

**New sections:**
- Two-mode operation (training: direct `act()` calls; production: MCP tools)
- `CrisisWorldClient` for remote env interaction (future)

**Estimated size**: ~220 lines (up from ~214)

---

### C.4 `evaluation/CLAUDE.md` — MODERATE REWRITE

**Location**: `evaluation/CLAUDE.md` (moved from `src/evaluation/CLAUDE.md`)

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | Drop `src/` |
| Behavior | Rewrite | Add: "Runner is the OpenEnv **orchestrator** — manages HTTP interface to env server. Generates `episode_id` per episode. Uses `env.state` for step tracking" |
| Exposed APIs — runner.py | Rewrite | `run_episode()` calls `env.reset(seed, episode_id=uuid4().hex)`. Loop reads `obs.done` and `obs.reward` directly. Uses `env.state.step_count`. No `StepResult` |
| Implementation Plan — runner.py | Rewrite | Step-by-step: `obs = env.reset(seed, episode_id=...)`, loop `obs = env.step(action)`, check `obs.done`, reward from `obs.reward`. Server lifecycle management |
| Implementation Plan — metrics.py | Edit | `CompositeReward` from `models`. Metric extraction from `Observation.reward` |
| External Dependencies | Rewrite | All `src/*` gone. `StepResult` row deleted. Env types from `models`. Agent types from `schemas.*` |
| Test Plan | Edit | Add: `test_episode_id_generated_per_episode`, `test_runner_uses_env_state_step_count` |
| Edge Cases | Edit | Add: "episode_id collision — uuid4, statistically impossible" |

**New sections:**
- OpenEnv orchestrator role description
- `episode_id` generation and propagation
- `CrisisWorldClient` for remote execution
- Server lifecycle management (start/stop Docker environments)

**Estimated size**: ~230 lines (up from ~203)

---

### C.5 `logging/CLAUDE.md` — TARGETED EDITS

**Location**: `logging/CLAUDE.md` (moved from `src/logging/CLAUDE.md`)

| Section | Action | Details |
|---------|--------|---------|
| Title | Edit | Drop `src/` |
| Implementation Plan — tracer.py | Edit | `episode_id` sourced from `CrisisState.episode_id`. `TurnRecord.turn` aligns with `CrisisState.step_count` |
| Implementation Plan — formatters.py | Edit | Render `Observation.metadata` in turn output. Render `Action.metadata` if non-empty |
| External Dependencies | Rewrite | All `src/*` changed. `CompositeReward` and `Observation` from `models`. Agent types from `schemas.*` |
| Test Plan | Edit | Add: `test_tracer_episode_id_from_crisis_state`, `test_formatter_renders_observation_metadata` |

**New content:**
- Note about `episode_id` sourcing from `CrisisState`
- Note about logging `Observation.metadata` and `Action.metadata`
- Note about `logging/` shadowing stdlib — use `import logging as stdlib_logging` in project code

**Estimated size**: ~230 lines (minimal change)

---

### D.1 `.claude/CLAUDE.md` — FULL REWRITE

**Location**: `.claude/CLAUDE.md`

This is the master document. Every section changes.

| Section | Action | Details |
|---------|--------|---------|
| Title + overview | Minor edit | Same project description |
| High-Level Architecture | Rewrite | Add Two-Interface model (HTTP + MCP). Diagram shows `server/` not `src/env/`, `models.py` as leaf. Show OpenEnv scaffold prominently |
| Directory Structure | **Full rewrite** | Replace entire `src/`-based tree with Target layout from this plan |
| Directory Responsibilities | Rewrite per directory | (see table below) |
| Dependency Graph | Rewrite | `models.py` is leaf. `server/` imports only `models.py`. Diagram from this plan |
| Absolute Rules 1-9 | Edit | Drop all `src/` references. Rule 2: note `CrisisState` exception (not frozen). Rule 10: Phase 1 includes OpenEnv scaffold |
| Commands — Environment Setup | Rewrite | `uv sync` replaces `pip install -e`. Add `uv sync --extra dev --extra server` |
| Commands — Running Tests | Rewrite | `--cov=.` replaces `--cov=src`. Add `test_models.py` |
| Commands — Linting | Rewrite | `. tests/` replaces `src/ tests/`. `mypy .` replaces `mypy src/` |
| Commands — Running Episodes | Rewrite | Add `uvicorn server.app:app --port 8000`. Add Docker commands |
| Commands — Git Workflow | Rewrite | Update pre-commit command paths |
| Reward Function | No change | |
| Ablation Conditions | No change | |
| Open Decisions | Edit | Add: "`StepResult` removed", "`CrisisState` not frozen", "`episode_id` flow" |

**Directory Responsibilities rewrites:**

| Directory | Key changes |
|-----------|-------------|
| `models.py` | **NEW section.** "Env-contract types. Extends OpenEnv bases. Leaf node. No project imports." |
| `client.py` | **NEW section.** "Typed EnvClient subclass for remote access." |
| `server/` | Replaces `src/env/`. "CrisisWorld(Environment) — reset/step/state. Imports ONLY from models.py." |
| `schemas/` | "Agent-side data models ONLY. Re-exports env types from models.py." |
| `protocols/` | Updated signatures. "EnvProtocol mirrors OpenEnv Environment ABC." |
| `cortex/` | Path change only |
| `agents/` | Path change + two-mode note |
| `evaluation/` | "OpenEnv orchestrator. Uses HTTP interface." |
| `logging/` | "Uses episode_id from CrisisState." |

**New sections to add:**
- `models.py` specification (contents, model_config, StepResult removal)
- `client.py` specification
- `openenv.yaml` specification
- `server/app.py` specification
- OpenEnv scaffold compliance checklist
- Import path migration table

**Estimated size**: ~550 lines (up from ~430)

---

### Cross-Cutting Rules (Apply to ALL CLAUDE.md Files)

#### Rule 1: Zero `src/` References

Every `src/` or `src.` in import paths, dependency tables, code examples,
cross-references, and prose must be removed.

| Find pattern | Replace with |
|-------------|-------------|
| `src/schemas/observation` | `models` |
| `src/schemas/action` | `models` |
| `src/schemas/reward` | `models` |
| `src/schemas/state` | `models` |
| `src/schemas/config` (RewardWeights, EnvConfig) | `models` |
| `src/schemas/config` (CortexConfig, ExperimentConfig) | `schemas.config` |
| `src/schemas/artifact` | `schemas.artifact` |
| `src/schemas/budget` | `schemas.budget` |
| `src/schemas/episode` | `schemas.episode` |
| `src/protocols/*` | `protocols.*` |
| `src/env/*` | `server.*` |
| `src/env/world` | `server.CrisisWorld_environment` |
| `src/cortex/*` | `cortex.*` |
| `src/agents/*` | `agents.*` |
| `src/evaluation/*` | `evaluation.*` |
| `src/logging/*` | `logging.*` |

#### Rule 2: Zero `StepResult` References

Every CLAUDE.md that mentions `StepResult` must be updated:

| File | Change |
|------|--------|
| `schemas/CLAUDE.md` | Delete entire `StepResult` spec |
| `protocols/CLAUDE.md` | `step()` returns `Observation` not `StepResult` |
| `server/CLAUDE.md` | Step pipeline returns `Observation` with `done`+`reward` |
| `evaluation/CLAUDE.md` | Runner reads `obs.done`, `obs.reward` directly |
| `.claude/CLAUDE.md` | EnvProtocol promise updated |

#### Rule 3: OpenEnv Base Type Inheritance Noted

Every file referencing `Observation`, `OuterAction`, or `CrisisState` must note:
- `Observation` inherits `done: bool`, `reward: float|None`, `metadata: dict`
- `OuterAction` inherits `metadata: dict`
- `CrisisState` inherits `episode_id: str|None`, `step_count: int`

#### Rule 4: `CrisisState` Frozen Exception

Root CLAUDE.md and `server/CLAUDE.md` must explicitly document that `CrisisState`
is the **sole exception** to the `frozen=True` rule — it uses `ConfigDict(extra="allow")`
because the server mutates `step_count` between steps.

---

### Verification Checklist (Run After All Updates)

```bash
# 1. Zero src/ references in any CLAUDE.md
grep -r "src/" --include="CLAUDE.md" .
# Expected: 0 matches (excluding git-ignored, quoted historical refs)

# 2. Zero StepResult references
grep -r "StepResult" --include="CLAUDE.md" .
# Expected: 0 matches

# 3. Cross-reference consistency
# Every "External Dependencies" path must map to an actual file in target structure

# 4. EnvProtocol signature consistency
# protocols/CLAUDE.md, server/CLAUDE.md, and .claude/CLAUDE.md must all show:
#   reset(seed, episode_id) -> Observation
#   step(action, timeout_s) -> Observation
#   state -> CrisisState
#   get_metadata() -> EnvironmentMetadata
```

---

## Implementation Order

### Phase 0 — Preparation & Verification
```
0.1  uv init (if not already)           # Set up uv in project
0.2  uv add openenv-core                # Verify package installs
0.3  Run existing tests                  # Green baseline
0.4  git commit current state            # Safety checkpoint
```

### Phase 1 — Create OpenEnv Scaffold (Empty Shell)
```
1.1  Create root models.py              # Empty placeholder with imports
1.2  Create root client.py              # Empty CrisisWorldClient
1.3  Create openenv.yaml                # Environment configuration
1.4  Create server/__init__.py          # Empty
1.5  Create server/app.py               # Placeholder
1.6  Create server/CrisisWorld_environment.py  # Empty class stub
1.7  Create server/Dockerfile           # Basic Python container
1.8  Create server/requirements.txt     # Server deps
1.9  Update pyproject.toml              # Add openenv-core, update packages
1.10 Update root __init__.py            # Replace src/__init__.py role
```

### Phase 2 — Models Migration (Leaf Node First)
```
2.1  Populate models.py
     - Move types from src/schemas/ (observation, action, reward, state)
     - Move RewardWeights + EnvConfig from config.py
     - Extend OpenEnv base types (add done, reward, metadata)
     - Add CrisisState with episode_id + step_count
     - Add EnvironmentMetadata
     - Remove StepResult

2.2  Create tests/unit/test_models.py
     - All env-contract type tests

2.3  Update schemas/ (move out of src/)
     - Move src/schemas/ → schemas/
     - Delete observation.py, action.py, reward.py, state.py
     - Reduce config.py (CortexConfig + ExperimentConfig only)
     - Update __init__.py to re-export from models
     - Update CLAUDE.md

2.4  Update tests/unit/test_schemas.py
     - Agent-side only

2.5  Run tests — fix import errors
```

### Phase 3 — Environment Migration
```
3.1  Move src/env/*.py → server/
     - world.py → CrisisWorld_environment.py
     - All other .py files keep their names
     - Update all internal imports

3.2  Refactor CrisisWorld in CrisisWorld_environment.py
     - Subclass Environment[ActionUnion, Observation, CrisisState]
     - Add state property
     - Add episode_id to reset()
     - Add timeout_s to step()
     - Return Observation (with done+reward) from step()
     - Add get_metadata()
     - Add SUPPORTS_CONCURRENT_SESSIONS = False

3.3  Populate server/app.py
     - create_fastapi_app() wiring

3.4  Populate client.py
     - CrisisWorldClient typed subclass

3.5  Delete src/env/ entirely

3.6  Update env tests — fix import paths

3.7  Run tests
```

### Phase 4 — Move Remaining Code Out of `src/`
```
4.1  Move src/protocols/ → protocols/
4.2  Move src/cortex/ → cortex/
4.3  Move src/agents/ → agents/
4.4  Move src/evaluation/ → evaluation/
4.5  Move src/logging/ → logging/

4.6  Update ALL imports across every file
     - Drop src. prefix everywhere
     - Env types from models, not schemas
     - Update test imports

4.7  Update protocols/env.py
     - New reset() signature (episode_id)
     - New step() signature (timeout_s, returns Observation)
     - Add state property
     - Add get_metadata()

4.8  Update inference.py imports

4.9  Delete src/ directory entirely

4.10 Run full test suite
```

### Phase 5 — CLAUDE.md Refactoring (Dependency-Ordered)

This phase follows the order defined in the "CLAUDE.md Refactoring" section above.
Each step corresponds to a phase label (A/B/C/D).

```
Phase 5A — Leaf Nodes (parallel-safe):
  5A.1  schemas/CLAUDE.md              # FULL REWRITE — scope halved
  5A.2  server/CLAUDE.md               # NEW FILE — replaces src/env/CLAUDE.md

Phase 5B — Contract Layer:
  5B.1  protocols/CLAUDE.md            # FULL REWRITE — EnvProtocol signatures

Phase 5C — Consumers (parallel-safe):
  5C.1  cortex/CLAUDE.md               # TARGETED EDITS — imports + episode_id
  5C.2  cortex/roles/CLAUDE.md         # TARGETED EDITS — imports + new fields
  5C.3  agents/CLAUDE.md               # MODERATE REWRITE — two-mode operation
  5C.4  evaluation/CLAUDE.md           # MODERATE REWRITE — orchestrator role
  5C.5  logging/CLAUDE.md              # TARGETED EDITS — episode_id + metadata

Phase 5D — Master Document:
  5D.1  .claude/CLAUDE.md              # FULL REWRITE — must be LAST
```

### Phase 6 — Final Documentation + Lockfile
```
6.1  Write README.md (OpenEnv environment docs)
6.2  Generate uv.lock (uv lock)
6.3  Run verification checklist (grep for src/, StepResult, cross-ref consistency)
6.4  Final full test suite + coverage check
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Massive import breakage | HIGH | Phase-gated test runs after every move; `schemas/__init__` re-exports as bridge |
| `openenv-core` unavailable or API mismatch | HIGH | Phase 0.2 verifies upfront; pin `>=0.2.1,<0.3`; vendor base types as fallback |
| `logging/` shadows stdlib `logging` | MEDIUM | Already existed under `src/logging`; use `import logging as stdlib_logging` where needed |
| `models.py` at root conflicts with common name | LOW | OpenEnv convention; no stdlib conflict |
| `frozen=True` vs OpenEnv `validate_assignment` | MEDIUM | Use frozen (project rule); CrisisState is sole exception |
| `StepResult` removal cascades widely | HIGH | Phase 2 adds done+reward to Observation first; consumers fix in Phase 4 |
| uv migration from pip | LOW | uv is drop-in compatible; existing deps transfer cleanly |
| git history fragmentation from moves | LOW | Use `git mv` for all moves; single commit per phase |

---

## Success Criteria

**OpenEnv Scaffold (non-negotiable):**
- [ ] `__init__.py` at root — exports models + client
- [ ] `models.py` at root — extends OpenEnv Action/Observation/State
- [ ] `client.py` at root — typed `EnvClient` subclass
- [ ] `openenv.yaml` at root — environment configuration
- [ ] `pyproject.toml` at root — includes openenv-core
- [ ] `README.md` at root — environment documentation
- [ ] `uv.lock` at root — dependency lockfile
- [ ] `server/__init__.py` — exports CrisisWorld
- [ ] `server/app.py` — `create_fastapi_app()` entry point
- [ ] `server/CrisisWorld_environment.py` — `Environment` subclass
- [ ] `server/Dockerfile` — container specification
- [ ] `server/requirements.txt` — server deps

**Architecture:**
- [ ] `src/` directory fully eliminated
- [ ] `CrisisWorld` subclasses `Environment` ABC with all required methods
- [ ] `Observation` carries `done`, `reward`, `metadata` (OpenEnv inheritance)
- [ ] `Action` models carry `metadata` (OpenEnv inheritance)
- [ ] `CrisisState` has `episode_id` + `step_count`
- [ ] `StepResult` fully removed
- [ ] `state` property implemented
- [ ] `episode_id` flows from reset through logging and evaluation

**CLAUDE.md Completeness:**
- [ ] 9 existing CLAUDE.md files updated (correct paths, no `src/`)
- [ ] 1 new CLAUDE.md created (`server/CLAUDE.md`)
- [ ] Zero `src/` in any CLAUDE.md import path or dependency table
- [ ] Zero `StepResult` in any CLAUDE.md
- [ ] EnvProtocol signature consistent across `protocols/`, `server/`, `.claude/` CLAUDE.md files
- [ ] `Observation` documented as carrying `done`, `reward`, `metadata` in every referencing file
- [ ] `CrisisState` non-frozen exception documented in root and server CLAUDE.md
- [ ] `schemas/CLAUDE.md` covers exactly 4 files: artifact.py, budget.py, config.py (reduced), episode.py
- [ ] `server/CLAUDE.md` covers all OpenEnv mandatory files + all migrated env modules

**Quality:**
- [ ] All existing tests pass with updated imports
- [ ] New tests cover OpenEnv-specific models
- [ ] Coverage >= 80%
- [ ] `ruff check` + `mypy` clean
