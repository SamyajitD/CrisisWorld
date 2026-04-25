# CrisisWorld + Cortex

Outbreak-control environment and structured deliberation agent.
Central question: does budgeted structured reasoning (Cortex) beat a flat policy under matched compute?

## High-Level Architecture

Two tightly coupled systems connected through an OpenEnv step loop with a
Two-Interface architecture (HTTP for orchestration, MCP for production agents):

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Episode Runner (Orchestrator)                      │
│  episode_id = uuid4()                                               │
│  obs = env.reset(seed, episode_id)                                  │
│  for turn in episode:                                               │
│    action     = agent.act(obs)             # Agent decides          │
│    obs        = env.step(action)           # CrisisWorld advances   │
│    traces.record(obs, action)              # Everything is logged   │
│    if obs.done: break                      # Termination check      │
└─────────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐       ┌───────────────────────────────────┐
│   CrisisWorld   │       │         Agent (swappable)         │
│   (server/)     │       │                                   │
│                 │       │  ┌─────────┐   ┌───────────────┐  │
│  hidden epi     │       │  │  Flat   │   │    Cortex     │  │
│  state, noisy   │       │  │ Agent   │   │    Agent      │  │
│  observations,  │       │  │ (base-  │   │  ┌─────────┐  │  │
│  resources,     │       │  │  line)  │   │  │Executive│  │  │
│  stakeholders,  │       │  └─────────┘   │  │Planner  │  │  │
│  constraints,   │       │                │  │Critic   │  │  │
│  delayed fx     │       │                │  │Modeler  │  │  │
│                 │       │                │  │Percep.  │  │  │
│  state prop:    │       │                │  │Budget   │  │  │
│  episode_id     │       │                │  └─────────┘  │  │
│  step_count     │       │                                │  │
└─────────────────┘       └───────────────────────────────────┘
```

### Two-Interface Model (OpenEnv)

| Interface | Consumer | Purpose |
|-----------|----------|---------|
| **HTTP** | ExperimentRunner (orchestrator) | `reset()`, `step()`, `state`, health, metrics |
| **MCP** | Agent (production mode, future) | Environment-defined tools only; `reset`/`step`/`state` NEVER exposed |

### Outer loop -- CrisisWorld

Stateful outbreak simulator in `server/`. Subclasses OpenEnv `Environment` ABC.
Maintains hidden epidemiological state, resources, stakeholders, policy constraints,
and delayed action effects. Emits partial, noisy, lagged observations. Computes
composite reward embedded directly in `Observation.reward`.

### Inner loop -- Cortex

Structured deliberation system with five roles (Perception, World Modeler,
Planner, Critic, Executive). Each role invocation has a typed input, typed
output artifact, and a budget cost. The Executive decides whether to think
more or act. Thinking is explicitly expensive.

### Agents

Swappable policies behind a single `AgentProtocol`. Flat agents skip Cortex.
Cortex agents delegate to the deliberation loop. Same interface, different
compute organization -- this is what the experiment isolates.

### Evaluation

Multi-seed runner (OpenEnv orchestrator) that executes episodes under
matched-compute conditions, generates `episode_id` per episode, collects
primary/secondary/diagnostic metrics, and produces ablation comparison tables.

---

## Directory Structure -- Function Signature Promises

The project root follows the exact `openenv init` scaffold. Non-env packages
(protocols, schemas, cortex, agents, evaluation, tracing) live alongside the
OpenEnv mandatory files at the root level. No `src/` wrapper.

```
MetaFinals/
│
│   ── OPENENV MANDATORY FILES ────────────────────────────────────
│
├── __init__.py                        # Package root -- exports models + client
├── models.py                          # Env-contract types (extends OpenEnv bases)
├── client.py                          # CrisisWorldClient (EnvClient subclass)
├── openenv.yaml                       # OpenEnv environment configuration
├── pyproject.toml                     # Project config (OpenEnv + project deps)
├── README.md                          # Environment + project documentation
├── uv.lock                            # uv lockfile
│
├── server/                            # ── CRISISWORLD SERVER ──
│   ├── __init__.py                    # Exports: CrisisWorld
│   ├── app.py                         # create_fastapi_app(env, Action, Obs)
│   ├── CrisisWorld_environment.py     # CrisisWorld(Environment) -- reset/step/state
│   ├── Dockerfile                     # Container for isolated execution
│   ├── requirements.txt               # Server-specific Python deps
│   ├── dynamics.py                    # Epidemiological model
│   ├── regions.py                     # Grid/region state management
│   ├── resources.py                   # Resource pool tracking
│   ├── stakeholders.py               # Signal generation
│   ├── constraints.py                # Policy enforcement
│   ├── actions.py                     # Action validation + delayed effects
│   ├── observations.py               # Partial observation assembly
│   ├── rewards.py                     # Composite reward
│   ├── scenarios.py                   # Seeded scenario generation
│   ├── termination.py                # Episode end conditions
│   └── _internal.py                   # InternalState, ScheduledEffect, EpiParams
│
│   ── PROJECT PACKAGES ──────────────────────────────────────────
│
├── protocols/                         # ── CONTRACT LAYER (no implementation) ──
│   ├── __init__.py
│   ├── env.py                         # EnvProtocol (mirrors OpenEnv Environment ABC)
│   ├── agent.py                       # AgentProtocol (+ MCP mode docs)
│   ├── role.py                        # RoleProtocol
│   ├── budget.py                      # BudgetProtocol
│   ├── memory.py                      # MemoryProtocol
│   └── logger.py                      # LoggerProtocol
│
├── schemas/                           # ── AGENT-SIDE DATA (Pydantic, no logic) ──
│   ├── __init__.py                    # Re-exports env types from models.py
│   ├── artifact.py                    # CleanState, BeliefState, Plan, Critique
│   ├── budget.py                      # BudgetStatus, BudgetLedger
│   ├── config.py                      # CortexConfig, ExperimentConfig
│   └── episode.py                     # EpisodeResult, EpisodeTrace
│
├── cortex/                            # ── CORTEX (implements RoleProtocol per role) ──
│   ├── __init__.py                    # Exports: CortexDeliberator
│   ├── deliberator.py                 # Orchestrates the inner deliberation loop
│   ├── budget.py                      # Budget accounting (implements BudgetProtocol)
│   ├── memory.py                      # Episode memory store (implements MemoryProtocol)
│   └── roles/
│       ├── __init__.py
│       ├── perception.py              # Raw obs -> CleanState + anomalies
│       ├── world_modeler.py           # CleanState -> BeliefState + forecasts
│       ├── planner.py                 # BeliefState -> candidate Plans
│       ├── critic.py                  # Plan -> failure modes + risk score
│       └── executive.py              # All artifacts -> act|call|wait|escalate|stop
│
├── agents/                            # ── AGENTS (implement AgentProtocol) ──
│   ├── __init__.py                    # Exports: FlatAgent, CortexAgent
│   ├── flat.py                        # Direct policy, no deliberation
│   └── cortex_agent.py               # Delegates to CortexDeliberator
│
├── evaluation/                        # ── EVALUATION (OpenEnv orchestrator) ──
│   ├── __init__.py                    # Exports: ExperimentRunner
│   ├── runner.py                      # Multi-seed episode execution
│   ├── metrics.py                     # Metric collection and aggregation
│   ├── ablations.py                   # Condition setup (flat-lite/fat, cortex-lite/full)
│   └── analysis.py                    # Comparison tables, diagnostics
│
├── tracing/                           # ── TRACING (implements LoggerProtocol) ──
│   ├── __init__.py                    # Exports: EpisodeTracer
│   ├── tracer.py                      # Per-turn event recording
│   ├── serializer.py                  # Trace -> JSON/file output
│   └── formatters.py                 # Human-readable trace rendering
│
├── tests/
│   ├── unit/
│   │   ├── test_models.py            # Env-contract types (models.py)
│   │   ├── test_dynamics.py
│   │   ├── test_regions.py
│   │   ├── test_resources.py
│   │   ├── test_rewards.py
│   │   ├── test_budget.py
│   │   ├── test_roles.py
│   │   ├── test_schemas.py           # Agent-side schemas only
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
├── inference.py                            # Composition root -- wires everything
├── project.md                         # Project specification
├── traces/                            # Git-ignored. Episode trace output.
└── results/                           # Git-ignored. Experiment results.
```

### Directory Responsibilities & Inter-Relations

#### `models.py` -- Env-Contract Types (LEAF NODE)
**Promise**: All environment-facing Pydantic models extending OpenEnv base types.
`Observation`, `ActionUnion`, `CrisisState`, `RegionState`, `ResourcePool`,
`CompositeReward`, `EnvConfig`, `EnvironmentMetadata`.
**Depended on by**: every other package.
**Depends on**: `openenv.core` + `pydantic` only. Nothing from the project.

#### `client.py` -- Remote Environment Client
**Promise**: `CrisisWorldClient(EnvClient)` typed for `ActionUnion`/`Observation`/`CrisisState`.
**Depends on**: `models.py` only.

#### `server/` -- CrisisWorld (implements Environment ABC)
**Promise**: `CrisisWorld(Environment)` -- `reset(seed, episode_id) -> Observation`,
`step(action, timeout_s) -> Observation`, `state -> CrisisState`, `get_metadata()`,
`close()`.
**Depended on by**: `inference.py` (wiring only). Never imported by `agents/` or
`cortex/` directly.
**Depends on**: `models.py` only. NOT `schemas/`, `protocols/`, or any other package.

#### `protocols/` -- Contract Layer
**Promise**: Pure `typing.Protocol` classes. Zero implementation.
`EnvProtocol` mirrors OpenEnv `Environment` ABC. `AgentProtocol` supports
two-mode operation (direct + MCP).
**Depended on by**: every other package.
**Depends on**: `models.py`, `schemas/`.

#### `schemas/` -- Agent-Side Data Layer
**Promise**: Immutable Pydantic models for Cortex artifacts, budget, episode traces,
experiment config. Re-exports env types from `models.py` for convenience.
**Depended on by**: `protocols/`, `cortex/`, `agents/`, `evaluation/`, `logging/`.
**Depends on**: `models.py` (for re-exports only).

#### `cortex/` -- Deliberation System
**Promise**: `CortexDeliberator.deliberate(observation, budget) -> (Action, DeliberationLog)`.
Each role implements `RoleProtocol.invoke(input) -> Artifact`.
Budget tracker implements `BudgetProtocol`.
**Depended on by**: `agents/cortex_agent.py` (via protocol).
**Depends on**: `protocols/`, `schemas/`, `models.py`.

#### `agents/` -- Agent Implementations
**Promise**: `AgentProtocol.act(observation) -> ActionUnion`.
**Depended on by**: `evaluation/`, `inference.py`.
**Depends on**: `protocols/`, `schemas/`, `models.py`. CortexAgent receives a
`CortexDeliberator` via constructor injection (protocol-typed).

#### `evaluation/` -- Experiment Runner (OpenEnv Orchestrator)
**Promise**: `ExperimentRunner.run(config) -> ExperimentResults`.
Generates `episode_id` per episode. Uses `env.state` for step tracking.
**Depended on by**: `inference.py`.
**Depends on**: `protocols/`, `schemas/`, `models.py`. Receives env and agent factories.

#### `tracing/` -- Traceability
**Promise**: `LoggerProtocol.record(event) -> None`, `save(path) -> Path`.
Uses `episode_id` from `CrisisState`. Aligns turn numbers with `state.step_count`.
**Depended on by**: `evaluation/`, `agents/`, `cortex/` (via protocol).
**Depends on**: `protocols/`, `schemas/`, `models.py`.

> **Note:** Renamed from `logging/` to avoid shadowing Python's stdlib `logging`.

### Dependency Graph (allowed imports)

```
models.py  ←── server/           (env-contract types, LEAF)
           ←── schemas/          (re-exports + agent types)
           ←── protocols/        (type references)
           ←── cortex/
           ←── agents/
           ←── evaluation/
           ←── tracing/
           ←── inference.py

schemas/   ←── protocols/
           ←── cortex/
           ←── agents/
           ←── evaluation/
           ←── tracing/
```

**Forbidden**: `server/` <-> `cortex/` direct import. `server/` <-> `agents/` direct
import. `server/` imports NOTHING from `schemas/`, `protocols/`, or any project package.
All cross-package communication goes through `protocols/` + `schemas/` + `models.py`.

---

## Absolute Rules

### 1. Protocol-First Design

- Every directory exports a `Protocol` before any implementation exists.
- New functionality starts as a protocol method signature, then schema, then test, then implementation.
- Cross-directory imports MUST go through `protocols/`, `schemas/`, or `models.py`. If you need to import from a sibling package, you are violating the architecture -- inject via the composition root instead.

### 2. Immutable Data

- All schemas are `frozen=True` Pydantic models.
- **Exception**: `CrisisState` uses `ConfigDict(extra="allow")` per OpenEnv `State` base -- NOT frozen, because `step_count` is mutated by the server between steps.
- State transitions in `server/` return new state objects; they do not mutate in place.
- Cortex artifacts are append-only per turn; never overwrite a previous artifact.

### 3. Test-Driven Development

Every feature follows RED -> GREEN -> REFACTOR:

1. **Write the test first.** The test must fail.
2. **Write minimal implementation** to pass the test.
3. **Refactor** without changing behavior.
4. **Coverage gate**: 80% minimum. CI blocks merge below this.

Test organization mirrors project root:
- `tests/unit/` -- single function/class, no I/O, no network.
- `tests/integration/` -- multiple components wired together (env+agent, cortex+budget).
- `tests/e2e/` -- full episode from reset to termination.

Every Cortex role must have:
- A unit test that verifies input->output schema compliance.
- A unit test that verifies budget is charged.
- An integration test within a deliberation loop.

### 4. Typed Artifacts Everywhere

- Every Cortex role invocation produces a typed Pydantic artifact.
- No free-text reasoning. If it isn't a schema-validated artifact, it doesn't exist.
- The deliberation log is a sequence of typed artifacts, not strings.

### 5. Budget Accounting Is Mandatory

- Every role call MUST go through `BudgetProtocol.charge()`.
- A role call that would exceed remaining budget MUST raise `BudgetExhaustedError`.
- Budget state is part of every observation and every log entry.
- Tests must verify budget is decremented on every role invocation.

### 6. Logging Is Not Optional

- Every turn records: observation, each role artifact, executive decision, chosen action, reward, budget snapshot.
- Episode traces are written as structured JSON (one file per episode).
- Traces live in `traces/` (git-ignored). Results live in `results/` (git-ignored).
- If a turn is not logged, it is a bug.
- `episode_id` flows from `env.reset()` through `CrisisState`, logging, and evaluation.

### 7. Seeded Reproducibility

- Every stochastic operation takes an explicit `seed` or `rng` parameter.
- `random.random()` or unseeded numpy calls are forbidden.
- Same seed + same agent config = identical episode trajectory.
- Tests must verify determinism: run twice with same seed, assert identical results.

### 8. Coding Style

- **Python 3.11+**. Type hints on all public function signatures.
- **Pydantic v2** for all data models.
- **`ruff`** for linting and formatting (replaces black + isort + flake8).
- Functions < 50 lines. Files < 400 lines (800 hard max).
- No wildcard imports. No circular imports.
- Prefer explicit over clever. No metaprogramming unless unavoidable.
- Error handling: explicit, never silenced. Raise domain-specific exceptions.
- No print statements. Use `logging` module or structured logger.
- Docstrings only on public protocol methods and non-obvious logic.

### 9. Commit Discipline

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- No commits with failing tests.
- No secrets, API keys, or credentials in source. Use env vars or `.env` (git-ignored).
- `.gitignore` must include: `traces/`, `results/`, `.env`, `__pycache__/`, `*.pyc`.

### 10. Build Sequence (DO NOT SKIP)

Follow this order strictly:

**Phase 0** -- OpenEnv scaffold creation (`models.py`, `client.py`, `openenv.yaml`, `server/app.py`, `Dockerfile`).
**Phase 1** -- Environment skeleton + flat baseline + one E2E episode.
**Phase 2** -- Cortex-lite (Perception + Planner + Executive) + first comparison.
**Phase 3** -- Cortex-full (add World Modeler + Critic + Memory) + ablation run.
**Phase 4** -- Executive tuning, reward sweeps, diagnostics.

> **Hard rule**: Do NOT start Cortex before the flat baseline runs end-to-end.

---

## Commands

### Environment Setup

```bash
# Install with uv
uv sync
uv sync --extra dev --extra server
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only (fast, run frequently)
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# E2E tests (slow, run before commit)
pytest tests/e2e/ -v

# Coverage report (must be >= 80%)
pytest --cov=. --cov-report=term-missing --cov-fail-under=80
```

### Linting & Formatting

```bash
# Check
ruff check . tests/
ruff format --check . tests/

# Fix
ruff check --fix . tests/
ruff format . tests/

# Type checking
mypy .
```

### Running Episodes

```bash
# Single flat-baseline episode
python inference.py --agent flat --seed 42

# Single cortex episode
python inference.py --agent cortex --seed 42

# Full ablation experiment
python inference.py --experiment configs/experiment_ablation.yaml

# View trace
python -m tracing.formatters traces/<episode_id>.json
```

### OpenEnv Server

```bash
# Start CrisisWorld server (HTTP + WebSocket)
uvicorn server.app:app --port 8000

# Docker
docker build -t crisis-world server/
docker run -p 8000:8000 crisis-world
```

### Git Workflow

```bash
# Before every commit
ruff check . tests/ && ruff format --check . tests/ && mypy . && pytest --cov=. --cov-fail-under=80

# Commit format
git commit -m "feat: add resource depletion dynamics"
```

---

## Reward Function Reference

```
R = alpha*R_outcome + beta*R_timeliness - gamma*C_inner_compute - delta*R_safety_violations + epsilon*R_comms_quality
```

Weights are configured in `configs/reward_weights.yaml`. Tuning these is part
of the experimental design, not an afterthought.

---

## Ablation Conditions

| Condition    | Roles                          | Compute  | Memory | Critic |
|-------------|-------------------------------|----------|--------|--------|
| Flat-lite    | single policy                 | low      | no     | no     |
| Flat-fat     | single policy                 | matched  | no     | no     |
| Cortex-lite  | Perception+Planner+Executive  | low      | no     | no     |
| Cortex-full  | all 5 roles                   | matched  | yes    | yes    |
| Cortex-tuned | all 5 + learned Executive     | matched  | yes    | yes    |

Primary comparison: **Flat-fat vs Cortex-full** (same compute, different organization).

---

## Open Decisions (Resolve During Implementation)

1. **Budget units**: start with role-call count, add token accounting later.
2. **Memory model**: start with keyed append-only log.
3. **Critic scope**: evaluate one plan at a time (not comparative).
4. **Stakeholder signals**: start aggregated, separate later.
5. **Communicator role**: defer to post-MVP.
6. **Executive training**: prompt-driven first, RL only if time permits.
7. **`StepResult` removed**: `Observation` carries `done`, `reward`, `metadata` per OpenEnv.
8. **`CrisisState` not frozen**: sole exception to Rule 2 (mutable `step_count`).
9. **`episode_id` flow**: generated by runner, passed to `reset()`, stored in `CrisisState`, used by logger.
