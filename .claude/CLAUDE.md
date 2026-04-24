# CrisisWorld + Cortex

Outbreak-control environment and structured deliberation agent.
Central question: does budgeted structured reasoning (Cortex) beat a flat policy under matched compute?

## High-Level Architecture

Two tightly coupled systems connected through an OpenEnv-compatible step loop:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Episode Runner                               │
│  for turn in episode:                                               │
│    obs          = env.step(action)          # CrisisWorld advances  │
│    action, log  = agent.act(obs)            # Agent decides         │
│    traces.record(obs, action, log)          # Everything is logged  │
└─────────────────────────────────────────────────────────────────────┘
         │                           │
         ▼                           ▼
┌─────────────────┐       ┌───────────────────────────────────┐
│   CrisisWorld   │       │         Agent (swappable)         │
│   (outer env)   │       │                                   │
│                 │       │  ┌─────────┐   ┌───────────────┐  │
│  hidden epi     │       │  │  Flat   │   │    Cortex     │  │
│  state, noisy   │       │  │ Agent   │   │    Agent      │  │
│  observations,  │       │  │ (base-  │   │  ┌─────────┐  │  │
│  resources,     │       │  │  line)  │   │  │Executive│  │  │
│  stakeholders,  │       │  └─────────┘   │  │Planner  │  │  │
│  constraints,   │       │                │  │Critic   │  │  │
│  delayed fx     │       │                │  │Modeler  │  │  │
│                 │       │                │  │Percep.  │  │  │
│                 │       │                │  │Budget   │  │  │
│                 │       │                │  └─────────┘  │  │
└─────────────────┘       └───────────────────────────────────┘
```

### Outer loop — CrisisWorld

Stateful outbreak simulator. Maintains hidden epidemiological state, resources,
stakeholders, policy constraints, and delayed action effects. Emits partial,
noisy, lagged observations. Computes composite reward.

### Inner loop — Cortex

Structured deliberation system with five roles (Perception, World Modeler,
Planner, Critic, Executive). Each role invocation has a typed input, typed
output artifact, and a budget cost. The Executive decides whether to think
more or act. Thinking is explicitly expensive.

### Agents

Swappable policies behind a single `AgentProtocol`. Flat agents skip Cortex.
Cortex agents delegate to the deliberation loop. Same interface, different
compute organization — this is what the experiment isolates.

### Evaluation

Multi-seed runner that executes episodes under matched-compute conditions,
collects primary/secondary/diagnostic metrics, and produces ablation
comparison tables.

---

## Directory Structure — Function Signature Promises

Each directory exports a **protocol** (its "promise") and hides implementation.
Other directories depend ONLY on protocols and schemas, never on concrete
internals. Wiring happens in `main.py` / the composition root.

```
MetaFinals/
├── CLAUDE.md
├── project.md
├── pyproject.toml
├── main.py                    # Composition root — wires concrete impls
│
├── src/
│   ├── __init__.py
│   │
│   ├── protocols/             # ── CONTRACT LAYER (no implementation) ──
│   │   ├── __init__.py
│   │   ├── env.py             # EnvProtocol
│   │   ├── agent.py           # AgentProtocol
│   │   ├── role.py            # RoleProtocol
│   │   ├── budget.py          # BudgetProtocol
│   │   ├── memory.py          # MemoryProtocol
│   │   └── logger.py          # LoggerProtocol
│   │
│   ├── schemas/               # ── DATA LAYER (Pydantic models, no logic) ──
│   │   ├── __init__.py
│   │   ├── observation.py     # Observation, IncidentReport, Telemetry
│   │   ├── action.py          # OuterAction variants (deploy, restrict, ...)
│   │   ├── reward.py          # RewardComponents, CompositeReward
│   │   ├── state.py           # RegionState, ResourcePool, Constraint
│   │   ├── artifact.py        # CleanState, BeliefState, Plan, Critique
│   │   ├── budget.py          # BudgetStatus, BudgetLedger
│   │   ├── config.py          # EnvConfig, CortexConfig, ExperimentConfig
│   │   └── episode.py         # EpisodeResult, EpisodeTrace
│   │
│   ├── env/                   # ── CRISISWORLD (implements EnvProtocol) ──
│   │   ├── __init__.py        # Exports: CrisisWorld
│   │   ├── world.py           # CrisisWorld class — reset(), step()
│   │   ├── dynamics.py        # Epidemiological model, spread, recovery
│   │   ├── regions.py         # Grid/region state management
│   │   ├── resources.py       # Resource pool tracking, depletion
│   │   ├── stakeholders.py    # Signal generation (hospitals, media, etc.)
│   │   ├── constraints.py     # Policy/legal constraint enforcement
│   │   ├── actions.py         # Action validation + delayed-effect scheduling
│   │   ├── observations.py    # Partial observation assembly, noise, lag
│   │   ├── rewards.py         # Composite reward computation
│   │   ├── scenarios.py       # Seeded scenario generation
│   │   └── termination.py     # Episode end conditions
│   │
│   ├── cortex/                # ── CORTEX (implements RoleProtocol per role) ──
│   │   ├── __init__.py        # Exports: CortexDeliberator
│   │   ├── deliberator.py     # Orchestrates the inner deliberation loop
│   │   ├── budget.py          # Budget accounting (implements BudgetProtocol)
│   │   ├── memory.py          # Episode memory store (implements MemoryProtocol)
│   │   └── roles/
│   │       ├── __init__.py
│   │       ├── perception.py  # Raw obs → CleanState + anomalies
│   │       ├── world_modeler.py # CleanState → BeliefState + forecasts
│   │       ├── planner.py     # BeliefState → candidate Plans
│   │       ├── critic.py      # Plan → failure modes + risk score
│   │       └── executive.py   # All artifacts → act|call|wait|escalate|stop
│   │
│   ├── agents/                # ── AGENTS (implement AgentProtocol) ──
│   │   ├── __init__.py        # Exports: FlatAgent, CortexAgent
│   │   ├── flat.py            # Direct policy, no deliberation
│   │   └── cortex_agent.py    # Delegates to CortexDeliberator
│   │
│   ├── evaluation/            # ── EVALUATION (experiment orchestration) ──
│   │   ├── __init__.py        # Exports: ExperimentRunner
│   │   ├── runner.py          # Multi-seed episode execution
│   │   ├── metrics.py         # Metric collection and aggregation
│   │   ├── ablations.py       # Condition setup (flat-lite/fat, cortex-lite/full)
│   │   └── analysis.py        # Comparison tables, diagnostics
│   │
│   └── logging/               # ── LOGGING (implements LoggerProtocol) ──
│       ├── __init__.py        # Exports: EpisodeLogger
│       ├── tracer.py          # Per-turn event recording
│       ├── serializer.py      # Trace → JSON/file output
│       └── formatters.py      # Human-readable trace rendering
│
├── tests/
│   ├── unit/
│   │   ├── test_dynamics.py
│   │   ├── test_regions.py
│   │   ├── test_resources.py
│   │   ├── test_rewards.py
│   │   ├── test_budget.py
│   │   ├── test_roles.py
│   │   └── test_schemas.py
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
├── traces/                    # Git-ignored. Episode trace output.
└── results/                   # Git-ignored. Experiment results.
```

### Directory Responsibilities & Inter-Relations

#### `src/protocols/` — Contract Layer
**Promise**: Pure `typing.Protocol` classes. Zero implementation, zero imports
beyond `schemas/`.
**Depended on by**: every other package.
**Depends on**: `schemas/` only.

#### `src/schemas/` — Data Layer
**Promise**: Immutable Pydantic models. No behavior, no side effects. Every
data structure that crosses a directory boundary is defined here.
**Depended on by**: every other package.
**Depends on**: nothing (leaf node).

#### `src/env/` — CrisisWorld
**Promise**: `EnvProtocol` — `reset(seed) -> Observation`,
`step(action) -> StepResult`, `close() -> None`.
**Depended on by**: `main.py` (wiring only). Never imported by `agents/` or
`cortex/` directly.
**Depends on**: `protocols/`, `schemas/`.

#### `src/cortex/` — Deliberation System
**Promise**: `CortexDeliberator.deliberate(observation, budget) -> (Action, DeliberationLog)`.
Each role implements `RoleProtocol.invoke(input) -> Artifact`.
Budget tracker implements `BudgetProtocol`.
**Depended on by**: `agents/cortex_agent.py` (via protocol).
**Depends on**: `protocols/`, `schemas/`.

#### `src/agents/` — Agent Implementations
**Promise**: `AgentProtocol.act(observation) -> Action`.
**Depended on by**: `evaluation/`, `main.py`.
**Depends on**: `protocols/`, `schemas/`. CortexAgent receives a
`CortexDeliberator` via constructor injection (protocol-typed).

#### `src/evaluation/` — Experiment Runner
**Promise**: `ExperimentRunner.run(config) -> ExperimentResults`.
**Depended on by**: `main.py`.
**Depends on**: `protocols/`, `schemas/`. Receives env and agent factories.

#### `src/logging/` — Traceability
**Promise**: `LoggerProtocol.record(event) -> None`, `save(path) -> Path`.
**Depended on by**: `evaluation/`, `agents/`, `cortex/` (via protocol).
**Depends on**: `protocols/`, `schemas/`.

### Dependency Graph (allowed imports)

```
schemas/  ←── protocols/  ←── env/
                          ←── cortex/
                          ←── agents/
                          ←── evaluation/
                          ←── logging/
                          ←── main.py (wires everything)
```

**Forbidden**: `env/` ↔ `cortex/` direct import. `env/` ↔ `agents/` direct
import. All cross-package communication goes through `protocols/` + `schemas/`.

---

## Absolute Rules

### 1. Protocol-First Design

- Every directory exports a `Protocol` before any implementation exists.
- New functionality starts as a protocol method signature, then schema, then test, then implementation.
- Cross-directory imports MUST go through `protocols/` or `schemas/`. If you need to import from a sibling package, you are violating the architecture — inject via the composition root instead.

### 2. Immutable Data

- All schemas are `frozen=True` Pydantic models.
- State transitions in `env/` return new state objects; they do not mutate in place.
- Cortex artifacts are append-only per turn; never overwrite a previous artifact.

### 3. Test-Driven Development

Every feature follows RED → GREEN → REFACTOR:

1. **Write the test first.** The test must fail.
2. **Write minimal implementation** to pass the test.
3. **Refactor** without changing behavior.
4. **Coverage gate**: 80% minimum. CI blocks merge below this.

Test organization mirrors `src/`:
- `tests/unit/` — single function/class, no I/O, no network.
- `tests/integration/` — multiple components wired together (env+agent, cortex+budget).
- `tests/e2e/` — full episode from reset to termination.

Every Cortex role must have:
- A unit test that verifies input→output schema compliance.
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

**Phase 1** — Environment skeleton + flat baseline + one E2E episode.
**Phase 2** — Cortex-lite (Perception + Planner + Executive) + first comparison.
**Phase 3** — Cortex-full (add World Modeler + Critic + Memory) + ablation run.
**Phase 4** — Executive tuning, reward sweeps, diagnostics.

> **Hard rule**: Do NOT start Cortex before the flat baseline runs end-to-end.

---

## Commands

### Environment Setup

```bash
# Create virtual environment and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
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
pytest --cov=src --cov-report=term-missing --cov-fail-under=80
```

### Linting & Formatting

```bash
# Check
ruff check src/ tests/
ruff format --check src/ tests/

# Fix
ruff check --fix src/ tests/
ruff format src/ tests/

# Type checking
mypy src/
```

### Running Episodes

```bash
# Single flat-baseline episode
python main.py --agent flat --seed 42

# Single cortex episode
python main.py --agent cortex --seed 42

# Full ablation experiment
python main.py --experiment configs/experiment_ablation.yaml

# View trace
python -m src.logging.formatters traces/<episode_id>.json
```

### Git Workflow

```bash
# Before every commit
ruff check src/ tests/ && ruff format --check src/ tests/ && mypy src/ && pytest --cov=src --cov-fail-under=80

# Commit format
git commit -m "feat: add resource depletion dynamics"
```

---

## Reward Function Reference

```
R = α·R_outcome + β·R_timeliness - γ·C_inner_compute - δ·R_safety_violations + ε·R_comms_quality
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
