# CrisisWorld + Cortex

An outbreak-control simulation environment paired with a budgeted structured reasoning agent.

**Core question:** Does budgeted structured reasoning (Cortex) beat a flat heuristic policy under matched compute?

## How It Works (Simple Version)

Think of this as a game:

1. **The World (CrisisWorld)** simulates a disease outbreak across multiple regions. Each turn, the disease spreads, people recover or die, and resources decay.
2. **The Agent** (either Flat or Cortex) looks at the current situation and picks an action -- deploy medical resources, restrict movement, request data, etc.
3. **The Loop** runs for up to 50 turns. The episode ends early if the outbreak is contained or becomes catastrophic.
4. **The Score** is a composite reward measuring outcome quality, timeliness, safety, and communication.

The experiment compares two agent architectures:
- **Flat Agent** -- Simple priority rules, no deliberation
- **Cortex Agent** -- Multi-role structured reasoning (Perception, World Modeler, Planner, Critic, Executive) with a thinking budget

## Setup

```bash
# Clone and enter project
cd MetaFinals

# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,server]"

# Or with uv
uv sync --extra dev --extra server
```

## Running

```bash
# Single flat-agent episode
python inference.py --agent flat --seed 42

# Single cortex-agent episode
python inference.py --agent cortex --seed 42

# Full ablation experiment (all 5 conditions)
python inference.py --experiment configs/experiment_ablation.yaml
```

Results are saved to `results/comparison.md`. Traces go to `traces/`.

## Running Tests

```bash
# All tests (should see 271 passed)
pytest tests/ -v

# Just unit tests (fast)
pytest tests/unit/ -v

# Just integration tests
pytest tests/integration/ -v

# Just E2E tests (full episodes)
pytest tests/e2e/ -v

# With coverage
pytest --cov=. --cov-report=term-missing
```

---

## What the Agent Sees: Observation

Every turn, the agent receives an `Observation` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `turn` | `int` | Current turn number (0-indexed) |
| `regions` | `tuple[RegionState, ...]` | Per-region epidemic state (noisy!) |
| `stakeholder_signals` | `tuple[StakeholderSignal, ...]` | Hospital/media/gov/public reports |
| `incidents` | `tuple[IncidentReport, ...]` | High-severity region alerts |
| `telemetry` | `Telemetry` | Aggregated counts (lagged 1 turn) |
| `resources` | `ResourcePool` | Available medical/personnel/funding |
| `active_constraints` | `tuple[Constraint, ...]` | Active policy rules |
| `budget_status` | `BudgetStatus` | Cortex thinking budget remaining |
| `done` | `bool` | True if episode is over |
| `reward` | `float or None` | Composite reward for this step (None on reset) |
| `metadata` | `dict` | Episode context: episode_id, step_count, termination_reason |

### Sub-models in Observation

**RegionState** (per region, values are noisy estimates):
| Field | Description |
|-------|-------------|
| `region_id` | e.g. "r0", "r1" |
| `population` | Total population (exact) |
| `infected` | Current infected count (noisy) |
| `recovered` | Cumulative recovered (noisy) |
| `deceased` | Cumulative deceased (noisy) |
| `restricted` | Whether movement is restricted |

**ResourcePool**:
| Field | Description |
|-------|-------------|
| `medical` | Medical supplies (decays 2%/turn) |
| `personnel` | Personnel count (stable) |
| `funding` | Funding units (decays 1%/turn) |

**Telemetry** (aggregated, lagged 1 turn):
| Field | Description |
|-------|-------------|
| `total_infected` | Sum of infected across all regions |
| `total_recovered` | Sum of recovered |
| `total_deceased` | Sum of deceased |
| `data_staleness` | Turns since last fresh data |

**StakeholderSignal** (4 sources per turn):
| Field | Description |
|-------|-------------|
| `source` | "hospital", "media", "government", "public" |
| `urgency` | 0.0 to 1.0 (higher = more urgent) |
| `message` | Human-readable description |

---

## What the Agent Can Do: Actions

The agent picks exactly one action per turn:

| Action | Parameters | Delay | Description |
|--------|-----------|-------|-------------|
| `DeployResource` | resource, region_id, amount | 2 turns | Send medical/personnel/funding to a region |
| `RestrictMovement` | region_id, level (1-3) | 1 turn | Restrict travel in a region |
| `RequestData` | source | 1 turn | Ask for better data from a source |
| `PublicCommunication` | audience, message | 0 turns | Issue a public statement |
| `Escalate` | agency | 2 turns | Call in external authority |
| `ReallocateBudget` | from_category, to_category, amount | 0 turns | Shift budget between categories |
| `NoOp` | (none) | 0 turns | Do nothing this turn |

**Note:** Actions with delay > 0 take effect on a future turn, not immediately.

---

## How CrisisWorld Steps (The 12-Step Pipeline)

Each `env.step(action)` does this internally:

1. Guard checks (initialized? not closed? not done?)
2. Increment turn counter
3. Validate action + check policy constraints -> violations list
4. Schedule delayed effects (e.g. DeployResource arrives in 2 turns)
5. Apply effects that are due this turn
6. Advance SIR epidemiological model (disease spreads between regions)
7. Apply per-turn resource decay (medical -2%, funding -1%)
8. Check termination conditions (5 possible endings, see below)
9. Compute composite reward
10. Record action in history
11. Generate stakeholder signals + assemble noisy observation
12. Return Observation with done/reward/metadata set

## How the Episode Ends

Checked in priority order (first match wins):

| Condition | Trigger | Bonus |
|-----------|---------|-------|
| **Contained** | All regions < 1% infected AND turn >= 5 | +2.0 |
| **Catastrophic mortality** | Deceased > 5% of total population | -3.0 |
| **Catastrophic spread** | Infected > 60% of total population | -3.0 |
| **Resource collapse** | All resources < 5% of initial | 0 |
| **Max turns** | Turn >= max_turns (default 50) | -1.0 |

---

## How Cortex Thinks (The Deliberation Loop)

When using the Cortex agent, each `act()` call triggers a structured deliberation:

1. **Perception** (cost: 1) -- Cleans noisy observation, detects anomalies (spikes, contradictions)
2. **Executive** evaluates what info is missing and decides:
   - "call world_modeler" -- Estimate hidden variables, forecast trajectories
   - "call planner" -- Generate ranked candidate actions
   - "call critic" -- Analyze failure modes and risk
   - "act" -- Pick the best candidate action and commit
   - "escalate" -- Call external authority
   - "wait"/"stop" -- Return NoOp

The Executive loops until it decides to act, runs out of budget, or hits 20 iterations.

**Budget:** Every role call costs budget units. When budget is exhausted, the agent is forced to act immediately with whatever plan it has (or NoOp).

---

## Ablation Conditions

| Condition | Agent | Budget | Roles | Memory | Critic |
|-----------|-------|--------|-------|--------|--------|
| flat-lite | Flat | 10 | none | no | no |
| flat-fat | Flat | 50 | none | no | no |
| cortex-lite | Cortex | 10 | perception+planner+executive | no | no |
| cortex-full | Cortex | 50 | all 5 | yes | yes |
| cortex-tuned | Cortex | 50 | all 5 + tuned | yes | yes |

Primary comparison: **flat-fat vs cortex-full** (same compute budget, different organization).

---

## Manual Test Case: Step-by-Step Walkthrough

Run this to see exactly what happens at each step:

```python
from models import EnvConfig, NoOp, DeployResource
from server import CrisisWorld

env = CrisisWorld(config=EnvConfig(max_turns=5, num_regions=4))
obs = env.reset(seed=42, episode_id="manual-test")
```

### Turn 0 (Reset)

```
turn=0, done=False, reward=None
episode_id=manual-test
  r0: pop=63387 inf=365 rec=0 dec=0 restricted=False  <-- origin region
  r1: pop=166094 inf=0 rec=0 dec=0 restricted=False
  r2: pop=148186 inf=0 rec=0 dec=0 restricted=False
  r3: pop=115832 inf=0 rec=0 dec=0 restricted=False
resources: medical=161 personnel=122 funding=387
state.step_count=0
```

What to check:
- Only r0 has infected (it's the origin)
- `done=False`, `reward=None` (no step taken yet)
- Resources are randomly generated but deterministic for seed=42

### Turn 1 (NoOp)

```python
obs = env.step(NoOp())
```

```
turn=1, done=False, reward=0.9990
  r0: pop=63387 inf=417 rec=45 dec=12   <-- disease spreading, some recovery/death
  r1: pop=166094 inf=10 rec=0 dec=0     <-- inter-region spillover!
  r2: pop=148186 inf=6 rec=0 dec=0      <-- spillover here too
  r3: pop=115832 inf=0 rec=0 dec=0
resources: medical=157 personnel=122 funding=383  <-- decay applied
state.step_count=1
```

What to check:
- r0 infected grew (SIR model), some recovered (45), some died (12)
- r1 and r2 now have infected (inter-region spread)
- Resources decayed (medical: 161->157, funding: 387->383)
- Reward is ~1.0 (mostly healthy population)

### Turn 2 (NoOp)

```python
obs = env.step(NoOp())
```

```
turn=2, done=False, reward=0.9988
  r0: inf=504 rec=128 dec=22
  r1: inf=17 rec=4 dec=0
  r2: inf=16 rec=1 dec=0
  r3: inf=0 rec=0 dec=0
```

What to check:
- Infection continues growing in r0
- Spillover accumulating in r1, r2
- r3 still clean

### Turn 3 (DeployResource to r0)

```python
obs = env.step(DeployResource(resource="medical", region_id="r0", amount=10))
```

```
turn=3, done=False, reward=0.9986
  r0: inf=496 rec=182 dec=37
  r1: inf=22 rec=7 dec=1       <-- first death in r1
  r2: inf=20 rec=3 dec=0
  r3: inf=1 rec=0 dec=0        <-- r3 now has its first case
```

What to check:
- Deploy has a 2-turn delay, so resources don't arrive yet
- Disease still spreading (r3 now infected)
- Recovery accumulating in r0

### Turn 4 (NoOp)

```
turn=4, done=False, reward=0.9983
  r0: inf=538 rec=185 dec=48
  r1: inf=34 rec=9 dec=1
  r2: inf=46 rec=5 dec=0       <-- r2 infection accelerating
  r3: inf=1 rec=1 dec=0
```

### Turn 5 (NoOp) -- Episode Ends

```
turn=5, done=True, reward=-0.0019
termination_reason=max_turns
  r0: inf=694 rec=339 dec=44
  r1: inf=39 rec=11 dec=3
  r2: inf=64 rec=9 dec=1
  r3: inf=1 rec=2 dec=0
```

What to check:
- `done=True` -- max_turns reached
- `reward=-0.0019` -- negative because max_turns penalty (-1.0)
- r0 infection still growing (no containment achieved)
- Calling `env.step()` again would raise `RuntimeError("Episode is done")`

### Verify Determinism

Run the exact same sequence with seed=42 again -- you should get identical numbers at every step. That's the seeded reproducibility guarantee.

---

## Project Structure

```
MetaFinals/
  inference.py          -- Entry point (composition root)
  models.py             -- Env-contract types (extends OpenEnv)
  client.py             -- Remote env client
  openenv.yaml          -- OpenEnv config

  server/               -- CrisisWorld environment (SIR model)
  protocols/            -- Contract interfaces (6 protocols)
  schemas/              -- Agent-side data models
  cortex/               -- Structured deliberation system
    roles/              -- 5 specialist roles
  agents/               -- FlatAgent + CortexAgent
  evaluation/           -- Experiment runner + metrics
  tracing/              -- Episode trace recording

  tests/
    unit/               -- Per-module tests
    integration/        -- Multi-component tests
    e2e/                -- Full episode tests
  configs/              -- YAML experiment configs
  traces/               -- Episode trace output (git-ignored)
  results/              -- Experiment results (git-ignored)
```

## Reward Formula

```
R = alpha * R_outcome
  + beta  * R_timeliness
  - gamma * C_inner_compute
  - delta * R_safety_violations
  + epsilon * R_comms_quality
  + terminal_bonus
```

Where `R_outcome = 1.0 - 2.0 * mortality_rate - infection_rate` (clamped to [-1, 1]).
