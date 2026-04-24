# CrisisWorld + Cortex — Project Specification

*Source basis: adapted from the uploaded `CrisisWorld + Cortex` team working document, narrowed strictly to the outbreak-control project scope and stripped of pitch-only material.*

## 1. Project Overview

This project defines a high-stakes outbreak-control environment and an agent architecture designed to operate inside it under explicit reasoning costs.

The project has two tightly coupled parts:

- **CrisisWorld**: the **outer environment**, a stateful outbreak-control simulator built as an OpenEnv environment.
- **Cortex**: the **inner deliberation system**, a structured set of specialist reasoning roles that decide how the agent spends limited deliberation budget before taking actions in CrisisWorld.

The project is not about general AI claims, free-form multi-agent chat, or a broad crisis benchmark covering many domains. It is specifically about **outbreak control** under:

- partial observability,
- delayed and noisy information,
- real resource constraints,
- policy and safety constraints,
- delayed consequences, and
- limited thinking budget.

The central project question is:

> Given the same overall model budget, does a structured, budgeted inner reasoning system control an outbreak better than a flat single-agent policy?

This question is tested in a stateful environment where wrong decisions can compound over time and where over-deliberation is also costly.

---

## 2. Project Scope

### In scope

This project includes:

- one **outbreak-control** scenario family,
- one OpenEnv-compatible environment server,
- one flat-agent baseline,
- one structured Cortex agent,
- matched-compute evaluation between the two,
- per-episode logging and traceability,
- composite reward and diagnostics,
- ablation-ready experiment setup.

### Out of scope

This project does **not** include:

- disaster logistics,
- hospital surge as a separate environment family,
- misinformation-only benchmarks,
- benchmarking different foundation models against one another,
- broad “AGI” claims,
- unrestricted multi-agent free-text discussion,
- pitch framing, slide content, or storytelling artifacts.

This document concerns the **project itself**, not extra pitch material.

---

## 3. Why an Environment Is Required

A one-shot prompt chain cannot properly test the intended research question. The project requires a **stateful environment** because outbreak management is a sequential control problem with hidden state, delayed feedback, and policy trade-offs.

The environment form is necessary because it provides:

1. **Persistent hidden state**  
   The true outbreak state exists inside the simulator, not inside the prompt. The agent must infer infection spread from imperfect signals.

2. **Structured valid actions**  
   The agent must choose from a typed action space. It cannot hand-wave with natural-language plans that bypass operational constraints.

3. **Delayed consequences**  
   A correct or incorrect decision may only reveal its effects several turns later.

4. **Matched-compute comparison**  
   The same scenario seed and same total compute budget can be used for both the flat baseline and Cortex, making organization of reasoning the main experimental variable.

This makes CrisisWorld a true sequential decision environment rather than a prompt-formatting exercise.

---

## 4. High-Level Architecture

The project has a recursive structure:

### 4.1 Outer world: CrisisWorld

CrisisWorld is the operational outbreak simulator. It contains:

- hidden epidemiological state,
- observed reports and telemetry,
- stakeholders,
- resources,
- policy constraints,
- delayed effects of previous actions,
- turn-based progression of simulated time.

### 4.2 Inner world: Cortex

Cortex is the agent’s internal decision organization. Instead of one monolithic reasoning pass, it routes thinking through specialist roles. Each role invocation consumes part of a fixed deliberation budget.

### 4.3 Core interaction loop

At every outer step:

1. CrisisWorld emits a partial observation.
2. Perception processes that observation into a cleaner internal state summary.
3. The Executive decides whether to:
   - act immediately,
   - call another specialist role,
   - wait,
   - escalate, or
   - stop.
4. If specialists are called, they produce typed artifacts.
5. The Executive eventually commits an outer action.
6. CrisisWorld advances its hidden state and returns the next observation and reward signals.

The defining property is that **thinking itself has cost**. Cortex is useful only if it improves control enough to justify its budget consumption.

---

## 5. Outer Environment: CrisisWorld

### 5.1 Objective of CrisisWorld

CrisisWorld simulates outbreak control in a partially observable operational setting. The agent must manage spread, preserve system stability, allocate scarce resources, and avoid unsafe or illegal interventions.

The environment should be legible enough to analyze, but rich enough that poor reasoning causes cascading failures.

### 5.2 Outbreak-control framing

The chosen scenario family for this project is **Outbreak Control**.

This is the recommended scenario because it provides:

- clear dynamics,
- high stakes,
- natural partial observability,
- time-sensitive actions,
- meaningful resource trade-offs,
- interpretable reward design,
- straightforward reproducibility through seeded simulation.

The implementation should behave like an operational command problem, not a public-health essay. The focus is on control decisions under uncertainty.

### 5.3 State maintained by the environment

At minimum, CrisisWorld maintains the following underlying state:

### Population / region state

A small grid or region set where each cell or region tracks:

- population-related status,
- infection-related status,
- health-system strain,
- sentiment or public reaction,
- local resource condition.

### Resource pools

Global or regional resources such as:

- medical staff,
- supplies,
- testing capacity,
- budget.

### Incidents

Active operational events such as:

- local outbreak surges,
- shortages,
- contradictory reports,
- secondary disruption signals.

### Time

The environment tracks:

- turn number,
- simulated day/time,
- pending delayed effects scheduled from prior actions.

### Stakeholders

The environment exposes noisy signals from actors such as:

- hospitals,
- agencies,
- media,
- public channels.

These signals are not guaranteed to be fully correct or consistent.

### Constraints

The environment tracks policy or legal constraints that may be active at a given moment. Certain interventions may be valid operationally but still incur safety or governance penalties.

### Action history with pending consequences

Past actions must remain part of state because their effects may unfold later. For example, a resource deployment may reduce future damage, while a poor restriction decision may later amplify instability.

---

## 6. Observation Space

The agent never receives full ground truth. Each call to `step()` should return a structured observation bundle containing at least:

- current turn,
- current simulated time,
- incident reports,
- telemetry,
- current known resources,
- stakeholder signals,
- active constraints,
- recent actions and pending effects,
- remaining outer and inner budget.

A representative structure is:

```json
{
  "turn": 12,
  "simulated_time": "Day 6, 14:00",
  "incident_reports": [...],
  "telemetry": {...},
  "resources": {...},
  "stakeholder_signals": [...],
  "active_constraints": [...],
  "recent_actions": [...],
  "budget_remaining": {
    "outer_time_turns": 38,
    "inner_deliberation_tokens": 4200
  }
}
```

### Observation design principles

The observation space must intentionally enforce the following:

- **partial observability**: the true infection map is hidden,
- **noise**: reports can be contradictory,
- **lag**: telemetry may arrive one to three turns late,
- **known precision where appropriate**: resources may be known exactly even when outbreak state is not,
- **decision relevance**: observations should directly support planning, querying, escalation, and communication decisions.

---

## 7. Outer Action Space

The action space should remain small, typed, and legible.

### Required actions

- `deploy_resource(kind, region, amount)`  
  Moves operational resources into a target region.

- `restrict_movement(region, level)`  
  Applies movement restrictions with both epidemiological effect and governance cost.

- `request_data(source)`  
  Pays a cost to obtain additional information. This is crucial for handling uncertainty.

- `public_communication(audience, message_spec)`  
  Issues a public-facing message with downstream effects on trust, sentiment, or compliance.

- `escalate(agency)`  
  Hands off or requests higher-level intervention.

- `reallocate_budget(from, to, amount)`  
  Shifts scarce budget across operational priorities.

- `no_op`  
  Advances time without taking a direct intervention.

### Action-space principles

- Actions must be schema-valid and constraint-checked.
- The space should be expressive enough to support meaningful policy trade-offs.
- The space should not be bloated with low-value action variants.
- Actions should create delayed and context-dependent effects rather than immediate trivial rewards.

---

## 8. Reward Design

The project uses a **composite reward** with both end-of-episode and shaped per-turn components.

A representative form is:

```text
R = α·R_outcome
  + β·R_timeliness
  - γ·C_inner_compute
  - δ·R_safety_violations
  + ε·R_comms_quality
```

### Reward components

#### 8.1 Outcome reward

Rewards the actual control quality achieved, such as:

- lives saved,
- damage averted,
- outbreak stabilization,
- system stability preserved.

#### 8.2 Timeliness reward

Penalizes delay when action should have been taken sooner. This prevents the agent from always delaying until certainty is perfect.

#### 8.3 Inner-compute cost

Charges Cortex for role calls, token use, or both. This is the critical term that forces the compute-allocation question.

#### 8.4 Safety / policy violation penalty

Penalizes governance failures such as:

- illegal action selection,
- policy breaches,
- fairness failures,
- unsafe interventions.

#### 8.5 Communication quality reward

Rewards public communication quality when public-facing actions are taken.

### Reward design importance

The compute-cost term is the main experimental lever:

- If it is too low, the structured system can think excessively without consequence.
- If it is too high, even a weaker flat policy may dominate by acting quickly.

Therefore, tuning reward weights is part of the project’s empirical design, not an afterthought.

---

## 9. Episode Termination

An episode ends when one of the following occurs:

- the crisis is stabilized,
- a catastrophic threshold is crossed,
- time expires,
- resources collapse,
- governance confidence falls below threshold.

Termination should make success and failure unambiguous while preserving enough horizon for strategic planning to matter.

---

## 10. Inner Deliberation System: Cortex

### 10.1 Purpose

Cortex is the structured inner system that determines how reasoning budget is spent before committing to outer actions.

Its purpose is not to imitate a human brain. Its purpose is to make reasoning organization:

- explicit,
- measurable,
- budgeted,
- logged,
- comparable against a flat baseline.

### 10.2 Design rule for every role

Every role included in Cortex must have all four of the following:

1. a measurable function,
2. a typed input/output schema,
3. a budget cost per invocation,
4. a visible artifact recorded in the deliberation log.

If a role does not satisfy all four, it should not exist in the MVP.

---

## 11. Cortex Roles

The MVP Cortex configuration uses five roles.

### 11.1 Perception

### Input
- raw outer observation bundle

### Output
- `clean_state`
- `salient_changes_since_last_turn`
- `flagged_anomalies`

### Function
Perception normalizes the raw observation into a cleaner internal representation and detects anomalies or inconsistencies that may justify deeper reasoning.

### Budget profile
- automatic once per turn
- small cost

Perception should be cheap and reliable because it is the default entry point into every decision cycle.

### 11.2 World Modeler

### Input
- clean state
- memory digest

### Output
- `belief_state`
- `hidden_var_estimates`
- `three_forecast_trajectories`
- `confidence`

### Function
World Modeler infers the likely hidden outbreak state and proposes short-horizon future trajectories under uncertainty.

### Budget profile
- medium cost
- callable one to three times per turn if needed

This role is especially important when reports conflict or telemetry is stale.

### 11.3 Planner

### Input
- belief state
- goals
- constraints
- memory

### Output
A small set of candidate plans, each with:

- `action`
- `rationale`
- `expected_effect`
- `confidence`

### Function
Planner translates beliefs and constraints into candidate operational actions.

### Budget profile
- medium cost
- invoked on Executive request

Planner should return a small, legible candidate set rather than a large search tree.

### 11.4 Critic

### Input
- one proposed plan

### Output
- `failure_modes`
- `risk_score`
- `policy_violations`
- `recommended_amendments`

### Function
Critic evaluates serious candidate plans and identifies why a plan might fail, be unsafe, or violate policy.

### Budget profile
- small to medium cost
- used on plans that the Executive is genuinely considering

Critic exists to reduce catastrophic mistakes, not to create open-ended debate.

### 11.5 Executive

### Input
- all current deliberation artifacts
- remaining budget

### Output
One of:

- `act(plan)`
- `call(role, query)`
- `wait`
- `escalate`
- `stop`

### Function
Executive is the final arbiter. It decides whether current uncertainty justifies more reasoning or whether the system should commit.

### Budget profile
- small cost
- always runs at the end of each inner cycle

The Executive is the role where compute allocation becomes policy.

---

## 12. Optional Extensions

The following roles are optional and should be considered post-MVP unless implementation time allows:

### Memory / Historian
A passive store can later evolve into a role that surfaces precedents, trends, and summaries.

### Communicator
A specialized role for drafting `public_communication` outputs and improving communication-quality reward.

These are useful, but they are secondary to getting the 5-role system stable and evaluable.

---

## 13. Inner Action Space

The Executive operates over the following inner action choices:

- `act(outer_action)`
- `call(role, query)`
- `wait`
- `escalate`
- `stop`

### Interpretation

- `act` commits to the outer environment.
- `call` spends deliberation budget to obtain another typed artifact.
- `wait` advances time without immediate intervention.
- `escalate` triggers a higher-authority handling path in the outer world.
- `stop` terminates early, usually with penalty or cost.

This makes the inner loop a decision process over reasoning allocation, not just reasoning content.

---

## 14. Budget Mechanics

Budgeting is the core mechanism that makes Cortex meaningful.

Each episode starts with a fixed deliberation budget, measured in one or more of:

- role-call count,
- token budget,
- combined budget.

Each inner action reduces this budget. Once it is exhausted, the Executive can no longer keep consulting specialists freely and must act or wait based on current state.

### Why this matters

Without explicit budget accounting, Cortex would just be a more elaborate prompt template. With explicit budget accounting, Cortex becomes a real compute-allocation policy under pressure.

This is the most important design property of the project.

---

## 15. What Cortex Is Not

To keep the implementation disciplined, Cortex is explicitly **not**:

- hidden chain-of-thought,
- free-form role-play,
- a debate club,
- a many-agent democracy,
- unlogged reasoning.

All useful reasoning must become typed, inspectable artifacts.

---

## 16. Baselines and Ablations

The project requires matched-compute comparisons.

### 16.1 Conditions

### A. Flat-lite
- single-policy agent
- lower compute budget
- no memory
- no critic

### B. Flat-fat
- single-policy agent
- compute budget matched to Cortex-full
- no memory
- no critic

### C. Cortex-lite
- Perception + Planner + Executive
- compute budget matched to Flat-lite
- no memory
- no critic

### D. Cortex-full
- 5 roles
- compute budget matched to Flat-fat
- memory enabled
- critic enabled

### E. Cortex-tuned (stretch)
- 5 roles
- matched compute
- learned budget allocation or better Executive policy

### 16.2 Main comparison

The most important comparison is:

- **Flat-fat vs Cortex-full**

This isolates whether better internal organization beats a flat policy when total compute is held constant.

All other conditions provide context and decomposition.

---

## 17. Metrics

### 17.1 Primary metric

- **Composite reward** from the full reward function.

### 17.2 Secondary metrics

- lives saved,
- time to first correct action,
- safety violations,
- catastrophic cascade rate,
- tokens or budget spent per decision.

### 17.3 Diagnostic metrics

- role-call frequency,
- which role outputs best predict final reward,
- where in the episode budget is consumed,
- whether the agent over-queries or under-queries under uncertainty.

These metrics help explain not just whether Cortex wins, but **how** and **when** it wins or fails.

---

## 18. What Learning Improves

If training is added, the intended learning target is primarily the **Executive policy**.

Training should improve decisions such as:

- when to act,
- when to query a role,
- which role to query,
- when to wait,
- how to allocate remaining deliberation budget across remaining turns.

The specialist roles themselves remain prompt-driven in the MVP. This keeps the project focused on the organization-of-cognition question rather than on retraining the entire system.

---

## 19. Example Turn Flow

A representative Cortex-full turn proceeds as follows:

1. CrisisWorld emits an observation showing a possible surge in one region, stale telemetry, and contradictory stakeholder signals.
2. Perception converts the raw observation into a clean state summary and flags the conflict as an anomaly.
3. Executive requests World Modeler because the anomaly is operationally relevant.
4. World Modeler returns several forecast possibilities with confidence estimates.
5. Executive requests Planner for candidate actions.
6. Planner proposes options such as pre-emptive deployment versus requesting more data.
7. Executive sends a serious candidate to Critic.
8. Critic identifies failure modes or policy risks.
9. Executive checks remaining budget and time sensitivity.
10. Executive chooses either to commit an outer action or spend more budget for further information.
11. CrisisWorld advances and returns the next observation.

This sequence illustrates the intended balance between caution, speed, and compute cost.

---

## 20. Logging and Traceability

Every episode should record both outer and inner behavior.

### Required logged artifacts

- raw outer observations,
- Perception outputs,
- World Modeler outputs,
- Planner candidates,
- Critic assessments,
- Executive decisions,
- budget consumption,
- chosen outer actions,
- resulting rewards,
- termination reason.

### Why logging matters

Logging is necessary for:

- debugging the environment,
- understanding failure modes,
- comparing ablations,
- proving that Cortex is using typed reasoning rather than hidden free text,
- analyzing whether budget was spent effectively.

A per-episode trace file is therefore part of the core project, not optional tooling.

---

## 21. Workstreams

### 21.1 Environment workstream

- implement OpenEnv scaffold,
- build outbreak dynamics,
- add observation partiality and telemetry lag,
- implement action validation and constraint checks,
- implement reward function,
- add seeded scenario generation.

### 21.2 Cortex workstream

- define role I/O schemas,
- write and version role prompts,
- implement Executive loop,
- implement budget accounting,
- create deliberation log format,
- implement flat baseline.

### 21.3 Evaluation workstream

- build multi-seed runner,
- collect metrics,
- run ablations,
- aggregate results,
- inspect traces.

---

## 22. Recommended Build Sequence

The project should be built in the following order:

### Phase 1
- environment skeleton,
- flat baseline,
- one full episode end to end.

### Phase 2
- Cortex-lite,
- first matched-compute numbers.

### Phase 3
- Cortex-full,
- memory support,
- critic integration,
- full ablation run.

### Phase 4
- optional Executive tuning,
- reward-weight sweeps,
- diagnostics and analysis cleanup.

The main rule is:

> Do not start expanding Cortex before the flat baseline runs end to end.

This prevents the project from collapsing under premature complexity.

---

## 23. Open Design Questions

The source document leaves several choices open. These should be explicitly decided during implementation:

### 23.1 Budget units
Should the system charge by role calls, tokens, or both?

A simple starting point is role-call accounting, with token accounting added later if needed.

### 23.2 Memory model
Should memory be:

- keyed append-only,
- retrieval indexed,
- learned summarized memory?

The simplest practical starting point is a keyed append-only log with per-step Executive summaries.

### 23.3 Critic scope
Should Critic evaluate one plan at a time or compare multiple plans jointly?

The clearest MVP choice is per-plan critique.

### 23.4 Stakeholder signals
Should hospitals, media, agencies, and public channels be fully separated from day one?

A simpler MVP can begin with aggregated noisy signals, with stakeholder separation added later.

### 23.5 Communicator role timing
Should the Communicator be in the MVP?

It is useful, but likely better deferred unless the core system is already stable.

### 23.6 Prompting vs training
Should the Executive remain prompt-driven for the demo, or be RL-tuned?

The practical default is prompting first, then training only if time permits.

### 23.7 Fairness / equity scoring
How strongly should the project penalize uneven or unfair allocation across regions?

A modest-weight fairness term is appropriate for MVP without overwhelming the main outbreak-control objective.

---

## 24. Final Project Definition

This project is an OpenEnv-compatible outbreak-control environment and agent system built to test whether structured, budgeted internal reasoning improves high-stakes sequential decision-making.

- **CrisisWorld** provides the outer operational world.
- **Cortex** provides the inner structured deliberation policy.
- **The benchmark** measures whether better organization of cognition yields better outbreak control than a flat baseline under matched compute.

The project succeeds if it is built cleanly, runs reproducibly, and produces an honest comparison between flat and structured reasoning under explicit budget pressure.
