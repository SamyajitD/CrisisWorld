# tracing/ -- Episode Tracing

## Behavior

Implements `LoggerProtocol` for structured episode tracing.

**Every turn records:** observation, role artifacts, executive decision, action, reward, budget snapshot.
Traces are structured JSON (one file per episode) in `traces/`. Results in `results/`.
**Hard rule:** If a turn is not logged, it is a bug (Absolute Rule 6).

`EpisodeTracer` uses the `episode_id` from OpenEnv's `CrisisState` (set by
`env.reset(episode_id=...)`) rather than generating its own. `TurnRecord.turn`
should align with `CrisisState.step_count`. Each turn also logs
`Observation.metadata` and `Action.metadata` fields.

> **Note:** This package was renamed from `logging/` to `tracing/` to avoid
> shadowing Python's stdlib `logging` module.

---

## Exposed APIs

### tracer.py -- EpisodeTracer

```python
class TracerFinalizedError(Exception): ...

class EpisodeTracer:
    def __init__(self, episode_id: str) -> None: ...
    def record(self, event: LogEvent) -> None: ...
    def finalize(self) -> EpisodeTrace: ...
    def save(self, path: Path) -> Path: ...  # LoggerProtocol bridge
```

- `record` -- Append LogEvent. Raises `TracerFinalizedError` after finalize().
- `finalize` -- Close trace, return EpisodeTrace. Idempotent (second call returns cached).
- `save` -- Implicitly finalizes, delegates to `serializer.save_trace()`.

### serializer.py

```python
def save_trace(trace: EpisodeTrace, path: Path) -> Path: ...
def load_trace(path: Path) -> EpisodeTrace: ...
```

### formatters.py

```python
def render_human_readable(trace: EpisodeTrace) -> str: ...
def render_turn(record: TurnRecord) -> str: ...
```

---

## Implementation Plan

### tracer.py -- EpisodeTracer

**Internal state:** `_episode_id: str`, `_events: list[LogEvent]`, `_finalized: bool`, `_cached_trace: EpisodeTrace | None`

**record(event):**
1. If `_finalized` -> raise `TracerFinalizedError(episode_id)`.
2. Append event to `_events`.
3. No content validation (schema layer handles it).

**finalize():**
1. If `_cached_trace is not None` -> return it (idempotent).
2. Set `_finalized = True`.
3. Group events by turn into TurnRecords (private `_group_events_by_turn`).
4. Build EpisodeTrace from episode_id + TurnRecords.
5. Cache and return.

**save(path):** Calls `finalize()`, then `serializer.save_trace(self._cached_trace, path)`. Returns path.

**LoggerProtocol conformance:** record() maps directly. save() bridges. flush() clears `_events` (only if not finalized).

**Thread safety:** Not implemented. Single-threaded by design.

**Edge cases:**
1. record after finalize -> TracerFinalizedError with episode_id in message
2. finalize twice -> same object returned (identity: `trace1 is trace2`)
3. empty trace -> valid EpisodeTrace with empty turns
4. 10000+ events -> ~10MB in memory, acceptable
5. unexpected LogEvent kind -> accepted (tracer is dumb append log)
6. save before any record -> empty trace saved
7. empty episode_id -> accepted, caller's responsibility

### serializer.py -- Atomic Trace I/O

**save_trace(trace, path):**
1. Resolve path, create parent dirs: `path.parent.mkdir(parents=True, exist_ok=True)`.
2. Serialize: `trace.model_dump_json(indent=2)`.
3. Atomic write: write to `path.with_suffix(".tmp")`, then `tmp.replace(path)`.
4. Cleanup temp in `finally` block.
5. Return path.

**load_trace(path):**
1. `raw = path.read_text(encoding="utf-8")`
2. `EpisodeTrace.model_validate_json(raw)`
3. FileNotFoundError, JSONDecodeError, ValidationError propagate naturally.

**Edge cases:**
1. Nonexistent directory -> created by mkdir
2. Corrupted JSON -> JSONDecodeError propagates
3. Nonexistent file -> FileNotFoundError propagates
4. Empty trace -> valid small JSON
5. Unicode in data -> preserved (Pydantic handles correctly)
6. Path is directory -> IsADirectoryError propagates
7. Concurrent writes -> last writer wins (acceptable, unique paths per episode)

### formatters.py -- Human-Readable Output

**render_human_readable(trace):**

```
========================================
Episode: {episode_id}
Turns: {len(turns)} | Total Reward: {sum}
========================================
[per-turn blocks via render_turn()]
========================================
Final Summary
  Total Reward: {total}
  Turns Played: {count}
  Termination: {reason}
========================================
```

**render_turn(record):**

```
--- Turn {n} ---
Obs: [region=north infected=120 | region=south infected=45] (truncated)
Roles:                          # omitted if no artifacts (flat agent)
  Perception: 2 anomalies
  Planner: 3 candidates
Action: deploy_resource(north, medical, 50)
Reward: -0.32 (outcome=0.1, timeliness=-0.2, safety=0.0)
Budget: 18/30 remaining
```

**Truncation:** `MAX_LINE_WIDTH = 100`. Long values truncated with `...`. Observation shows max 3 regions + `(+N more)`. Newlines in field values replaced with `\n` literal.

**Helpers:** `_truncate(text, max_len)`, `_summarize_observation()`, `_summarize_artifacts()`, `_render_header()`, `_render_footer()`

**Edge cases:**
1. Empty trace -> header + footer with Turns: 0, no turn blocks
2. 1000+ turns -> all rendered (no pagination, caller slices if needed)
3. No artifacts -> Roles: section omitted
4. Long values -> truncated to MAX_LINE_WIDTH
5. Special characters -> newlines/tabs escaped
6. None reward/budget -> "N/A"
7. Non-ASCII episode_id -> rendered as-is

---

## External Dependencies

| Import | Source | Why |
|--------|--------|-----|
| `LoggerProtocol` | `protocols.logger` | EpisodeTracer implements this |
| `EpisodeTrace`, `TurnRecord`, `LogEvent` | `schemas.episode` | Core data types |
| `Artifact` | `schemas.artifact` | Logged in turn records |
| `BudgetStatus` | `schemas.budget` | Budget snapshot per turn |
| `CompositeReward` | `models` | Reward per turn |
| `Observation` | `models` | Observation per turn (includes `metadata` to log) |

---

## Test Plan

### tracer.py tests

```
test_tracer_record_appends_events_in_order         -- 5 events, finalize, check order by kind
test_tracer_finalize_returns_episode_trace          -- return type, episode_id matches
test_tracer_finalize_is_idempotent                 -- trace1 is trace2 (identity)
test_tracer_record_after_finalize_raises           -- TracerFinalizedError, episode_id in message
test_tracer_finalize_empty_trace                   -- valid EpisodeTrace with 0 turns
test_tracer_large_event_count                      -- 10000 events, all present
test_tracer_save_delegates_to_serializer           -- file exists, load matches
test_tracer_save_finalizes_implicitly              -- record after save raises
```

### serializer.py tests

```
test_serializer_round_trip                         -- save + load == original
test_serializer_creates_parent_dirs                -- deeply nested path works
test_serializer_atomic_write                       -- no .tmp file remains
test_serializer_load_nonexistent_raises            -- FileNotFoundError
test_serializer_load_corrupted_json                -- exception raised
test_serializer_save_empty_trace                   -- round-trip succeeds
test_serializer_unicode_preservation               -- CJK/emoji preserved
```

### formatters.py tests

```
test_formatter_human_readable_structure            -- contains episode_id, Turn 0/1/2, Final Summary
test_formatter_render_turn_contains_key_fields     -- turn number, action kind, reward total, budget
test_formatter_empty_trace_renders                 -- Turns: 0, no Turn lines
test_formatter_turn_without_artifacts_omits_roles  -- "Roles:" absent
test_formatter_truncates_long_values               -- no line > MAX_LINE_WIDTH+10, "..." present
test_formatter_special_characters_escaped          -- no raw \n in field values
test_formatter_none_optional_fields                -- "N/A" for None reward/budget
```

---

## Implementation Order

1. `tracer.py` (TracerFinalizedError + EpisodeTracer)
2. `serializer.py` (save_trace + load_trace)
3. `formatters.py` (render_turn + render_human_readable)
4. `__init__.py` (exports)
5. Unit tests for all three modules

## File Size Targets

| File | Estimated | Hard Max |
|------|-----------|----------|
| tracer.py | ~80 | 150 |
| serializer.py | ~50 | 100 |
| formatters.py | ~160 | 300 |
| __init__.py | ~10 | 20 |

## Risks

- **Schema models not yet implemented**: episode.py must be built first. Write tests with inline fixtures matching spec until schemas ready.
- **EpisodeTrace structure ambiguous**: `_group_events_by_turn` adapts to actual schema. Tracer public API is stable either way.
- **Atomic writes POSIX-only**: Project targets Linux. Documented assumption.
