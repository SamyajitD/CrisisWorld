"""Structural subtyping tests for all protocols."""

from __future__ import annotations

from pathlib import Path

from CrisisWorld.models import Observation, OuterAction
from CrisisWorld.schemas.artifact import Artifact, CleanState, RoleInput
from CrisisWorld.schemas.budget import BudgetStatus
from CrisisWorld.schemas.episode import LogEvent, MemoryDigest


# ---------------------------------------------------------------------------
# EnvProtocol
# ---------------------------------------------------------------------------

class _ValidEnv:
    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation: ...  # type: ignore[empty-body]
    def step(self, action, timeout_s=None, **kwargs) -> Observation: ...  # type: ignore[empty-body]

    @property
    def state(self): ...  # type: ignore[empty-body]

    def get_metadata(self): ...  # type: ignore[empty-body]
    def close(self) -> None: ...


class _MissingStep:
    def reset(self, seed=None, episode_id=None, **kwargs) -> Observation: ...  # type: ignore[empty-body]

    @property
    def state(self): ...  # type: ignore[empty-body]

    def get_metadata(self): ...  # type: ignore[empty-body]
    def close(self) -> None: ...


def test_env_protocol_structural_subtyping() -> None:
    from CrisisWorld.protocols.env import EnvProtocol

    assert isinstance(_ValidEnv(), EnvProtocol)


def test_env_protocol_rejects_incomplete_impl() -> None:
    from CrisisWorld.protocols.env import EnvProtocol

    assert not isinstance(_MissingStep(), EnvProtocol)


def test_env_protocol_rejects_unrelated_object() -> None:
    from CrisisWorld.protocols.env import EnvProtocol

    assert not isinstance("not an env", EnvProtocol)
    assert not isinstance(42, EnvProtocol)


# ---------------------------------------------------------------------------
# AgentProtocol
# ---------------------------------------------------------------------------

class _ValidAgent:
    def act(self, observation: Observation) -> OuterAction: ...  # type: ignore[empty-body]
    def reset(self) -> None: ...


class _AgentMissingReset:
    def act(self, observation: Observation) -> OuterAction: ...  # type: ignore[empty-body]


class _AgentMissingAct:
    def reset(self) -> None: ...


def test_agent_protocol_structural_subtyping() -> None:
    from CrisisWorld.protocols.agent import AgentProtocol

    assert isinstance(_ValidAgent(), AgentProtocol)


def test_agent_protocol_rejects_missing_reset() -> None:
    from CrisisWorld.protocols.agent import AgentProtocol

    assert not isinstance(_AgentMissingReset(), AgentProtocol)


def test_agent_protocol_rejects_missing_act() -> None:
    from CrisisWorld.protocols.agent import AgentProtocol

    assert not isinstance(_AgentMissingAct(), AgentProtocol)


# ---------------------------------------------------------------------------
# RoleProtocol
# ---------------------------------------------------------------------------

class _ValidRole:
    @property
    def role_name(self) -> str:
        return "mock"

    @property
    def cost(self) -> int:
        return 1

    def invoke(self, role_input: RoleInput) -> Artifact: ...  # type: ignore[empty-body]


class _RoleMissingInvoke:
    @property
    def role_name(self) -> str:
        return "mock"

    @property
    def cost(self) -> int:
        return 1


class _RolePlainAttrs:
    role_name: str = "mock"
    cost: int = 1

    def invoke(self, role_input: RoleInput) -> Artifact: ...  # type: ignore[empty-body]


def test_role_protocol_structural_subtyping() -> None:
    from CrisisWorld.protocols.role import RoleProtocol

    assert isinstance(_ValidRole(), RoleProtocol)


def test_role_protocol_rejects_missing_invoke() -> None:
    from CrisisWorld.protocols.role import RoleProtocol

    assert not isinstance(_RoleMissingInvoke(), RoleProtocol)


def test_role_protocol_accepts_plain_attributes() -> None:
    from CrisisWorld.protocols.role import RoleProtocol

    assert isinstance(_RolePlainAttrs(), RoleProtocol)


# ---------------------------------------------------------------------------
# BudgetProtocol
# ---------------------------------------------------------------------------

class _ValidBudget:
    def charge(self, cost: int) -> None: ...
    def remaining(self) -> BudgetStatus: ...  # type: ignore[empty-body]
    def is_exhausted(self) -> bool: ...  # type: ignore[empty-body]
    def reset(self, total: int) -> None: ...


class _BudgetMissingCharge:
    def remaining(self) -> BudgetStatus: ...  # type: ignore[empty-body]
    def is_exhausted(self) -> bool: ...  # type: ignore[empty-body]
    def reset(self, total: int) -> None: ...


class _BudgetMissingReset:
    def charge(self, cost: int) -> None: ...
    def remaining(self) -> BudgetStatus: ...  # type: ignore[empty-body]
    def is_exhausted(self) -> bool: ...  # type: ignore[empty-body]


def test_budget_protocol_structural_subtyping() -> None:
    from CrisisWorld.protocols.budget import BudgetProtocol

    assert isinstance(_ValidBudget(), BudgetProtocol)


def test_budget_protocol_rejects_missing_charge() -> None:
    from CrisisWorld.protocols.budget import BudgetProtocol

    assert not isinstance(_BudgetMissingCharge(), BudgetProtocol)


def test_budget_protocol_rejects_missing_reset() -> None:
    from CrisisWorld.protocols.budget import BudgetProtocol

    assert not isinstance(_BudgetMissingReset(), BudgetProtocol)


# ---------------------------------------------------------------------------
# MemoryProtocol
# ---------------------------------------------------------------------------

class _ValidMemory:
    def store(self, key: str, artifact: Artifact) -> None: ...
    def retrieve(self, key: str) -> list[Artifact]: ...  # type: ignore[empty-body]
    def digest(self) -> MemoryDigest: ...  # type: ignore[empty-body]
    def clear(self) -> None: ...


class _MemoryMissingStore:
    def retrieve(self, key: str) -> list[Artifact]: ...  # type: ignore[empty-body]
    def digest(self) -> MemoryDigest: ...  # type: ignore[empty-body]
    def clear(self) -> None: ...


class _MemoryMissingClear:
    def store(self, key: str, artifact: Artifact) -> None: ...
    def retrieve(self, key: str) -> list[Artifact]: ...  # type: ignore[empty-body]
    def digest(self) -> MemoryDigest: ...  # type: ignore[empty-body]


def test_memory_protocol_structural_subtyping() -> None:
    from CrisisWorld.protocols.memory import MemoryProtocol

    assert isinstance(_ValidMemory(), MemoryProtocol)


def test_memory_protocol_rejects_missing_store() -> None:
    from CrisisWorld.protocols.memory import MemoryProtocol

    assert not isinstance(_MemoryMissingStore(), MemoryProtocol)


def test_memory_protocol_rejects_missing_clear() -> None:
    from CrisisWorld.protocols.memory import MemoryProtocol

    assert not isinstance(_MemoryMissingClear(), MemoryProtocol)


# ---------------------------------------------------------------------------
# LoggerProtocol
# ---------------------------------------------------------------------------

class _ValidLogger:
    def record(self, event: LogEvent) -> None: ...
    def save(self, path: Path) -> Path: ...  # type: ignore[empty-body]
    def flush(self) -> None: ...


class _LoggerMissingSave:
    def record(self, event: LogEvent) -> None: ...
    def flush(self) -> None: ...


class _LoggerMissingFlush:
    def record(self, event: LogEvent) -> None: ...
    def save(self, path: Path) -> Path: ...  # type: ignore[empty-body]


def test_logger_protocol_structural_subtyping() -> None:
    from CrisisWorld.protocols.logger import LoggerProtocol

    assert isinstance(_ValidLogger(), LoggerProtocol)


def test_logger_protocol_rejects_missing_save() -> None:
    from CrisisWorld.protocols.logger import LoggerProtocol

    assert not isinstance(_LoggerMissingSave(), LoggerProtocol)


def test_logger_protocol_rejects_missing_flush() -> None:
    from CrisisWorld.protocols.logger import LoggerProtocol

    assert not isinstance(_LoggerMissingFlush(), LoggerProtocol)
