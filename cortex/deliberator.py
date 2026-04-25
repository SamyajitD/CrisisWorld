"""CortexDeliberator — orchestrates the inner deliberation loop."""

from __future__ import annotations

import logging as stdlib_logging
from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

from models import ActionUnion, Escalate, NoOp, Observation, OuterAction
from protocols.budget import BudgetProtocol
from protocols.logger import LoggerProtocol
from protocols.memory import MemoryProtocol
from protocols.role import RoleProtocol
from schemas.artifact import Artifact, ExecutiveDecision, RoleInput
from schemas.budget import BudgetExhaustedError
from schemas.episode import LogEvent

_log = stdlib_logging.getLogger(__name__)

MAX_INNER_ITERATIONS = 20
PERCEPTION_COST = 1
EXECUTIVE_COST = 1
_action_adapter: TypeAdapter[ActionUnion] = TypeAdapter(ActionUnion)


class DeliberationLog(BaseModel):
    """Immutable record of a single deliberation call."""

    model_config = ConfigDict(frozen=True)
    artifacts: tuple[dict[str, Any], ...] = ()
    iterations: int = 0
    budget_at_start: dict[str, Any] = {}
    budget_at_end: dict[str, Any] = {}
    termination_reason: str = ""
    forced: bool = False


class CortexDeliberator:
    """Orchestrates Perception -> Executive loop with budget accounting."""

    def __init__(
        self,
        roles: dict[str, RoleProtocol],
        memory: MemoryProtocol,
        logger: LoggerProtocol,
    ) -> None:
        if "perception" not in roles:
            raise ValueError("roles must contain 'perception'")
        if "executive" not in roles:
            raise ValueError("roles must contain 'executive'")
        self._roles = roles
        self._memory = memory
        self._logger = logger
        self._turn_artifacts: list[dict[str, Any]] = []

    def deliberate(
        self, observation: Observation, budget: BudgetProtocol,
    ) -> tuple[OuterAction, DeliberationLog]:
        budget_start = budget.remaining().model_dump()
        self._turn_artifacts = []

        # -- Perception --
        try:
            budget.charge(PERCEPTION_COST)
        except BudgetExhaustedError:
            return self._forced(budget, budget_start, 0, "budget_exhausted")

        p_in = RoleInput(role_name="perception", payload={"observation": observation.model_dump()})
        self._record(self._roles["perception"].invoke(p_in), "perception")

        # -- Executive loop --
        iterations = 0
        for iterations in range(1, MAX_INNER_ITERATIONS + 1):  # noqa: B007
            try:
                budget.charge(EXECUTIVE_COST)
            except BudgetExhaustedError:
                return self._forced(budget, budget_start, iterations, "budget_exhausted")

            e_in = RoleInput(
                role_name="executive",
                payload={"artifacts": list(self._turn_artifacts), "budget_status": budget.remaining().model_dump()},
            )
            e_art = self._roles["executive"].invoke(e_in)
            self._record(e_art, "executive")

            if not isinstance(e_art, ExecutiveDecision):
                continue
            d = e_art.decision

            if d == "act":
                return self._parse_action(e_art.target_action), self._log(budget, budget_start, iterations, "act")
            if d in ("wait", "stop"):
                return NoOp(), self._log(budget, budget_start, iterations, d)
            if d == "escalate":
                return Escalate(agency="crisis_authority"), self._log(budget, budget_start, iterations, "escalate")
            if d == "call":
                res = self._dispatch(e_art.target_role, budget, budget_start, iterations)
                if res is not None:
                    return res

        return self._forced(budget, budget_start, iterations, "max_iterations")

    def reset(self) -> None:
        self._turn_artifacts = []
        self._memory.clear()

    # -- private helpers --

    def _dispatch(
        self, target: str | None, budget: BudgetProtocol, bs: dict[str, Any], iters: int,
    ) -> tuple[OuterAction, DeliberationLog] | None:
        if not target or target not in self._roles:
            _log.warning("Invalid or missing role: %s — skipping", target)
            return None
        role = self._roles[target]
        try:
            budget.charge(role.cost)
        except BudgetExhaustedError:
            return self._forced(budget, bs, iters, "budget_exhausted")
        r_in = RoleInput(
            role_name=target,
            payload={"artifacts": list(self._turn_artifacts), "memory_digest": self._memory.digest().model_dump()},
        )
        self._record(role.invoke(r_in), target)
        return None

    def _record(self, artifact: Artifact, role_name: str) -> None:
        dumped = artifact.model_dump() if hasattr(artifact, "model_dump") else artifact
        self._turn_artifacts.append(dumped)
        self._memory.store(role_name, artifact)
        self._logger.record(LogEvent(kind="artifact", turn=0, data={"role": role_name, **dumped}))

    def _parse_action(self, target_action: dict[str, Any] | None) -> OuterAction:
        if target_action is None:
            return NoOp()
        try:
            return _action_adapter.validate_python(target_action)
        except Exception:
            _log.warning("Failed to parse target_action %s — defaulting to NoOp", target_action)
            return NoOp()

    def _log(
        self, budget: BudgetProtocol, bs: dict[str, Any], iters: int, reason: str, *, forced: bool = False,
    ) -> DeliberationLog:
        return DeliberationLog(
            artifacts=tuple(self._turn_artifacts), iterations=iters,
            budget_at_start=bs, budget_at_end=budget.remaining().model_dump(),
            termination_reason=reason, forced=forced,
        )

    def _forced(
        self, budget: BudgetProtocol, bs: dict[str, Any], iters: int, reason: str,
    ) -> tuple[OuterAction, DeliberationLog]:
        return NoOp(), self._log(budget, bs, iters, reason, forced=True)
