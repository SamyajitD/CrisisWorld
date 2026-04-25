"""CortexAgent — delegates to CortexDeliberator via protocol injection."""

from __future__ import annotations

import logging as stdlib_logging

from models import NoOp, Observation, OuterAction
from protocols.budget import BudgetProtocol
from protocols.logger import LoggerProtocol
from schemas.budget import BudgetExhaustedError
from schemas.episode import LogEvent

logger = stdlib_logging.getLogger(__name__)


class CortexAgent:
    """Agent that delegates decision-making to a CortexDeliberator."""

    def __init__(
        self,
        deliberator: object,
        budget: BudgetProtocol,
        ep_logger: LoggerProtocol,
        initial_budget: int = 20,
    ) -> None:
        self._deliberator = deliberator
        self._budget = budget
        self._logger = ep_logger
        self._initial_budget = initial_budget
        self._turn_count = 0

    def act(self, observation: Observation) -> OuterAction:
        if observation.done:
            return NoOp()

        if self._budget.is_exhausted():
            return NoOp()

        try:
            action, delib_log = self._deliberator.deliberate(
                observation, self._budget
            )
        except BudgetExhaustedError:
            logger.warning("Budget exhausted during deliberation")
            action = NoOp()

        try:
            self._logger.record(
                LogEvent(
                    kind="deliberation",
                    turn=self._turn_count,
                    data={"action_kind": action.kind},
                )
            )
        except Exception:
            logger.warning("Logger failed, continuing")

        self._turn_count += 1
        return action

    def reset(self) -> None:
        self._deliberator.reset()
        self._budget.reset(self._initial_budget)
        self._turn_count = 0
