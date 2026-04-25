"""Artifact schemas for Cortex role outputs."""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class RoleInput(BaseModel):
    """Typed input envelope for a role invocation."""

    model_config = ConfigDict(frozen=True)

    role_name: str
    payload: dict[str, Any]


class CleanState(BaseModel):
    """Perception output."""

    model_config = ConfigDict(frozen=True)

    salient_changes: tuple[str, ...] = ()
    flagged_anomalies: tuple[str, ...] = ()
    cleaned_observation: dict[str, Any] = {}


class BeliefState(BaseModel):
    """World Modeler output."""

    model_config = ConfigDict(frozen=True)

    hidden_var_estimates: dict[str, Any] = {}
    forecast_trajectories: tuple[dict[str, Any], ...] = ()
    confidence: float = Field(ge=0.0, le=1.0)


class CandidateAction(BaseModel):
    """Single candidate within a Plan."""

    model_config = ConfigDict(frozen=True)

    action: dict[str, Any]
    rationale: str
    expected_effect: str
    confidence: float = Field(ge=0.0, le=1.0)


class Plan(BaseModel):
    """Planner output: list of candidates."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[CandidateAction, ...] = ()


class Critique(BaseModel):
    """Critic output."""

    model_config = ConfigDict(frozen=True)

    failure_modes: tuple[str, ...] = ()
    policy_violations: tuple[str, ...] = ()
    recommended_amendments: tuple[str, ...] = ()
    risk_score: float = Field(ge=0.0, le=1.0)


class ExecutiveDecision(BaseModel):
    """Executive output."""

    model_config = ConfigDict(frozen=True)

    decision: Literal["act", "call", "wait", "escalate", "stop"]
    target_action: dict[str, Any] | None = None
    target_role: str | None = None
    reasoning: str


Artifact = Union[CleanState, BeliefState, Plan, Critique, ExecutiveDecision]
