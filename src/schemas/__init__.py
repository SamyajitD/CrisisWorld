"""Data layer — frozen Pydantic v2 models, no behavior, no side effects."""

from src.schemas.action import (
    ActionUnion,
    DeployResource,
    Escalate,
    NoOp,
    OuterAction,
    PublicCommunication,
    ReallocateBudget,
    RequestData,
    RestrictMovement,
)
from src.schemas.artifact import (
    Artifact,
    BeliefState,
    CandidateAction,
    CleanState,
    Critique,
    ExecutiveDecision,
    Plan,
    RoleInput,
)
from src.schemas.budget import (
    BudgetExhaustedError,
    BudgetLedger,
    BudgetStatus,
    LedgerEntry,
)
from src.schemas.config import (
    CortexConfig,
    EnvConfig,
    ExperimentConfig,
    RewardWeights,
)
from src.schemas.episode import (
    EpisodeResult,
    EpisodeTrace,
    LogEvent,
    MemoryDigest,
    TurnRecord,
)
from src.schemas.observation import (
    IncidentReport,
    Observation,
    StakeholderSignal,
    Telemetry,
)
from src.schemas.reward import CompositeReward, RewardComponents
from src.schemas.state import Constraint, RegionState, ResourcePool, StepResult

# Resolve forward references in StepResult (Observation, CompositeReward)
StepResult.model_rebuild(
    _types_namespace={
        "Observation": Observation,
        "CompositeReward": CompositeReward,
    }
)

__all__ = [
    # action
    "ActionUnion",
    "DeployResource",
    "Escalate",
    "NoOp",
    "OuterAction",
    "PublicCommunication",
    "ReallocateBudget",
    "RequestData",
    "RestrictMovement",
    # artifact
    "Artifact",
    "BeliefState",
    "CandidateAction",
    "CleanState",
    "Critique",
    "ExecutiveDecision",
    "Plan",
    "RoleInput",
    # budget
    "BudgetExhaustedError",
    "BudgetLedger",
    "BudgetStatus",
    "LedgerEntry",
    # config
    "CortexConfig",
    "EnvConfig",
    "ExperimentConfig",
    "RewardWeights",
    # episode
    "EpisodeResult",
    "EpisodeTrace",
    "LogEvent",
    "MemoryDigest",
    "TurnRecord",
    # observation
    "IncidentReport",
    "Observation",
    "StakeholderSignal",
    "Telemetry",
    # reward
    "CompositeReward",
    "RewardComponents",
    # state
    "Constraint",
    "RegionState",
    "ResourcePool",
    "StepResult",
]
