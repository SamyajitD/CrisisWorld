"""Data layer — frozen Pydantic v2 models, no behavior, no side effects.

Agent-side types are canonical here. Env-contract types are re-exported
from models.py for convenience.
"""

# --- Agent-side types (canonical) ---
from schemas.artifact import (
    Artifact,
    BeliefState,
    CandidateAction,
    CleanState,
    Critique,
    ExecutiveDecision,
    Plan,
    RoleInput,
)
from schemas.budget import (
    BudgetExhaustedError,
    BudgetLedger,
    BudgetStatus,
    LedgerEntry,
)
from schemas.config import (
    CortexConfig,
    ExperimentConfig,
)
from schemas.episode import (
    EpisodeResult,
    EpisodeTrace,
    LogEvent,
    MemoryDigest,
    TurnRecord,
)

# --- Env-contract re-exports (canonical source: models.py) ---
from models import (
    ActionUnion,
    BudgetStatusSnapshot,
    CompositeReward,
    Constraint,
    CrisisState,
    DeployResource,
    EnvConfig,
    EnvironmentMetadata,
    Escalate,
    IncidentReport,
    NoOp,
    Observation,
    OuterAction,
    PublicCommunication,
    ReallocateBudget,
    RegionState,
    RequestData,
    ResourcePool,
    RestrictMovement,
    RewardComponents,
    RewardWeights,
    StakeholderSignal,
    Telemetry,
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
    "CrisisState",
    "EnvironmentMetadata",
    "RegionState",
    "ResourcePool",
]
