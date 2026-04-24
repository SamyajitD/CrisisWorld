"""State schemas for environment state representation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from src.schemas.observation import Observation
    from src.schemas.reward import CompositeReward


class RegionState(BaseModel):
    """Epidemiological state of one region."""

    model_config = ConfigDict(frozen=True)

    region_id: str
    population: int = Field(gt=0)
    infected: int = Field(default=0, ge=0)
    recovered: int = Field(default=0, ge=0)
    deceased: int = Field(default=0, ge=0)
    restricted: bool = False

    @model_validator(mode="after")
    def _check_population_invariant(self) -> RegionState:
        total = self.infected + self.recovered + self.deceased
        if total > self.population:
            raise ValueError(
                f"infected ({self.infected}) + recovered ({self.recovered}) "
                f"+ deceased ({self.deceased}) = {total} "
                f"exceeds population ({self.population})"
            )
        return self


class ResourcePool(BaseModel):
    """Available operational resources."""

    model_config = ConfigDict(frozen=True)

    medical: int = Field(default=0, ge=0)
    personnel: int = Field(default=0, ge=0)
    funding: int = Field(default=0, ge=0)


class Constraint(BaseModel):
    """Active policy or legal constraint."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    active: bool = True

class StepResult(BaseModel):
    """Return value of env.step()."""

    model_config = ConfigDict(frozen=True)

    observation: Observation
    reward: CompositeReward
    done: bool
    info: dict[str, Any] = {}
