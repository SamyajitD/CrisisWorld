"""Action schemas for outer environment actions."""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OuterAction(BaseModel):
    """Base model for all outer actions."""

    model_config = ConfigDict(frozen=True)

    kind: str


class DeployResource(OuterAction):
    """Send resources to a region."""

    kind: Literal["deploy_resource"] = "deploy_resource"
    resource: str
    region_id: str
    amount: int = Field(gt=0)


class RestrictMovement(OuterAction):
    """Impose movement restrictions on a region."""

    kind: Literal["restrict_movement"] = "restrict_movement"
    region_id: str
    level: int = Field(ge=0, le=3)


class RequestData(OuterAction):
    """Request better data from a source (costs budget)."""

    kind: Literal["request_data"] = "request_data"
    source: str


class PublicCommunication(OuterAction):
    """Issue a public statement."""

    kind: Literal["public_communication"] = "public_communication"
    audience: str
    message: str = Field(min_length=1)


class Escalate(OuterAction):
    """Escalate to a higher authority."""

    kind: Literal["escalate"] = "escalate"
    agency: str


class ReallocateBudget(OuterAction):
    """Shift budget between categories."""

    kind: Literal["reallocate_budget"] = "reallocate_budget"
    from_category: str
    to_category: str
    amount: int = Field(gt=0)

    @model_validator(mode="after")
    def _check_categories_differ(self) -> ReallocateBudget:
        if self.from_category == self.to_category:
            raise ValueError("from_category and to_category must differ")
        return self


class NoOp(OuterAction):
    """Do nothing this turn."""

    kind: Literal["noop"] = "noop"


ActionUnion = Annotated[
    Union[
        DeployResource,
        RestrictMovement,
        RequestData,
        PublicCommunication,
        Escalate,
        ReallocateBudget,
        NoOp,
    ],
    Field(discriminator="kind"),
]
