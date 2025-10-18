from langchain.agents.middleware import AgentState
from pydantic import BaseModel, Field


class CatalystState(AgentState):
    pass


class Requirements(BaseModel):
    features: list[str] = Field(description="A list of required features")
    # stakeholders: list[str] = Field(description="")
    constraints: list[str] = Field(
        description="Any business-related constraints, such as maximum budget or time constraints"
    )
    success_criteria: list[str] = Field(
        description="A list of criteria necessary to achieve a minimum viable product"
    )
