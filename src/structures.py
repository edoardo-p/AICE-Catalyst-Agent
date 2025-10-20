from langchain.agents.middleware import AgentState
from pydantic import BaseModel, Field


class ComplexityEstimate(BaseModel):
    complexity_label: str
    estimated_days: int
    risks: list[str]
    confidence_level: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class Task(BaseModel):
    name: str = Field(
        description="A concise, human-readable name for the task or work item."
    )
    description: str = Field(
        description="A short explanation of what the task entails, including context and purpose."
    )
    ai_tool_prompt: str = Field(
        default="",
        description="A prompt or instruction specifically designed for an AI tool to assist in completing this task.",
    )
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="A list of conditions or measurable outcomes that define when the task is considered complete and successful.",
    )


class Tasks(BaseModel):
    tasks: list[Task] = Field(
        description="A list of Task objects representing individual work items."
    )


class Phase(BaseModel):
    features: dict[str, Tasks] = Field(
        description="A dictionary mapping each feature name to an associated list of tasks."
    )
    complexity: ComplexityEstimate = Field(
        description="An estimate of how difficult the task is, how long it is going to take, and any risks or assumptions made to reach the estimate."
    )


class Requirements(BaseModel):
    features: list[str] = Field(
        description="A list of inferred features necessary to complete the project."
    )
    stakeholders: list[str] = Field(
        default_factory=list,
        description="Any persons or groups of people mentioned that have an interest in the project.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Any business-related constraints mentioned, such as time or money constraints",
    )


class ProjectPlan(BaseModel):
    requirements: Requirements | None = None
    phases: list[Phase] | None = None


class PlannerState(AgentState):
    data: ProjectPlan
