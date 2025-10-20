from typing import Annotated, Any, TypeVar

from langchain.agents.middleware import AgentState
from pydantic import BaseModel, Field

T = TypeVar("T")


def reduce_list(left: list[T] | None, right: list[T] | None) -> list[T]:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right


class Task(BaseModel):
    task_id: str = Field(
        description="The unique id identifying this task. Should be used when referencing the task elsewhere."
    )
    name: str = Field(
        description="A concise, human-readable name for the task or work item."
    )
    description: str = Field(
        description="A short explanation of what the task entails, including context and purpose."
    )


class Feature(BaseModel):
    feature_id: str = Field(
        description="The unique id identifying this feature. Should be used when referencing the feature elsewhere."
    )
    name: str = Field(
        description="The name of the feature inferred from the requirements."
    )
    description: str = Field(
        description="A short description of the feature inferred from the requirements."
    )


class Tasks(BaseModel):
    """List of Task objects, grouped by feature_id"""

    feature_id: str = Field(
        description="The feature_id this group of tasks belongs to (unique)."
    )
    tasks: list[Task] = Field(
        default_factory=list,
        description="List of Task objects associated with the feature_id.",
    )


class Features(BaseModel):
    """List of Feature objects"""

    phase: str | None = Field(
        default=None,
        description="Optional phase name this feature list belongs to (e.g., Discovery, Core Functionality).",
    )
    data: list[Feature] = Field(
        default_factory=list,
        description="List of `Feature` objects inferred from requirements.",
    )


class ComplexityEstimate(BaseModel):
    feature_id: str = Field(description="The feature_id this estimate refers to.")
    complexity_label: str = Field(
        description="Human-readable label for complexity (e.g., Simple, Medium, Complex)."
    )
    estimated_days: int = Field(
        description="Estimated number of working days to implement the feature."
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Known risks or open questions affecting the estimate.",
    )
    confidence_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0-1.0) for the estimate.",
    )
    reasoning: str = Field(
        default="", description="Short explanation of reasoning behind the estimate."
    )


class AcceptanceCriteria(BaseModel):
    """Structured acceptance criteria for a single task."""

    task_id: str = Field(
        description="The unique id of the task these acceptance criteria apply to."
    )
    scenarios: list[dict[str, str]] = Field(
        default_factory=list,
        description="BDD-style scenarios with keys like given/when/then",
    )
    unit_tests: list[dict[str, str]] = Field(
        default_factory=list,
        description="Unit test definitions: {test_name, what_it_tests, expected_behavior}",
    )
    integration_tests: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Integration test definitions: {test_name, components, what_it_tests, expected_behavior}",
    )
    edge_cases: list[str] = Field(
        default_factory=list, description="Notable edge cases to validate against"
    )


class TaskPrompt(BaseModel):
    task_id: str = Field(
        description="The unique id of the task this prompt was generated for."
    )
    prompt: str = Field(
        description="The generated prompt string intended for an AI assistant."
    )


# class DependencyGraph(BaseModel):
#     # task_id -> list of blocking task_ids
#     task_relationships: Dict[str, List[str]]
#     execution_phases: List[List[str]]  # parallel groups (topologically sorted)
#     critical_path: List[str]
#     suggested_order: List[str]


class ProjectPlanState(AgentState):
    raw_requirements: str

    # Populated by parse_requirements_tool
    features: Features
    # constraints: list[str]
    # stakeholders: list[str]

    # Populated by generate_tasks_tool (per feature)
    tasks_by_feature: Annotated[list[Tasks], reduce_list]

    # Populated by estimate_complexity_tool (per feature)
    complexity_by_feature: Annotated[list[ComplexityEstimate], reduce_list]

    # Populated by create_acceptance_criteria_tool (per task)
    criteria_by_task: Annotated[list[AcceptanceCriteria], reduce_list]

    # # Populated by generate_prompt_for_copilot_tool (per task)
    prompts_by_task: Annotated[list[TaskPrompt], reduce_list]

    # Populated by detect_dependencies_tool
    # dependency_graph: DependencyGraph
