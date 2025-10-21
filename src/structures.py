import json
from collections import defaultdict
from typing import Annotated, Any, Literal, TypeVar

from langchain.agents.middleware import AgentState
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

T = TypeVar("T")


def reduce_dict(left: dict[str, T] | None, right: dict[str, T] | None) -> dict[str, T]:
    """Safely combine two dicts, handling cases where either or both inputs might be None.
    In case of key overlap, values in the right dict will overwrite those in the left one."""
    if not left:
        left = {}
    if not right:
        right = {}
    return left | right


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

    def __str__(self) -> str:
        return f"{self.name} (ID {self.task_id}): {self.description}"


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
    phase: str | None = Field(
        default=None,
        description="Optional phase name this feature list belongs to (e.g., Discovery, Core Functionality).",
    )

    def __str__(self) -> str:
        return f"{self.name} (ID {self.feature_id}): {self.description}"


class Tasks(BaseModel):
    tasks: list[Task] = Field(
        default_factory=list,
        description="List of `Task` objects.",
    )


class Features(BaseModel):
    features: list[Feature] = Field(
        default_factory=list,
        description="List of `Feature` objects inferred from requirements.",
    )


class ComplexityEstimate(BaseModel):
    complexity_label: Literal["Simple", "Medium", "Complex", "Very Complex"] = Field(
        description="Human-readable label for complexity"
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


class Scenario(BaseModel):
    """BDD-style scenario using Given/When/Then syntax."""

    given: str = Field(description="Initial context or precondition.")
    when: str = Field(description="Action or event that triggers the behavior.")
    then: str = Field(description="Expected outcome or result.")


class UnitTest(BaseModel):
    """Definition of a single unit test."""

    test_name: str = Field(description="Name of the unit test.")
    what_it_tests: str = Field(
        description="What functionality or logic the test verifies."
    )
    expected_behavior: str = Field(
        description="Expected outcome of the test when it passes."
    )


class IntegrationTest(BaseModel):
    """Definition of a single integration test."""

    test_name: str = Field(description="Name of the integration test.")
    components: list[str] = Field(
        description="List of system components or modules involved in the test."
    )
    what_it_tests: str = Field(
        description="What the test verifies at the system integration level."
    )
    expected_behavior: str = Field(
        description="Expected system behavior under integration conditions."
    )


class AcceptanceCriteria(BaseModel):
    """Structured acceptance criteria for a single task or feature."""

    scenarios: list[Scenario] = Field(
        description="BDD-style Given/When/Then scenarios describing expected behavior.",
    )

    unit_tests: list[UnitTest] = Field(
        description="Suite of unit tests covering individual components or functions.",
    )

    integration_tests: list[IntegrationTest] | None = Field(
        default=None,
        description="Suite of integration tests validating multi-component behavior.",
    )


class TaskBlockers(BaseModel):
    task_id: str = Field(description="The id of the current task which is blocked.")
    blocking_tasks: list[str] = Field(
        description="A list of task ids associated to tasks which need to be completed before the current task.",
    )


class DependencyGraph(BaseModel):
    task_relationships: list[TaskBlockers] = Field(
        description="List of `TaskBlockers` objects.",
    )


class ProjectPlanState(AgentState):
    raw_requirements: str

    features: Features
    # constraints: list[str]
    # stakeholders: list[str]

    tasks_by_feature: Annotated[dict[str, Tasks], reduce_dict]
    complexity_by_feature: Annotated[dict[str, ComplexityEstimate], reduce_dict]
    criteria_by_task: Annotated[dict[str, AcceptanceCriteria], reduce_dict]
    prompts_by_task: Annotated[dict[str, str], reduce_dict]
    execution_order: list[str]


def present_json_output(state: ProjectPlanState) -> dict[str, Any]:
    features = state.get("features")
    tasks_by_features = state.get("tasks_by_feature", {})
    criteria_by_task = state.get("criteria_by_task", {})
    prompts_by_task = state.get("prompts_by_task", {})

    tasks_by_id = {
        task.task_id: {
            "name": task.name,
            "description": task.description,
            "criteria": criteria_by_task[task.task_id].model_dump(),
            "prompt": prompts_by_task[task.task_id],
        }
        for _, tasks in tasks_by_features.items()
        for task in tasks.tasks
    }

    complexity_by_feature = state.get("complexity_by_feature", {})

    features_by_phase = defaultdict(list)
    for feature in features.features:
        features_by_phase[feature.phase].append(
            {
                "name": feature.name,
                "description": feature.description,
                "complexity": complexity_by_feature[feature.feature_id].model_dump(),
                "tasks": [
                    tasks_by_id[task.task_id]
                    for task in tasks_by_features[feature.feature_id].tasks
                ],
            }
        )

    out_object = {
        "raw_requirements": state["raw_requirements"],
        "phases": features_by_phase,
    }

    return {"messages": [AIMessage(json.dumps(out_object))]}
