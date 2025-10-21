from typing import Annotated

from langchain.messages import ToolMessage
from langchain.tools import InjectedToolCallId, tool
from langgraph.types import Command

from structures import (
    AcceptanceCriteria,
    ComplexityEstimate,
    Feature,
    Features,
    Scenario,
    Task,
    Tasks,
    UnitTest,
)


@tool
def parse_requirements(
    requirements: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Parse a set of natural-language software requirements into a structured list of features.

    This tool analyzes the provided `requirements` text (for example, a paragraph describing
    what a system or application should do) and identifies the distinct functional or
    non-functional features that must be implemented to satisfy those requirements.

    The tool uses a language model to interpret the intent and outputs a structured
    list of `Feature` objects, each representing an actionable system capability.

    Parameters
    ----------
    requirements : str
        Raw user-provided text describing desired functionality or system behavior.

    Returns
    -------
    features : Features
        A list of parsed `Feature` objects derived from the requirements.
    """
    f1 = Feature(
        feature_id="0",
        name="User Auth",
        description="Sign in/out and session management",
        phase="Core",
    )
    f2 = Feature(
        feature_id="1",
        name="Reports",
        description="Generate user activity reports",
        phase="Validate",
    )

    features = Features(features=[f1, f2])

    return Command(
        update={
            "features": features,
            "messages": [
                ToolMessage(f"parsed features {features}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool
def generate_tasks(
    feature: Feature, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Break down one or more software features into concrete implementation tasks.

    This tool takes a `Feature` object representing a high-level system
    capability and generates a list of `Task` objects. Each task includes
    a concise name and a clear description, defining the work needed to implement
    the feature.

    The output can be used to build project plans, execution steps, or dependency graphs.

    Parameters
    ----------
    feature : Feature
        A feature that needs to be decomposed into smaller, actionable tasks.

    Returns
    -------
    tasks : Tasks
        A list of tasks belonging to the input feature, paired with the feature_id.
    """
    t1 = Task(
        task_id=f"{feature.feature_id}-0",
        name=f"Implement {feature.name} API",
        description=f"API endpoints for {feature.name}",
    )
    t2 = Task(
        task_id=f"{feature.feature_id}-1",
        name=f"Add tests for {feature.name}",
        description=f"Unit tests and integration tests for {feature.name}",
    )

    tasks = Tasks(tasks=[t1, t2])

    return Command(
        update={
            "tasks_by_feature": {feature.feature_id: tasks},
            "messages": [
                ToolMessage(
                    f"generated tasks for {tasks}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def estimate_feature_complexity(
    feature: Feature, tasks: Tasks, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Estimate the implementation complexity of a feature based on its associated tasks.

    This tool analyzes a given `Feature` and its related `Tasks` to determine the overall
    complexity level required for implementation. The estimation considers technical,
    architectural, and organizational factors, including integration requirements,
    unknowns, and testing effort.

    The tool uses a language model to classify complexity into one of several categories
    (Simple, Medium, Complex, or Very Complex), with corresponding estimated time ranges.

    Parameters
    ----------
    feature : Feature
        The feature to be analyzed for implementation complexity.
    tasks : Tasks
        The collection of tasks associated with the feature. These provide context for
        understanding scope, dependencies, and effort.

    Returns
    -------
    complexity_by_feature : ComplexityEstimate
        An object containing the structured complexity estimation for the given feature.
    """
    estimate = ComplexityEstimate(
        complexity_label="Medium",
        estimated_days=5,
        risks=["third-party API unknowns"],
        confidence_level=0.7,
    )

    return Command(
        update={
            "complexity_by_feature": {feature.feature_id: estimate},
            "messages": [
                ToolMessage(
                    f"estimated complexity: {estimate}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def create_task_acceptance_criteria(
    task: Task, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Generate clear and testable acceptance criteria for a given software development task.

    This tool takes a single `Task` object and produces a structured list of acceptance
    criteria that describe the expected outcomes and completion conditions for that task.
    Acceptance criteria are expressed as concise, verifiable statements that can be used
    for validation, QA, and user acceptance testing.

    The tool uses a language model to analyze the task's name and description, and
    infer measurable success conditions based on functional intent, edge cases,
    and quality requirements.

    Parameters
    ----------
    task : Task
        The task for which to generate acceptance criteria. Should include a clear name
        and description outlining the purpose and scope of the work.

    Returns
    -------
    acceptance_criteria : AcceptanceCriteria
        A structured list of acceptance criteria generated for the given task.
    """
    ac = AcceptanceCriteria(
        scenarios=[
            Scenario(
                given="a user is authenticated",
                when="they call the endpoint",
                then="they receive a 200 response",
            )
        ],
        unit_tests=[
            UnitTest(
                test_name=f"test_{task.task_id}_basic",
                what_it_tests="basic happy path",
                expected_behavior="returns success",
            )
        ],
        integration_tests=None,
    )

    return Command(
        update={
            "criteria_by_task": {task.task_id: ac},
            "messages": [
                ToolMessage(
                    f"created criteria {ac}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def generate_task_prompt_for_copilot(
    task: Task, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Generate a detailed, context-aware prompt for an AI coding assistant (e.g., GitHub Copilot)
    to implement or refine the given task.

    This tool analyzes a `Task` object—containing its ID, name, and description—and
    produces a prompt that can be directly supplied to a coding model to accelerate
    implementation. The generated prompt provides relevant context, expected behavior,
    and implementation guidance derived from the task description.

    The goal is to ensure the coding assistant receives enough structured information
    to produce high-quality, aligned, and maintainable code outputs.

    Parameters
    ----------
    task : Task
        The task for which to generate a Copilot prompt. Should include a meaningful name
        and description explaining its intent and expected behavior.

    Returns
    -------
    copilot_prompt : CopilotPrompt
        A string prompt generated for use with coding assistants.
    """
    prompt = (
        f"Implement function for {task.name}. Ensure unit tests {task.task_id} pass."
    )

    return Command(
        update={
            "prompts_by_task": {task.task_id: prompt},
            "messages": [
                ToolMessage(
                    f"generated prompt: {prompt}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


@tool
def generate_execution_order(
    tasks: Tasks, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Determine the optimal execution order for a set of software development tasks based on inferred dependencies.

    This tool analyzes all provided `Task` objects to identify logical and technical dependencies
    (i.e., which tasks must be completed before others). It uses a language model to infer blockers
    between tasks, constructs a dependency graph, and computes a topological execution order that
    respects all dependency constraints.

    The resulting order ensures that each task appears only after all of its blockers have been
    completed, providing a dependency-safe sequence for execution or scheduling.

    Parameters
    ----------
    tasks : Tasks
        A collection of `Task` objects representing discrete units of work to be analyzed for
        dependency relationships and ordering.

    Returns
    -------
    execution_order : list[str]
        An ordered list of task names representing the recommended execution sequence
        (from first to last), ensuring all dependency constraints are satisfied.
    """
    execution_order = [task.name for task in tasks.tasks]
    return Command(
        update={
            "execution_order": execution_order,
            "messages": [
                ToolMessage(
                    f"generated execution order: {execution_order}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )
