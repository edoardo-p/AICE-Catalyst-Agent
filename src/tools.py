from collections import defaultdict, deque
from typing import Annotated

from langchain.messages import ToolMessage
from langchain.tools import InjectedToolCallId, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from prompts import (
    CREATE_ACCEPTANCE_CRITERIA_LLM_PROMPT,
    DETECT_DEPENDENCIES_LLM_PROMPT,
    ESTIMATE_COMPLEXITY_LLM_PROMPT,
    GENERATE_COPILOT_PROMPTS_LLM_PROMPT,
    GENERATE_TASKS_LLM_PROMPT,
    PARSE_REQUIREMENTS_LLM_PROMPT,
)
from structures import (
    AcceptanceCriteria,
    ComplexityEstimate,
    DependencyGraph,
    Feature,
    Features,
    Task,
    TaskBlockers,
    Tasks,
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
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PARSE_REQUIREMENTS_LLM_PROMPT),
            ("human", requirements),
        ]
    )

    features: Features = (prompt | llm.with_structured_output(Features)).invoke({})
    for i, feature in enumerate(features.features):
        feature.feature_id = str(i)

    return Command(
        update={
            "features": features,
            "messages": [
                ToolMessage(
                    f"Filled in 'features' field:\n{features}",
                    tool_call_id=tool_call_id,
                )
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
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATE_TASKS_LLM_PROMPT),
            ("human", str(feature)),
        ]
    )

    tasks: Tasks = (prompt | llm.with_structured_output(Tasks)).invoke({})
    for i, task in enumerate(tasks.tasks):
        task.task_id = f"{feature.feature_id}-{i}"

    return Command(
        update={
            "tasks_by_feature": {feature.feature_id: tasks},
            "messages": [
                ToolMessage(
                    f"Filled in 'tasks_by_feature' field:\n{tasks}",
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
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ESTIMATE_COMPLEXITY_LLM_PROMPT),
            (
                "human",
                f"Feature: {feature}\nTasks:\n{'\n'.join(str(task) for task in tasks.tasks)}",
            ),
        ]
    )

    output = (prompt | llm.with_structured_output(ComplexityEstimate)).invoke({})

    return Command(
        update={
            "complexity_by_feature": {feature.feature_id: output},
            "messages": [
                ToolMessage(
                    f"Filled in 'complexity_by_feature' field:\n{output}",
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
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CREATE_ACCEPTANCE_CRITERIA_LLM_PROMPT),
            ("human", f"Generate a list of acceptance criteria for this task:\n{task}"),
        ]
    )

    criteria = (prompt | llm.with_structured_output(AcceptanceCriteria)).invoke({})

    return Command(
        update={
            "criteria_by_task": {task.task_id: criteria},
            "messages": [
                ToolMessage(
                    f"Filled in 'criteria_by_task' field:\n{criteria}",
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
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GENERATE_COPILOT_PROMPTS_LLM_PROMPT),
            ("human", f"Write a prompt for this task:\n{task}"),
        ]
    )

    task_prompt = (prompt | llm).invoke({})

    return Command(
        update={
            "prompts_by_task": {task.task_id: prompt},
            "messages": [
                ToolMessage(
                    f"Filled in 'prompts_by_task' field:\n{task_prompt}",
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
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DETECT_DEPENDENCIES_LLM_PROMPT),
            ("human", "\n".join(str(task) for task in tasks.tasks)),
        ]
    )

    tasks_and_blockers: DependencyGraph = (
        prompt | llm.with_structured_output(DependencyGraph)
    ).invoke({})

    blockers_by_task = {
        task.task_id: task for task in tasks_and_blockers.task_relationships
    }

    # Get task order and create list with task names
    execution_order_ids = _get_non_blocked_task_order(blockers_by_task)
    tasks_dict = {task.task_id: task for task in tasks.tasks}
    execution_order = [tasks_dict[task_id].name for task_id in execution_order_ids]

    return Command(
        update={
            "execution_order": execution_order,
            "messages": [
                ToolMessage(
                    f"Filled in 'prompts_by_task' field:\n{tasks_and_blockers}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


def _get_non_blocked_task_order(blockers_by_task: dict[str, TaskBlockers]) -> list[str]:
    n = len(blockers_by_task)
    if n == 0:
        return []

    completed = set()
    result = []

    blockers_count = {
        task_id: len(blocks.blocking_tasks)
        for task_id, blocks in blockers_by_task.items()
    }
    dependents = defaultdict(list)
    for task_id, blockers in blockers_by_task.items():
        for blocker in blockers.blocking_tasks:
            dependents[blocker].append(task_id)

    ready_queue = deque(
        task_id for task_id, num_blockers in blockers_count.items() if num_blockers
    )
    while ready_queue:
        current_task = ready_queue.popleft()
        completed.add(current_task)
        result.append(current_task)

        # Update all dependent tasks
        for dependent in dependents[current_task]:
            blockers_count[dependent] -= 1
            if blockers_count[dependent] == 0:
                ready_queue.append(dependent)

    if len(result) != n:
        return []  # Cycle detected

    return result
