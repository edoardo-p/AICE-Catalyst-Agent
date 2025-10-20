from typing import Annotated

from langchain.messages import ToolMessage
from langchain.tools import InjectedToolCallId, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from structures import (
    AcceptanceCriteria,
    ComplexityEstimate,
    Feature,
    Features,
    Task,
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
            (
                "system",
                "You are an expert agent capable of parsing raw requirements and "
                "generating a list of features which need to be implemented in order "
                "to achieve those requirements.",
            ),
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
                    f"Generated the following features:\n{features}",
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
            (
                "system",
                "You are an expert agent capable of breaking down a feature description into a list of tasks."
                "Generate a list of tasks for the feature provided by the user.",
            ),
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
                    f"Generated the following tasks:\n{tasks}",
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
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert software architect specializing in complexity estimation. 
Analyze features thoroughly considering technical complexity, dependencies, and unknowns. 
Consider:
- The complexity of the task breakdown
- Technical complexity (algorithms, data structures, architecture)
- Integration points and dependencies
- Unknown factors and research needed
- Testing requirements
- Edge cases and error handling
Simple: 1-3 days, straightforward implementation
Medium: 4-7 days, moderate complexity or integration
Complex: 8-15 days, significant complexity or multiple integrations
Very Complex: 16+ days, high complexity, research, or many unknowns.""",
            ),
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
                    f"Calculated complexities:\n{output}", tool_call_id=tool_call_id
                )
            ],
        }
    )


def classify_features_into_phase(
    features: Features, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Break down one or more software features into concrete implementation tasks.

    This tool modifies the `Features` object passed as input by assigning a
    concise and effective phase name to each feature in the list.

    Parameters
    ----------
    features : Features
        A list of features that need to be sorted into distinct phases.

    Returns
    -------
    features : Features
        The same list of features with phase fields populated correctly.
    """
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert agent capable of grouping feature descriptions into phases."
                "Do not alter the names or descriptions of features, simply modify the phase appropriately.",
            ),
            ("human", "\n".join(str(feature) for feature in features.features)),
        ]
    )

    new_features = (prompt | llm.with_structured_output(Features)).invoke({})

    return Command(
        update={
            "features": new_features,
            "messages": [
                ToolMessage(
                    f"Updated feature phases:\n{new_features}",
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
            (
                "system",
                "You are a senior software engineer responsible for defining clear, "
                "verifiable acceptance criteria for development tasks. Each criterion "
                "should describe an observable condition that determines when the task "
                "can be considered complete. Only include integration tests if this task "
                "has to integrate with some other task or feature.",
            ),
            ("human", f"Generate a list of acceptance criteria for this task:\n{task}"),
        ]
    )

    criteria = (prompt | llm.with_structured_output(AcceptanceCriteria)).invoke({})

    return Command(
        update={
            "criteria_by_task": {task.task_id: criteria},
            "messages": [
                ToolMessage(
                    f"Generated task acceptance criteria:\n{criteria}",
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
            (
                "system",
                "You are an expert software engineer tasked with writing high-quality, "
                "context-aware prompts for AI coding assistants such as GitHub Copilot. "
                "Your goal is to produce a clear and focused prompt that helps the AI "
                "understand exactly what needs to be built and how it should behave."
                "Include:\n"
                "- What the task aims to achieve\n"
                "- Key inputs and outputs\n"
                "- Any relevant edge cases or constraints\n"
                "- Any architectural or stylistic considerations\n\n"
                "Return only the prompt, with no premise or feedback.",
            ),
            ("human", f"Write a prompt for this task:\n{task}"),
        ]
    )

    task_prompt = (prompt | llm).invoke({})

    return Command(
        update={
            "prompts_by_task": {task.task_id: prompt},
            "messages": [
                ToolMessage(
                    f"Generated task prompt:\n{task_prompt}", tool_call_id=tool_call_id
                )
            ],
        }
    )


def detect_dependencies(
    features: Features, tasks: Tasks, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    pass
