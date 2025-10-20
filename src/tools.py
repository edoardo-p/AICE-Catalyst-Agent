from typing import Annotated

from langchain.messages import ToolMessage
from langchain.tools import InjectedToolCallId, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.types import Command

from structures import ComplexityEstimate, Feature, Features, Tasks


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

    output = (prompt | llm.with_structured_output(Features)).invoke({})

    return Command(
        update={
            "features": output,
            "messages": [ToolMessage(f"{output}", tool_call_id=tool_call_id)],
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
                "Break the feature down into tasks, and reference the feature id",
            ),
            (
                "human",
                f"{feature.name} (ID {feature.feature_id}): {feature.description}",
            ),
        ]
    )

    output = (prompt | llm.with_structured_output(Tasks)).invoke({})

    return Command(
        update={
            "tasks_by_feature": [output],
            "messages": [ToolMessage(f"{output}", tool_call_id=tool_call_id)],
        }
    )


@tool
def estimate_task_complexity(
    feature_description: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Estimate the development complexity of a proposed feature using an LLM-based software
    architecture analysis model.

    This function evaluates a feature description and returns a structured
    `ComplexityEstimate` object containing the projected effort, risk factors,
    and rationale. The analysis considers multiple dimensions of technical work,
    including algorithmic difficulty, integration points, testing scope, and unknowns.

    The LLM is instructed to classify tasks into one of four levels:
        - **Simple**: 1-3 days — straightforward implementation
        - **Medium**: 4-7 days — moderate complexity or integration
        - **Complex**: 8-15 days — significant complexity or multiple integrations
        - **Very Complex**: 16+ days — extensive research or high uncertainty

    Args:
        feature_description (str):
            A natural-language description of the feature or enhancement to assess.
            Example: "Implement real-time collaborative editing with conflict resolution."

    Returns:
        ComplexityEstimate:
            A structured object with the following fields:
              - **complexity_label (str)**: One of "Simple", "Medium", "Complex", or "Very Complex"
              - **estimated_days (int)**: Approximate number of development days required
              - **risks (list[str])**: Identified risks, dependencies, or unknowns
              - **confidence_level (float)**: Confidence in the estimation (0.0-1.0)
              - **reasoning (str)**: Concise justification for the assigned complexity rating

    Example:
        >>> result = estimate_task_complexity(
        ...     "Add AI-based image tagging with object detection and cloud storage integration"
        ... )
        >>> print(result.complexity_label)
        "Complex"
        >>> print(result.estimated_days)
        12
        >>> print(result.risks)
        ["Cloud API rate limits", "Model accuracy tuning", "Performance optimization"]
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
- Technical complexity (algorithms, data structures, architecture)
- Integration points and dependencies
- Unknown factors and research needed
- Testing requirements
- Edge cases and error handling
Simple: 1-3 days, straightforward implementation
Medium: 4-7 days, moderate complexity or integration
Complex: 8-15 days, significant complexity or multiple integrations
Very Complex: 16+ days, high complexity, research, or many unknowns
Always respond with valid JSON only.""",
            ),
            (
                "human",
                f"""Estimate the complexity of implementing the following feature:\nFeature: {feature_description}""",
            ),
        ]
    )

    return (prompt | llm.with_structured_output(ComplexityEstimate)).invoke({})
