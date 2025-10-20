from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from structures import ComplexityEstimate, Requirements, Task, Tasks


@tool
def parse_requirements(requirements: str) -> Requirements:
    """
    Parse unstructured business requirements into a structured `Requirements` model.

    This function uses an Azure-hosted GPT model to analyze raw textual input
    describing business or product requirements. It extracts and organizes
    relevant information into a structured `Requirements` object that includes:

      - **features**: A list of key functional capabilities or deliverables 
        mentioned or implied in the requirements text.
      - **constraints**: Any explicit or inferred business, technical, or time-related
        limitations (e.g., deadlines, budget, compliance rules, or technology stack constraints).
      - **success_criteria**: Quantifiable or qualitative conditions that define
        what a “successful” initial product (e.g., MVP) should achieve.

    The model is instructed to infer missing but reasonable elements when the
    input requirements are incomplete or ambiguous.

    Args:
        requirements (str): 
            A freeform text input containing business or technical requirements 
            (e.g., stakeholder goals, desired features, timelines, or constraints).

    Returns:
        Requirements:
            A Pydantic model instance containing structured fields:
            - `features`: list[str]
            - `constraints`: list[str]
            - `success_criteria`: list[str]

    Example:
        >>> text = "We need a mobile app for booking fitness classes. It should work on iOS and Android, \
        allow payment via Stripe, and be ready within 3 months."
        >>> result = parse_requirements(text)
        >>> result.features
        ['Mobile booking system', 'Stripe payment integration', 'Cross-platform support']
        >>> result.constraints
        ['3-month delivery timeline']
        >>> result.success_criteria
        ['Users can book and pay for fitness classes successfully']
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
                "inferring structured starting information that can then be expanded upon. "
                "Your assumptions should not be of a quantitative nature.",
            ),
            ("human", requirements),
        ]
    )
    return (prompt | llm.with_structured_output(Requirements)).invoke({})


@tool
def generate_tasks(feature: str) -> list[Task]:
    """
    Generate a structured list of tasks from a high-level feature description.

    This function uses an Azure-hosted GPT model to break down a single feature
    into a set of actionable tasks. Each task is returned as a `Task` object
    containing a name, detailed description, AI prompt, and acceptance criteria.

    The model is instructed to:
        - Identify all relevant sub-tasks necessary to implement the feature
        - Provide clear, human-readable task names and descriptions
        - Optionally suggest AI tool prompts that could assist in completing the task

    Args:
        feature (str):
            A high-level description of the feature or functionality to implement.
            Example: "Allow users to reset their password via email verification."

    Returns:
        Tasks:
            A list of structured Task objects with the following fields:
                - name: concise name of the task
                - description: detailed explanation of the task
                - ai_tool_prompt: optional AI instruction for generating or completing the task
                - acceptance_criteria: list of conditions that define task completion

    Example:
        >>> tasks = generate_tasks("Add social login support for Google and Facebook")
        >>> for t in tasks.tasks:
        ...     print(t.name, t.description)
        ...
        "Implement OAuth flow" "Set up OAuth authentication endpoints for Google and Facebook"
        "Update UI for login" "Add buttons and UI logic for social login options"

    Notes:
        - Designed for integration into AI-assisted project planning pipelines.
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
                "You are an expert agent capable of breaking down a feature into "
                "a list of tasks. Just provide a name and a descriptions for each task.",
            ),
            ("human", feature),
        ]
    )
    return (prompt | llm.with_structured_output(Tasks)).invoke({})


@tool
def estimate_task_complexity(feature_description: str) -> ComplexityEstimate:
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
