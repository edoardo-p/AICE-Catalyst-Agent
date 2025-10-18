from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI

from structures import Requirements


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
    requirements_parser = prompt | llm.with_structured_output(Requirements)
    return requirements_parser.invoke({})
