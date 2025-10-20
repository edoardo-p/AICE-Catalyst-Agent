from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import CompiledStateGraph, StateGraph

from structures import ProjectPlan
from tools import estimate_task_complexity, generate_tasks, parse_requirements

load_dotenv()


def main():
    main_model = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.0,
    )

    catalyst_agent = create_agent(
        model=main_model,
        tools=[parse_requirements, generate_tasks, estimate_task_complexity],
        system_prompt="You are an expert agent capable of generating a "
        "structured project plan from raw business requirements. "
        "Use the tools provided to you to fill in missing portions of the state."
        "Answer purely with JSON.",
        state_schema=PlannerState,
    )

    with open("examples/1/input1.txt") as f:
        question = f.read()

    input = {"messages": [{"role": "user", "content": question}]}
    output = agent.invoke(input)
    print(output["messages"][-1].content)


if __name__ == "__main__":
    main()
