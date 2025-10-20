import json
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import CompiledStateGraph, StateGraph

from structures import PlannerState, ProjectPlan
from tools import estimate_task_complexity, generate_tasks, parse_requirements

load_dotenv()


def should_continue(state: PlannerState) -> bool:
    return None in state["data"].model_dump().values()


def present_json_output(state: PlannerState) -> dict[str, Any]:
    plan_as_dict = state["data"].model_dump()
    return {"messages": [("ai", plan_as_dict)]}


def run_cli_agent(agent: CompiledStateGraph[ProjectPlan]):
    last_message_idx = 0
    while True:
        question = input("> ")
        if question in ("exit", "quit", "q"):
            break

        answer = agent.invoke({"messages": [("human", question)]})
        last_message_idx += 1
        messages = answer["messages"][last_message_idx:]
        print()
        for new_message in messages:
            if new_message.type == "ai":
                print(new_message.type)
                to_print = (
                    new_message.content
                    if new_message.content
                    else new_message.lc_attributes
                )
                print(to_print)
            elif new_message.type == "tool":
                print(new_message.name)
                print(new_message.content)
            else:
                print(new_message.type)
                print(new_message.content)
            last_message_idx += 1
            print()
        print()


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

    graph = StateGraph(ProjectPlan)
    graph.add_node("agent", catalyst_agent)
    graph.add_node("present_output", present_json_output)
    graph.set_entry_point("agent")
    graph.set_finish_point("present_output")
    graph.add_conditional_edges(
        "agent", should_continue, {True: "agent", False: "present_output"}
    )

    # run_cli_agent(catalyst_agent)

    with open(r"examples\\input1.txt") as f:
        question = f.read()

    output = catalyst_agent.invoke(
        {"messages": [("human", question)], "data": ProjectPlan()}
    )
    print(output["messages"][-1].content)


if __name__ == "__main__":
    main()
