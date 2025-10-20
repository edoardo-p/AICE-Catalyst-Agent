from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import CompiledStateGraph, StateGraph

from control_flow import add_user_input_to_state, present_json_output, should_continue
from structures import ProjectPlanState
from tools import (
    classify_features_into_phase,
    create_task_acceptance_criteria,
    estimate_feature_complexity,
    generate_task_prompt_for_copilot,
    generate_tasks,
    parse_requirements,
)

load_dotenv()


def run_cli_agent(agent: CompiledStateGraph[ProjectPlanState]):
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
        middleware=[TodoListMiddleware()],
        tools=[
            parse_requirements,
            generate_tasks,
            estimate_feature_complexity,
            classify_features_into_phase,
            create_task_acceptance_criteria,
            generate_task_prompt_for_copilot,
        ],
        system_prompt="You are an expert agent capable of generating a "
        "structured project plan from raw business requirements. "
        "Starting from an empty state, slowly fill in missing fields by"
        "using the tools provided to you."
        "These are the state fields you need to fill:"
        "- features"
        "- tasks_by_feature"
        "- complexity_by_feature"
        "- criteria_by_task"
        "- prompts_by_task"
        # "- dependency_graph"
        "You also have access to a todo writer tool which you can "
        "use to plan which tools to call next.",
    )

    graph = StateGraph(ProjectPlanState)
    graph.add_node("agent", catalyst_agent)
    graph.add_node("add_reqs", add_user_input_to_state)
    graph.add_node("present_output", present_json_output)

    graph.set_entry_point("add_reqs")
    graph.add_edge("add_reqs", "agent")
    graph.add_edge("agent", "present_output")
    graph.set_finish_point("present_output")
    # graph.add_conditional_edges(
    #     "agent", should_continue, {True: "agent", False: "present_output"}
    # )
    agent = graph.compile()

    # run_cli_agent(catalyst_agent)

    with open(r"examples\\input2.txt") as f:
        question = f.read()

    output = agent.invoke({"messages": [("human", question)]})
    print(output["messages"][-1].content)


if __name__ == "__main__":
    main()
