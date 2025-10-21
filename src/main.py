from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langgraph.graph.state import StateGraph

from control_flow import next_steps_hint_message, should_continue
from prompts import AGENT_SYSTEM_PROMPT
from structures import ProjectPlanState, present_json_output
from tools import (
    classify_features_into_phase,
    create_task_acceptance_criteria,
    estimate_feature_complexity,
    generate_execution_order,
    generate_task_prompt_for_copilot,
    generate_tasks,
    parse_requirements,
)

load_dotenv()


def add_user_input_to_state(state: ProjectPlanState) -> dict[str, Any]:
    user_input = state.get("messages")[-1].content
    return {"raw_requirements": user_input}


def create_project_planner_agent():
    main_model = AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2025-01-01-preview",
        temperature=0.0,
    )

    catalyst_agent = create_agent(
        model=main_model,
        middleware=[next_steps_hint_message],
        tools=[
            parse_requirements,
            generate_tasks,
            estimate_feature_complexity,
            # classify_features_into_phase, # unsure if this is really necessary; parse_requirements seems to populate the phase just fine
            create_task_acceptance_criteria,
            generate_task_prompt_for_copilot,
            generate_execution_order,
        ],
        system_prompt=AGENT_SYSTEM_PROMPT,
        state_schema=ProjectPlanState,
    )

    graph = StateGraph(ProjectPlanState)
    graph.add_node("agent", catalyst_agent)
    graph.add_node("add_reqs", add_user_input_to_state)
    graph.add_node("present_output", present_json_output)

    graph.set_entry_point("add_reqs")
    graph.add_edge("add_reqs", "agent")
    graph.set_finish_point("present_output")
    graph.add_conditional_edges(
        "agent", should_continue, {True: "agent", False: "present_output"}
    )
    return graph.compile()


def save_mermaid_diagram(agent):
    with open("agent_graph.png", "wb") as f:
        f.write(agent.get_graph(xray=True).draw_mermaid_png())


def main():
    agent = create_project_planner_agent()
    with open(r"examples\\input2.txt") as f:
        question = f.read()

    # mlflow.set_experiment("Project Planning Agent")
    # with mlflow.start_run():
    #     logged_model = mlflow.langchain.log_model(agent, name="ProjectPlanningAgent")
    # loaded_agent = mlflow.pyfunc.load_model(logged_model.model_uri)

    output = agent.invoke({"messages": [("human", question)]})
    print(output["messages"][-1].content)


if __name__ == "__main__":
    main()
