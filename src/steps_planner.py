from typing import Any

from langchain.agents.middleware import before_model
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from structures import ProjectPlanState


@before_model
def next_steps_hint_message(
    state: ProjectPlanState, runtime: Runtime[None]
) -> dict[str, Any]:
    messages = state.get("messages", [])
    if not any(message.type == "ai" for message in messages):
        return {}  # Let the agent do its thing the first time round

    field_statuses = _summarize_state(state)
    system_message_hint = SystemMessage(
        "Hint for next steps. "
        "This is the current state:\n"
        f"{field_statuses}.\n"
        "Fill in what is missing by using the tools provided."
    )

    return {"messages": [system_message_hint]}


def _summarize_state(state: ProjectPlanState) -> str:
    required_state_fields = (
        "features",
        "tasks_by_feature",
        "complexity_by_feature",
        "criteria_by_task",
        "prompts_by_task",
        # "dependency_graph",
    )

    field_status = {
        field: "missing" if field not in state or not state[field] else "filled"
        for field in required_state_fields
    }

    return "\n".join(f"{field}: {status}" for field, status in field_status.items())
