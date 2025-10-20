import json
from typing import Any

from structures import ProjectPlanState


def add_user_input_to_state(state: ProjectPlanState) -> dict[str, Any]:
    user_input = state.get("messages")[-1].content
    return {"raw_requirements": user_input}


def should_continue(state: ProjectPlanState) -> bool:
    # # 1) Must have features parsed
    # if not state["features"]:
    #     return False

    # # 2) All features must have complexity estimates
    # for f in state.features:
    #     if f.id not in state.complexity_by_feature:
    #         return False

    # # 3) Must have tasks
    # if not state.tasks:
    #     return False

    # # 4) All tasks must have acceptance criteria and prompts
    # task_ids = [t.task_id for t in state.tasks]
    # if any(tid not in state.criteria_by_task for tid in task_ids):
    #     return False
    # if any(tid not in state.prompts_by_task for tid in task_ids):
    #     return False

    # # 5) Dependency graph must be present
    # if not state.dependency_graph:
    #     return False

    # return True

    return state.get("features") is None or state.get("tasks_by_feature") is None


def present_json_output(state: ProjectPlanState) -> dict[str, Any]:
    features = state.get("features")
    tasks_by_features = state.get("tasks_by_feature")

    tasks_dict = {
        tasks.feature_id: [task.model_dump() for task in tasks.tasks]
        for tasks in tasks_by_features
    }

    out_object = {
        feature.name: tasks_dict.get(feature.feature_id, [])
        for feature in features.data
    }

    return {"messages": [("ai", json.dumps(out_object))]}
