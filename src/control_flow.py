import json
from typing import Any

from structures import ProjectPlanState


def add_user_input_to_state(state: ProjectPlanState) -> dict[str, Any]:
    user_input = state.get("messages")[-1].content
    return {"raw_requirements": user_input}


def should_continue(state: ProjectPlanState) -> bool:
    features = state.get("features")
    tasks_by_feature = state.get("tasks_by_feature", {})

    if not features:
        return True

    for feature in features.features:
        if (
            not feature.phase
            or feature.feature_id not in tasks_by_feature
            or feature.feature_id not in state.get("complexity_by_feature", {})
        ):
            return True

    for tasks in tasks_by_feature.values():
        for task in tasks.tasks:
            if task.task_id not in state.get(
                "criteria_by_task", {}
            ) or task.task_id not in state.get("prompts_by_task", {}):
                return True

    # if not state.dependency_graph:
    #     return True

    return False


def present_json_output(state: ProjectPlanState) -> dict[str, Any]:
    features = state.get("features")
    tasks = state.get("tasks_by_feature", {})

    out_object = {
        feature.name: tasks.get(feature.feature_id, []) for feature in features.features
    }

    return {"messages": [("ai", json.dumps(out_object))]}


# def present_json_output(state: ProjectPlanState) -> dict[str, Any]:
#     features = state.get("features")
#     features_by_phase = defaultdict(list)
#     for feature in features.features:
#         features_by_phase[feature.phase].append(feature)
#     tasks_by_features = state.get("tasks_by_feature")

#     tasks_dict = {
#         task.task_id: task for tasks in tasks_by_features for task in tasks.tasks
#     }

#     tasks_obj = [
#         {
#             **task.model_dump(exclude={"task_id"}),
#         }
#         for task_id, task in tasks_dict.items()
#     ]

#     out_object = {
#         feature.name: tasks_dict.get(feature.feature_id, [])
#         for feature in features.features
#     }

#     return {"messages": [("ai", json.dumps(out_object))]}
