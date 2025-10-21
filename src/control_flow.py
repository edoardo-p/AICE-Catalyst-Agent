import json
from collections import defaultdict
from typing import Any

from langchain_core.messages import AIMessage

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
    tasks_by_features = state.get("tasks_by_feature", {})
    criteria_by_task = state.get("criteria_by_task", {})
    prompts_by_task = state.get("prompts_by_task", {})

    tasks = {
        task.task_id: {
            "name": task.name,
            "description": task.description,
            "criteria": criteria_by_task[task.task_id],
            "prompt": prompts_by_task[task.task_id],
        }
        for _, tasks in tasks_by_features.items()
        for task in tasks.tasks
    }

    complexity_by_feature = state.get("complexity_by_feature", {})

    features_by_phase = defaultdict(list)
    for feature in features.features:
        features_by_phase[feature.phase].append(
            {
                "name": feature.name,
                "description": feature.description,
                "complexity": complexity_by_feature[feature.feature_id],
                "tasks": [
                    tasks[task.task_id]
                    for task in tasks_by_features[feature.feature_id].tasks
                ],
            }
        )

    # out_object = {}
    # for phase, features in features_by_phase.items():
    #     for feature in features:
    #         out_object[phase] = feature

    # out_object = {
    #     features: features
    #     for phase, features in features_by_phase.items()
    #     for feature in features
    # }

    return {"messages": [AIMessage(json.dumps(features_by_phase))]}
