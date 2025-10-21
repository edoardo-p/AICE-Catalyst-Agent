# AICE Catalyst Agent

This repository implements a small AI-driven "Catalyst" agent that converts free-form
business requirements into a structured, actionable project plan. The agent uses
LangChain-style tools and a state graph to iteratively parse requirements, generate
features and tasks, estimate complexity, and produce acceptance criteria and Copilot
prompts.

![Agent graph](agent_graph.png)

## Highlights

- State-driven agent built around a `ProjectPlanState` schema (see `src/structures.py`).
- Tools implemented in `src/tools.py` call an Azure LLM to produce structured outputs.
- A deterministic set of test tools is available in `src/test_tools.py` for offline testing.
- Entry point: `src/main.py` constructs the agent and runs the state graph.

## Quickstart (developer)

1. Create and activate a Python virtual environment. The repo was set up with uv.

2. Configure the following Azure credentials in a `.env` file:

```env
AZURE_OPENAI_API_KEY="your_api_key"
AZURE_OPENAI_ENDPOINT="your_endpoint"
```

3. Run the sample agent using an example input file from `/examples`, or come up with your own business requirement! The agent will print JSON output. The input can be changed in `src/main.py`.

## Using the test tools

For deterministic behavior during development or CI, use the test tools in
`src/test_tools.py`. They implement the same function signatures as the real tools
but return fixed outputs (consistent feature/task ids) so you can validate agent
flows without calling the LLM.

To temporarily run the agent with the test tools, open `src/main.py` and replace the import tools from `test_tools` instead of `tools`. The functions have the same names. This will exercise the state transitions end-to-end deterministically.

## Project structure

- `src/control_flow.py` — helper functions used to control the execution of the agent
- `src/main.py` — builds and runs the agent and state graph
- `src/prompts.py` — externalized prompts used by the tools and the agent
- `src/tools.py` — production tools that call the LLM (Azure)
- `src/test_tools.py` — deterministic test implementations of the same tools
- `src/structures.py` — Pydantic models describing the `ProjectPlanState` schema
- `examples/` — example input and expected output files

## Implemented tools

This project includes a set of tools (in `src/tools.py`) that the agent uses to
populate the `ProjectPlanState`. Each tool is also mirrored by a deterministic
test implementation in `src/test_tools.py` for offline testing.

List of implemented tools:

- `parse_requirements(requirements)` — parses free-form requirements into a `Features`
  object containing `Feature` entries (assigns `feature_id`s).
- `generate_tasks(feature)` — breaks a `Feature` into a `Tasks` list; assigns `task_id`s
  scoped by feature (e.g., `"0-0"`).
- `estimate_feature_complexity(feature, tasks)` — produces a `ComplexityEstimate` for a
  feature given its tasks (label + estimated days + risks + confidence).
- `create_task_acceptance_criteria(task)` — generates `AcceptanceCriteria` containing
  BDD-style `Scenario`s and unit/integration test specs for a `Task`.
- `generate_task_prompt_for_copilot(task)` — builds a concise prompt string that can
  be fed to a coding assistant to implement the task.
- `generate_execution_order(tasks)` / `detect_dependencies(features, tasks)` — (production)
  tools that infer blockers between tasks and compute a topological execution order.

These tools return `Command` objects with an `update` mapping that the StateGraph
merges into the shared agent state, and add a tool message with the result so the agent has access to them; the test tools return deterministic `Command`
updates so you can validate graph behavior without contacting the LLM.

## Notes

- The state schema uses `typing.Annotated` to associate reducer functions (see `reduce_dict`
  in `src/structures.py`) with specific dict fields. The LangGraph state graph recognizes
  these reducers when merging multiple updates for the same key.
