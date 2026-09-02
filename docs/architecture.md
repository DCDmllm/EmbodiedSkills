# Runtime architecture

EmbodiedSkills runs one shared blackboard through six stages: Observe, Plan,
Preflight, Execute, Verify, and Recover. The scheduler chooses a registered
skill; the phase policy validates that choice; the selected component executes
it; and the result is written back before the next decision.

The compact context is produced by `Blackboard.compact_context()`. It carries
the task, current observation and world state, active plan and subgoal, current
motion request, verification state, runtime diagnostics, and bounded recent
loop history. Training trajectory generation calls the same AgentLoop and the
same component prompts instead of reconstructing this context independently.

## Stage responsibilities

| Stage | Responsibility |
| --- | --- |
| Observe | Capture current views, perceive objects, localize task entities, and update world state. |
| Plan | Build or update the task plan, select the active subgoal, and allocate a bounded action budget. |
| Preflight | Refresh stale observations and validate environment, target, action-backend, and safety readiness. |
| Execute | Build a motion goal, ask pi0.5 for one bounded action chunk, and execute it in the environment. |
| Verify | Capture post-action evidence and decide whether the current subgoal progressed, finished, or needs another attempt. |
| Recover | Convert execution or verification failures into a concrete retry, re-observation, or replanning request. |

Local verification advances the plan. Only `env_adapter.task_status()` reports
whole-task success. This separation keeps RoboTwin, RMBench, and LIBERO native
evaluation semantics out of model prompts and component interfaces.

## Model routing

Components select a configured model key. A component can additionally route a
specific skill to a different key through `skill_models`. The vLLM launcher
loads every declared LoRA at server startup and rewrites those model keys to
resident served names. Switching stages therefore changes request routing, not
the loaded base model or process.

## Action boundary

The motion component sends only current images, state, the selected natural
language subtask, and an action budget to pi0.5. Candidate ids and metric
geometry remain optional metadata. The backend returns a typed bounded action
chunk; the environment adapter owns physical execution and native success
queries.
