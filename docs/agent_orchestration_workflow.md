# Agent Orchestration Workflow

This document explains the multi-agent flow in plain language first, then maps each step to concrete function names.

Main code:

- [run_dataset_batch.py](/c:/VScode/pipeline/run_dataset_batch.py)
- [orchestrator.py](/c:/VScode/pipeline/orchestrator.py)
- [agents/base.py](/c:/VScode/pipeline/agents/base.py)
- [schemas.py](/c:/VScode/pipeline/schemas.py)
- [openrouter_client.py](/c:/VScode/pipeline/openrouter_client.py)

## Quick Mental Model

For each task, the pipeline does this:

1. Planner writes a structured plan.
2. Executor writes the artifact (code or patch).
3. Critic optionally validates against a test harness.
4. Verifier optionally decides ACCEPT or REJECT with a targeted repair request.
5. Executor may do one repair pass.
6. HumanEval tasks may get one extra syntax-only repair pass.
7. Everything is logged, then final artifact is returned.

## Where This Starts

Pipeline mode starts in [run_dataset_batch.main](/c:/VScode/pipeline/run_dataset_batch.py:181):

- builds `TaskInput`
- builds `PipelineOrchestrator`
- calls [PipelineOrchestrator.run_task](/c:/VScode/pipeline/orchestrator.py:191) per task

`TaskInput` contract ([schemas.py](/c:/VScode/pipeline/schemas.py:6)):

- `task_id`
- `problem`
- `repo_context` (optional)
- `test_harness` (optional)

## One Task, Step by Step (Function Names)

Per task, `run_task(...)` executes this sequence in [orchestrator.py](/c:/VScode/pipeline/orchestrator.py):

1. Planner stage
- `planner.build_messages(problem, repo_context)`
- `planner.run(problem, repo_context)`
- `_log_call_success(..., agent="Planner", ...)`

2. Executor stage
- `executor.build_messages(problem, planner_spec, repo_context)`
- `executor.run(...)`
- `_log_call_success(..., agent="Executor", ...)`
- `infer_artifact_mode(task)`
- `patch_format_summary(executor_output)`

3. Critic stage (harness-gated)
- runs only if `task.test_harness` exists
- for HumanEval-tagged configs, code is first composed with `compose_humaneval_executable_code(...)`
- `Critic.build_messages(code_artifact, test_harness)`
- `Critic.run(...)`
- `parse_critic_passfail(critic_output)`

4. Verifier gating
- compute `disagreement` from Critic/format signals
- compute `should_invoke_verifier` from `enable_verifier` + `trigger_policy`

5. Verifier stage (optional)
- `verifier.build_messages(critic_summary, disagreement, artifact_mode, format_checks)`
- `verifier.run(...)`
- `parse_verifier(verifier_output)`

6. Verifier-triggered repair (optional, single pass)
- condition: `REJECT` + `allow_single_repair` + non-empty repair request
- second Executor call with appended:
  - `REPAIR REQUEST FROM VERIFIER:\n<request>`
- for HumanEval with harness, Critic is re-run on repaired executable artifact

7. Final parse check
- build `final_executable_code`
- `ast.parse(final_executable_code)`

8. HumanEval syntax hardening (optional, single pass)
- if HumanEval-tagged config and parse fails:
  - third Executor call with `SYNTAX REPAIR REQUEST (HumanEval)`
  - parse repaired output again

9. Final run summary and return
- `logger.log_run(RunRow(...))`
- return dict with final artifact, parse status, verifier/repair flags, tokens, latency

## Communication Schema (Who Sends What)

All agents use the same structure: a 2-message chat payload.

- message 1: system instruction
- message 2: user payload with task context

### Shared transport path

All agent `run(...)` calls use [AgentBase.run](/c:/VScode/pipeline/agents/base.py:29):

1. build messages
2. send via [OpenRouterClient.chat](/c:/VScode/pipeline/openrouter_client.py:23)
3. clean output with `sanitize_output(...)`
4. enforce guardrails with `assert_no_prohibited(...)`
5. return `AgentResult` with text, raw output, messages, tokens, latency

### Planner message shape

Builder: [PlannerAgent.build_messages](/c:/VScode/pipeline/agents/planner.py:25)

User content:

```text
Problem:
<problem>

Repo context (if any):
<repo_context>
```

Expected output: non-code structured specification.

### Executor message shape

Builder: [ExecutorAgent.build_messages](/c:/VScode/pipeline/agents/executor.py:20)

User content includes:

- Problem block
- Planner spec block
- Repo context block
- explicit output instruction:
  - patch-only if repo context exists
  - code-only otherwise

Expected output:

- patch mode: unified diff (`diff --git ...`)
- code mode: raw code artifact

### Critic message shape

Builder: [CriticAgent.build_messages](/c:/VScode/pipeline/agents/critic.py:15)

User content:

```text
Test harness:
<harness>

Code artifact:
<artifact>
```

Expected output starts with `PASS` or `FAIL`.


### Verifier message shape

Builder: [VerifierAgent.build_messages](/c:/VScode/pipeline/agents/verifier.py:35)

User content includes:

- Critic summary
- disagreement flag
- artifact mode (`code` or `patch`)
- format checks JSON

Expected output:

- `ACCEPT`
- or `REJECT: <targeted repair request>`

## Decision Rules in Plain English

### 1) Artifact mode

Function: [infer_artifact_mode](/c:/VScode/pipeline/orchestrator.py:76)

- if `repo_context` contains `repo=` -> treat as patch task
- else -> treat as code task

### 2) Critic parsing

Function: [parse_critic_passfail](/c:/VScode/pipeline/orchestrator.py:23)

- first line starts with `PASS` -> pass
- first line starts with `FAIL` -> fail
- otherwise -> unknown

### 3) Verifier parsing

Function: [parse_verifier](/c:/VScode/pipeline/orchestrator.py:33)

- exact `ACCEPT`
- `REJECT: ...` (or multiline reject fallback parsing)



### 5) Patch format signal

Function: [patch_format_summary](/c:/VScode/pipeline/orchestrator.py:65)

Tracks:

- diff header presence
- file markers
- hunks
- markdown fences
- overall patch-likeness

### 6) Verifier invocation

Verifier runs only when:

- `enable_verifier` is true
- and trigger policy condition passes:
  - `always`, or
  - `disagreement` and disagreement is true

Disagreement is true when any of:

- Critic explicitly fails
- artifact contains markdown fence
- patch task does not look like a valid patch

## Logging and Traceability

### Per-agent logging

Success: [_log_call_success](/c:/VScode/pipeline/orchestrator.py:130)

Error: [_log_call_error](/c:/VScode/pipeline/orchestrator.py:160)

Both write `AGENT_CALL` rows via [CSVLogger.log_agent_call](/c:/VScode/pipeline/csv_logger.py:228) with:

- task/run identifiers
- serialized messages
- raw and clean outputs
- token usage and latency
- error text when applicable

### Per-task summary logging

At end of `run_task(...)`, [CSVLogger.log_run](/c:/VScode/pipeline/csv_logger.py:236) writes one `RUN_SUMMARY` row with:

- verifier invocation and decision
- repair flags
- final executable artifact
- parse errors
- cumulative tokens and latency
- per-agent token/latency rollups
- stored stage answers and stage error text

## Output of `run_task(...)`

Returned keys:

- `task_id`
- `verifier_invoked`
- `verifier_decision`
- `repair_attempted`
- `final_artifact`
- `final_executable_code`
- `parse_error_type`
- `parse_error_text`
- `total_tokens`
- `end_to_end_latency_s`

## Important Operational Notes

- If a stage fails, error is logged and exception is re-raised.
- `run_dataset_batch.py` catches task-level failures and continues with next task.
- Critic is intentionally skipped when no test harness is provided.
- In current batch pipeline mode, trigger policy is set to `"always"` ([run_dataset_batch.py](/c:/VScode/pipeline/run_dataset_batch.py:215)).
- `origin_stage` is initialized in orchestrator but currently not populated with a non-`None` value.

