# Prediction Conversion and Evaluation Workflow

This document explains `run_dataset_eval.py` in plain language: how raw model output becomes executable code, and how local pass/fail results are produced.

Main code:

- [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py)
- [artifact_to_code.py](/c:/VScode/pipeline/utils/artifact_to_code.py)
- [dataset_utils.py](/c:/VScode/pipeline/dataset_utils.py)
- [io_utils.py](/c:/VScode/pipeline/io_utils.py)

## What This Script Does

It handles two jobs:

1. Convert `predictions.jsonl` into clean executable code (`predictions_executable.jsonl`).
2. Optionally run local boolean evaluation and write `boolean_results.jsonl`.

Optionally, for HumanEval, it can also invoke the official evaluator (`--run-eval`).

## Where It Fits in the Pipeline

Typical order:

1. `run_dataset_batch.py` generates raw predictions.
2. `run_dataset_eval.py` converts and evaluates predictions.
3. `aggregate_results.py` summarizes results.

## Inputs and Defaults

By default, paths are pulled from `config.py` using active dataset and workflow profile.

You can override with CLI flags:

- `--dataset`
- `--dataset-type {auto|humaneval|mbpp|bigcodebench}`
- `--input`
- `--output`
- `--completion-field` (default `completion`)
- `--keep-all-samples`
- `--run-bool-eval`
- `--bool-eval-output`
- `--timeout-s` (default 30)
- `--run-eval`

## End-to-End Flow (Function Map)

Inside [main](/c:/VScode/pipeline/run_dataset_eval.py:377), the flow is:

1. Validate `--dataset` and `--input` exist.
2. Resolve dataset type:
   - explicit via `--dataset-type`, or
   - auto via [detect_dataset_type](/c:/VScode/pipeline/dataset_utils.py:32).
3. Load dataset rows keyed by task ID with [load_dataset_rows_by_task](/c:/VScode/pipeline/run_dataset_eval.py:65).
4. Normalize predictions using [normalize_predictions](/c:/VScode/pipeline/run_dataset_eval.py:86).
5. Write executable predictions via [write_jsonl](/c:/VScode/pipeline/io_utils.py:17).
6. Optionally run HumanEval official evaluator via [run_humaneval_eval](/c:/VScode/pipeline/run_dataset_eval.py:291).
7. Optionally run local boolean evaluation via [evaluate_boolean](/c:/VScode/pipeline/run_dataset_eval.py:185).
8. Write boolean rows with `write_jsonl(...)`.

## Normalization: How Raw Artifacts Become Executable Code

Core function: [normalize_predictions](/c:/VScode/pipeline/run_dataset_eval.py:86)

### Task ID handling

- Reads `task_id` from each input row.
- For MBPP, numeric IDs are normalized to `MBPP/<id>`.
- Rows with missing or out-of-dataset IDs are skipped.

### Code extraction by dataset

HumanEval path:

- Uses [to_prompt_executable](/c:/VScode/pipeline/run_dataset_eval.py:81), which calls:
  - `compose_humaneval_executable_code(prompt, raw_completion)`
- This supports partial artifact fragments by composing under the prompt stub when needed.

MBPP and BigCodeBench path:

- Uses [to_executable_completion](/c:/VScode/pipeline/run_dataset_eval.py:74), which calls:
  - `extract_code_from_artifact_text(raw_completion)`
- Handles diffs, fenced output, and noisy wrappers.

Final cleanup:

- `strip_fences(...)` is applied before writing.

### Duplicate sample behavior

- default: keep latest row per task ID
- `--keep-all-samples`: preserve all converted rows

## Local Boolean Evaluation (Optional)

Core function: [evaluate_boolean](/c:/VScode/pipeline/run_dataset_eval.py:185)

Each task runs in a separate process (`multiprocessing.Process`) with timeout control.

### Pre-check before execution

Each completion is parsed with `ast.parse` first.

- parse failure -> immediate `passed=false` row (`eval_parse_error`++)

### Dataset-specific execution logic

Worker function: [_eval_worker](/c:/VScode/pipeline/run_dataset_eval.py:151)

HumanEval:

- `exec(prompt)`, `exec(completion)`, `exec(test)`
- resolve `candidate = ns[entry_point]`
- run `check(candidate)`

MBPP:

- `exec(completion)`
- `exec(mbpp_test_harness(task))` when harness exists

BigCodeBench:

- `exec(prompt)`, `exec(completion)`, `exec(test)`
- expects `TestCases` class
- runs unittest suite

### Timeout and exception behavior

If a worker exceeds timeout:

- process is terminated
- row is written with:
  - `passed=false`
  - `error_type=TimeoutError`

Any runtime exception becomes:

- `passed=false`
- `error_type=<ExceptionType>`
- `error=<message>`

## Output Files

`--output` (default `predictions_executable.jsonl`) rows contain:

- `task_id`
- `completion`
- optional passthrough fields (`model`, `model_name_or_path`)

`--bool-eval-output` (default `boolean_results.jsonl`) rows contain:

- `task_id`
- `passed`
- `error_type`
- `error`

## Reported Stats (What to Watch)

Conversion stats:

- `rows_converted`
- `rows_skipped_not_in_dataset`
- `rows_with_extraction_warning`
- `missing_task_count`

Eval stats:

- `eval_passed`
- `eval_failed`
- `eval_timeout`
- `eval_parse_error`
- `eval_missing_task`

## HumanEval Official Evaluator Path

With `--run-eval`, the script tries:

1. `evaluate_functional_correctness ...`
2. `python -m human_eval.evaluate_functional_correctness ...`

If both fail, it returns non-zero and prints manual-install guidance.

## Practical Notes

- This script uses `exec` for local evaluation. Missing runtime dependencies in generated code will fail as runtime errors.
- Because generation appends predictions over time, "latest sample wins" can change evaluation results between runs.
- If you are comparing historical attempts, use `--keep-all-samples` and analyze explicitly instead of default dedupe behavior.
