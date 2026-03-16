# Agentic LLM Evaluation Pipeline

This repository evaluates code-generation strategies across `HumanEval`, `MBPP`, and `BigCodeBench` using a shared Python pipeline, OpenRouter-backed model calls, local execution checks, and post-run aggregation scripts.

It supports three execution styles:

- `monolithic`: one model produces the final solution in a single call
- `agentic`: `Architect -> Developer -> QA`
- `agentic_plus_verifier`: `Architect -> Developer -> QA -> Verifier`, with a single repair loop when the verifier rejects

The repo already contains dataset files, run logs, converted predictions, boolean pass/fail outputs, and aggregated summaries for all three datasets and all three strategies.

## What This Project Does

The pipeline is designed to answer a practical question: how do different orchestration strategies affect code-generation quality, coverage, and token usage on benchmark datasets?

At a high level, the workflow is:

1. Load tasks from a dataset JSONL file.
2. Send each task through either a monolithic prompt or a multi-agent workflow.
3. Save raw predictions to JSONL.
4. Convert predictions into executable code artifacts.
5. Run local boolean evaluation where supported.
6. Aggregate per-task and per-strategy metrics into CSV summaries.
7. Optionally analyze structural similarity and complexity of generated artifacts against canonical solutions.

## Datasets

The repository is configured for these datasets:

- `humaneval` from `pipeline/data/human_eval.jsonl`
- `mbpp` from `pipeline/data/mbpp_sanitized_200.jsonl`
- `bigcodebench` from `pipeline/data/bigcodebench_200.jsonl`

Current aggregated summaries show full task coverage for the existing logged runs:

- `HumanEval`: 164 tasks
- `MBPP`: 200 tasks
- `BigCodeBench`: 200 tasks

## Pipeline Architecture

Detailed workflow docs:

- [Agent Orchestration Workflow](/c:/VScode/pipeline/docs/agent_orchestration_workflow.md)
- [Prediction Conversion and Evaluation Workflow](/c:/VScode/pipeline/docs/prediction_conversion_and_evaluation_workflow.md)
- [Aggregation Workflow](/c:/VScode/pipeline/docs/aggregation_workflow.md)
- [Artifact Similarity and Complexity Workflow](/c:/VScode/pipeline/docs/artifact_similarity_complexity_workflow.md)
- [Data Contracts and File Schemas](/c:/VScode/pipeline/docs/data_contracts_and_file_schemas.md)
- [Operations Runbook](/c:/VScode/pipeline/docs/operations_runbook.md)
- [Complete Workflow Diagram (BigCodeBench)](/c:/VScode/pipeline/docs/complete_workflow_diagram_bigcodebench.md)

### Agents

The multi-agent pipeline is implemented in [orchestrator.py](/c:/VScode/pipeline/orchestrator.py) and the agent prompt definitions live in [agents](/c:/VScode/pipeline/agents).

- `Architect`: writes a structured problem specification without code
- `Developer`: produces the code artifact
- `QA`: reviews the artifact against the provided test harness and emits `PASS` or `FAIL`
- `Verifier`: accepts or rejects based on QA/security summaries and can request one targeted repair
- `Security`: implemented, but disabled in the current default workflow settings

### Strategies

- `monolithic`
  Uses one model call with an internal Architect/Developer/QA instruction scaffold.
- `agentic`
  Runs explicit `Architect`, `Developer`, and `QA` stages.
- `agentic_plus_verifier`
  Adds a `Verifier` stage and supports one repair attempt if the verifier returns `REJECT`.

### Model Routing

Model selection is currently hard-coded in [config.py](/c:/VScode/pipeline/config.py):

- Architect: `google/gemini-3-pro-preview`
- Developer: `anthropic/claude-sonnet-4.5`
- Security: `openai/gpt-4o-mini`
- QA: `openai/gpt-5.1`
- Verifier: `google/gemini-3-pro-preview`

The pipeline calls models through OpenRouter via [openrouter_client.py](/c:/VScode/pipeline/openrouter_client.py).

## Repository Layout

```text
pipeline/
|- agents/                               # Agent prompt wrappers
|- data/                                 # Input benchmark datasets
|- logs/<dataset>/<strategy>/            # Main run artifacts
|- aggregated/                           # CSV summaries and analysis outputs
|- results/                              # Older or ad hoc experiment outputs
|- run_dataset_batch.py                  # Run generation over a dataset
|- run_dataset_eval.py                   # Convert predictions and execute eval
|- aggregate_results.py                  # Build summary CSVs
|- analyze_artifact_similarity_complexity.py
|- orchestrator.py                       # Multi-agent workflow
|- dataset_utils.py                      # Dataset detection and task extraction
|- csv_logger.py                         # Unified agent/run logging
|- openrouter_client.py                  # OpenRouter API client
`- config.py                             # Models, dataset defaults, workflow defaults
```

## Key Output Files

For each dataset/strategy pair, the pipeline writes:

- `predictions.jsonl`: raw generated outputs
- `predictions_executable.jsonl`: normalized executable code
- `boolean_results.jsonl`: per-task pass/fail execution results
- `runs.csv`: full trace log containing both agent-call rows and run-summary rows

Examples:

- [pipeline/logs/humaneval/agentic](/c:/VScode/pipeline/logs/humaneval/agentic)
- [pipeline/logs/mbpp/agentic_plus_verifier](/c:/VScode/pipeline/logs/mbpp/agentic_plus_verifier)
- [pipeline/logs/bigcodebench/monolithic](/c:/VScode/pipeline/logs/bigcodebench/monolithic)

Aggregated outputs are written to [aggregated](/c:/VScode/pipeline/aggregated):

- [accuracy_metrics.csv](/c:/VScode/pipeline/aggregated/accuracy_metrics.csv)
- [strategy_summary.csv](/c:/VScode/pipeline/aggregated/strategy_summary.csv)
- [task_level.csv](/c:/VScode/pipeline/aggregated/task_level.csv)
- [token_stats_by_agent.csv](/c:/VScode/pipeline/aggregated/token_stats_by_agent.csv)
- [data_quality_report.csv](/c:/VScode/pipeline/aggregated/data_quality_report.csv)

## Setup

### Requirements

The code itself has a very small direct dependency surface:

- Python 3.11+ recommended
- `requests`

Optional:

- `openai-human-eval` if you want to run the official HumanEval functional correctness evaluator via `--run-eval`

Minimal install:

```powershell
python -m pip install requests
```

If you want official HumanEval evaluation support:

```powershell
python -m pip install openai-human-eval
```

### API Key Configuration

Current repo behavior reads the OpenRouter API key directly from [config.py](/c:/VScode/pipeline/config.py). The `.env` file in the repo is not currently wired into the runtime.

Before running generation, set:

```python
OPENROUTER_API_KEY = "your-key-here"
```

in [config.py](/c:/VScode/pipeline/config.py).

You should also review:

- `WORKFLOW_PROFILE`
- `ACTIVE_DATASET`
- model IDs in [config.py](/c:/VScode/pipeline/config.py)

## Running The Pipeline

Run commands from the repo root:

```powershell
cd c:\VScode\pipeline
```

### 1. Generate Predictions

Run the active dataset with defaults from `config.py`:

```powershell
python run_dataset_batch.py
```

Useful options:

```powershell
python run_dataset_batch.py --mode monolithic
python run_dataset_batch.py --start-idx 0 --end-idx 24
python run_dataset_batch.py --limit 10
python run_dataset_batch.py --skip-existing
python run_dataset_batch.py --no-skip-verifier
```

Notes:

- default mode and output paths are derived from `WORKFLOW_PROFILE` and `ACTIVE_DATASET`
- `--skip-existing` defaults to `true`
- predictions are appended to dataset/strategy-specific JSONL files under `pipeline/logs`

### 2. Convert Predictions To Executable Code And Run Boolean Evaluation

Use the same dataset defaults from config:

```powershell
python run_dataset_eval.py --run-bool-eval
```

Explicit example for BigCodeBench:

```powershell
python run_dataset_eval.py `
  --dataset .\data\bigcodebench_200.jsonl `
  --dataset-type bigcodebench `
  --input .\logs\bigcodebench\agentic\predictions.jsonl `
  --output .\logs\bigcodebench\agentic\predictions_executable.jsonl `
  --bool-eval-output .\logs\bigcodebench\agentic\boolean_results.jsonl `
  --run-bool-eval
```

Optional HumanEval official evaluation:

```powershell
python run_dataset_eval.py --run-eval
```

### 3. Aggregate Results

```powershell
python aggregate_results.py
```

This script scans `pipeline/logs/<dataset>/<strategy>/` and writes project-level summary CSVs into `pipeline/aggregated/`.

### 4. Analyze Artifact Similarity And Complexity

```powershell
python analyze_artifact_similarity_complexity.py
```

This compares executable predictions against canonical dataset solutions and writes per-task plus summary outputs under `pipeline/aggregated/`.

## Evaluation Details

### `run_dataset_batch.py`

Main responsibilities:

- loads dataset tasks
- skips tasks already present in predictions output when requested
- runs either monolithic or multi-agent generation
- logs all calls and run summaries
- appends prediction artifacts to JSONL

### `run_dataset_eval.py`

Main responsibilities:

- converts raw artifacts into executable code
- normalizes diffs, fenced output, and partial code artifacts
- deduplicates to the latest sample per `task_id` by default
- runs local per-task execution for `HumanEval`, `MBPP`, and `BigCodeBench`
- writes `predictions_executable.jsonl` and `boolean_results.jsonl`

### `aggregate_results.py`

Main responsibilities:

- validates dataset/strategy directory structure
- reads `runs.csv`, `predictions.jsonl`, `predictions_executable.jsonl`, and `boolean_results.jsonl`
- computes coverage, pass/fail counts, confidence intervals, and data-quality checks
- writes cross-run and cross-strategy summary CSVs

## Existing Experiment Coverage

The repository already contains full coverage summaries for:

- `HumanEval`: `agentic`, `agentic_plus_verifier`, `monolithic`
- `MBPP`: `agentic`, `agentic_plus_verifier`, `monolithic`
- `BigCodeBench`: `agentic`, `agentic_plus_verifier`, `monolithic`

The current aggregated summaries in [strategy_summary.csv](/c:/VScode/pipeline/aggregated/strategy_summary.csv) show 100% logged coverage versus expected task counts for all nine dataset/strategy combinations.

## Known Constraints

- The OpenRouter API key is hard-coded in the current implementation rather than loaded from environment variables.
- There is no committed project-level `requirements.txt` for this repo yet.
- `SecurityAgent` exists, but the default workflow configuration keeps security disabled.
- Local execution for benchmark tasks may require third-party libraries depending on the generated code and benchmark task content, especially for `BigCodeBench`.
- The `results/` directory contains older or ad hoc outputs; `logs/` plus `aggregated/` represent the current standardized layout used by the main scripts.

## Recommended Workflow

For repeatable experiments:

1. Set model IDs, `WORKFLOW_PROFILE`, and `ACTIVE_DATASET` in [config.py](/c:/VScode/pipeline/config.py).
2. Run [run_dataset_batch.py](/c:/VScode/pipeline/run_dataset_batch.py) to generate predictions.
3. Run [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py) to create executable predictions and boolean results.
4. Run [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py) to update summary tables.
5. Run [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py) for deeper artifact-level analysis.

## GitHub Description

Short description suitable for the GitHub repository settings:

`Multi-agent LLM evaluation pipeline for HumanEval, MBPP, and BigCodeBench with OpenRouter orchestration, executable prediction conversion, boolean evaluation, and aggregated experiment analytics.`
