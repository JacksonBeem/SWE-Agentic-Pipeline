# Data Contracts and File Schemas

This document explains the pipeline data contracts in a practical format: what each file is for, who writes it, who reads it, and which fields must exist.

## Why This Matters

Most pipeline scripts are loosely coupled through files, not direct function calls.

If one file schema changes unexpectedly, downstream scripts can silently misreport metrics. This doc helps prevent that.

## Global Conventions

- Files use UTF-8.
- JSONL means one JSON object per non-empty line.
- Extra keys are usually tolerated.
- Missing required keys are not tolerated.
- For duplicate task rows, most readers keep the latest row encountered.

## File-by-File Contracts

### 1) `predictions.jsonl`

Typical path:

- `pipeline/logs/<dataset>/<strategy>/predictions.jsonl`

Written by:

- [append_prediction](/c:/VScode/pipeline/io_utils.py:38)
- called from [run_dataset_batch.py](/c:/VScode/pipeline/run_dataset_batch.py:307)

Read by:

- [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py:86)
- [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py:786) for line-count quality checks

Required fields:

- `task_id: str`
- `completion: str`

Optional fields:

- `model: str`
- `model_name_or_path: str`

Notes:

- writer strips markdown code fences before saving `completion`
- MBPP IDs may be normalized downstream to `MBPP/<id>`

### 2) `predictions_executable.jsonl`

Typical path:

- `pipeline/logs/<dataset>/<strategy>/predictions_executable.jsonl`

Written by:

- [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py:419)

Read by:

- [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py:437) for boolean eval
- [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py:787) for quality checks
- [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:414)

Required fields:

- `task_id: str`
- `completion: str` (normalized executable code)

Optional fields:

- `model`
- `model_name_or_path`

Notes:

- default behavior keeps latest sample per task
- use `--keep-all-samples` in evaluator to retain all samples

### 3) `boolean_results.jsonl`

Typical path:

- `pipeline/logs/<dataset>/<strategy>/boolean_results.jsonl`

Written by:

- [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py:443)

Read by:

- [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py:165)
- [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:184)

Required fields:

- `task_id: str`
- `passed: bool-like`
- `error_type: str | null`
- `error: str | null`

Notes:

- parser accepts common bool forms (`true/false`, `1/0`, `yes/no`)
- duplicate task IDs collapse to latest row per task in map-based consumers

### 4) `runs.csv`

Writer location:

- [CSVLogger](/c:/VScode/pipeline/csv_logger.py)

Typical path used by aggregation:

- `pipeline/logs/<dataset>/<strategy>/runs.csv`

Read by:

- [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py:227)
- [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:195) as fallback source

Schema style:

- one union table for two event types
- `event_type` is required and must be:
  - `AGENT_CALL`
  - `RUN_SUMMARY`

Canonical column list:

- [CSVLogger.FIELDNAMES](/c:/VScode/pipeline/csv_logger.py:102)

#### `AGENT_CALL` minimum compatibility

- `event_type`
- `ts_unix`, `run_id`, `task_id`, `pipeline_config`
- `agent`, `model`
- `messages`, `raw_output`, `clean_output`
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `latency_s`
- `error_text` (nullable)

#### `RUN_SUMMARY` minimum compatibility

- `event_type`
- `ts_unix`, `run_id`, `task_id`, `pipeline_config`
- `trigger_policy`
- `verifier_invoked`, `verifier_decision`, `repair_attempted`
- `run_total_tokens`
- `end_to_end_latency_s`

Important optional fields consumed downstream:

- `final_executable_code`
- parse info (`parse_error_type`, `parse_error_text`)
- model IDs and per-agent token/latency fields
- helper flags (`architect_error`, `developer_repair`, etc.)

Notes:

- unknown `event_type` rows are ignored and counted as quality issues
- security calls are intentionally excluded from token rollups in aggregation

## Aggregated Output Contracts

Written by [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py:1079):

- `task_level.csv`
- `strategy_summary.csv`
- `accuracy_metrics.csv`
- `token_stats_by_agent.csv`
- `data_quality_report.csv`

These are analysis outputs (not inputs to generation). They are expected to be regenerated, not hand-edited.

High-level meaning:

- `task_level.csv`: one latest task row per dataset/strategy
- `strategy_summary.csv`: headline strategy metrics
- `accuracy_metrics.csv`: snapshot + per-run accuracy scopes
- `token_stats_by_agent.csv`: token/latency rollups by agent
- `data_quality_report.csv`: health and consistency checks

## Similarity/Complexity Output Contracts

Written by [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:690):

- `artifact_similarity_complexity_per_task.csv`
- `artifact_similarity_complexity_per_task.jsonl`
- `artifact_similarity_complexity_summary.csv`

High-level meaning:

- per-task files: structural and similarity metrics per task
- summary file: grouped averages/medians by scope

## Where to Find Exact Field Definitions

If you need exact current headers, use:

- [csv_logger.py](/c:/VScode/pipeline/csv_logger.py) for `runs.csv`
- [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py) `*_fieldnames` lists
- [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py) `task_fieldnames()` and `summary_fieldnames()`

## Safe Schema Change Rules

1. Prefer additive changes (new columns) over destructive changes (rename/remove).
2. Keep `task_id` conventions stable (especially MBPP normalization).
3. If changing `runs.csv`, update:
- `CSVLogger.FIELDNAMES`
- all readers in aggregation and similarity scripts
4. Rebuild and validate after schema edits:

```powershell
python run_dataset_eval.py --run-bool-eval
python aggregate_results.py
python analyze_artifact_similarity_complexity.py
```
