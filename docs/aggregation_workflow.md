# Aggregation Workflow

This document explains `aggregate_results.py` in plain language: what it reads, how it decides which rows are "current," and what each output file means.

Main code:

- [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py)

## What This Script Does

`aggregate_results.py` turns many raw run artifacts into a clean analytics snapshot.

It takes data from `pipeline/logs/<dataset>/<strategy>/` and writes summary tables to `pipeline/aggregated/`.

At a high level:

1. Find every dataset/strategy folder.
2. Read `runs.csv` and `boolean_results.jsonl`.
3. Merge run metadata with pass/fail results by `task_id`.
4. Keep the latest run row per task.
5. Compute summary metrics.
6. Write five aggregated CSV files.

## Inputs It Expects

Default CLI values:

- `--logs-root pipeline/logs`
- `--output-dir pipeline/aggregated`

Folder pattern discovered by [iter_strategy_dirs(...)](/c:/VScode/pipeline/aggregate_results.py:154):

- `pipeline/logs/<dataset>/<strategy>/`

A folder is skipped if both are missing:

- `runs.csv`
- `boolean_results.jsonl`

## Important Validation Rules

Task IDs are checked against dataset family using [task_id_matches_dataset(...)](/c:/VScode/pipeline/aggregate_results.py:22):

- `humaneval` expects `HumanEval/...`
- `mbpp` expects `MBPP/...` or numeric
- `bigcodebench` expects `BigCodeBench/...`

Expected task counts are fixed in `EXPECTED_TASKS`:

- HumanEval: 164
- MBPP: 200
- BigCodeBench: 200

These counts drive coverage fields like `missing_vs_expected`.

## Step-by-Step with Function Names

### 1) Read run logs

Function: [load_run_summaries(...)](/c:/VScode/pipeline/aggregate_results.py:227)

What it does:

- reads `runs.csv`
- separates `AGENT_CALL` and `RUN_SUMMARY`
- parses numeric fields safely
- drops dataset-mismatched rows
- counts data-quality issues (unknown event types, missing task IDs, mismatches)

It also computes per-task run ordering:

- `run_index_for_task`
- `run_count_for_task`
- `is_latest_for_task`

### 2) Read boolean results

Function: [load_boolean_results(...)](/c:/VScode/pipeline/aggregate_results.py:165)

What it does:

- reads `boolean_results.jsonl`
- validates `task_id`
- parses `passed`
- tracks parse/missing/mismatch counters
- builds `by_task_id` map (latest row wins for duplicates)

### 3) Merge run rows + boolean rows

Function: [attach_boolean_results(...)](/c:/VScode/pipeline/aggregate_results.py:381)

What it adds to each run row:

- `has_boolean_result`
- `passed`
- `error_type`
- `error`

Important fallback:

- if boolean result exists but run summary is missing, it creates a synthetic row so accuracy coverage is still visible.

### 4) Keep only latest rows per task

In `main()`, latest rows are selected with:

- `is_latest_for_task == 1`

These `unique_task_rows` become the canonical snapshot for most outputs.

### 5) Build metrics

Strategy summary:

- [summarize_rows(...)](/c:/VScode/pipeline/aggregate_results.py:452)
- produces counts, coverage, accuracy, CI, token/latency stats, verifier/repair rates, parse-error rates, top failure types

Accuracy tables:

- [summarize_accuracy_scope(...)](/c:/VScode/pipeline/aggregate_results.py:548)
- emits:
  - one `unique_runs` row per dataset/strategy
  - one `run_id` row per run ID

Token stats:

- [dedupe_agent_calls(...)](/c:/VScode/pipeline/aggregate_results.py:601)
- [summarize_token_stats(...)](/c:/VScode/pipeline/aggregate_results.py:660)
- dedupes to one latest call per `(dataset, strategy, run_id, task_id, pipeline_config, agent)` before token rollups

Data quality:

- [build_quality_row(...)](/c:/VScode/pipeline/aggregate_results.py:722)
- reports file existence, row counts, duplicates, mismatches, and cross-file consistency checks

## Output Files (What They Mean)

The script writes:

1. [task_level.csv](/c:/VScode/pipeline/aggregated/task_level.csv)
- One row per task in the latest snapshot.
- Best table for task-by-task debugging.

2. [strategy_summary.csv](/c:/VScode/pipeline/aggregated/strategy_summary.csv)
- One row per dataset/strategy with headline metrics.
- Best table for quick comparison across strategies.

3. [accuracy_metrics.csv](/c:/VScode/pipeline/aggregated/accuracy_metrics.csv)
- Accuracy at two scopes:
  - `unique_runs` (snapshot)
  - `run_id` (per run)

4. [token_stats_by_agent.csv](/c:/VScode/pipeline/aggregated/token_stats_by_agent.csv)
- Prompt/completion/total token stats and latency stats grouped by agent.

5. [data_quality_report.csv](/c:/VScode/pipeline/aggregated/data_quality_report.csv)
- Data-health diagnostics for each dataset/strategy folder.

All float values are rounded to 6 decimal places before writing.

## How to Read Key Metrics

Coverage:

- `coverage_vs_expected_pct`: percent of expected tasks present
- `missing_vs_expected`: expected minus present

Accuracy:

- `accuracy_eval_pct`: pass rate on rows with boolean results
- `accuracy_expected_pct`: pass count normalized by full expected benchmark size
- `accuracy_eval_ci95_low_pct` / `high`: Wilson 95% confidence interval

Reliability signals:

- `rows_without_boolean`: tasks missing execution outcome
- `parse_error_rows`: rows with parse issues in run summaries
- top error fields (`top_error_type_1`, etc.) for failure distribution

## Practical Notes

- This script is intentionally tolerant of partial data.
- If `runs.csv` is incomplete, boolean-only rows can still appear via synthetic fallback rows.
- If dataset sizes change, update `EXPECTED_TASKS` or coverage fields will be misleading.

## Typical Use

Run after generating predictions and boolean eval:

```powershell
python aggregate_results.py
```

Then inspect:

- `strategy_summary.csv` for quick comparison
- `data_quality_report.csv` to verify artifact health
- `task_level.csv` for root-cause debugging
