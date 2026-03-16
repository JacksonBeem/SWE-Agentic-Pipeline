# Operations Runbook

This is the practical run guide for day-to-day pipeline use.

Use it when you need to:

- run experiments
- resume interrupted runs
- rebuild derived outputs
- troubleshoot failures quickly

## Before You Run Anything

1. Use Python 3.11+.
2. Install dependencies:

```powershell
python -m pip install requests
python -m pip install openai-human-eval  # optional, only for --run-eval
```

3. Set core config in [config.py](/c:/VScode/pipeline/config.py):

- `OPENROUTER_API_KEY`
- `WORKFLOW_PROFILE`
- `ACTIVE_DATASET`

## Standard Full Pipeline Run

Run from `c:\VScode\pipeline`.

```powershell
python run_dataset_batch.py
python run_dataset_eval.py --run-bool-eval
python aggregate_results.py
python analyze_artifact_similarity_complexity.py
```

## Fast Smoke Run (Recommended First)

Use a tiny slice before expensive full runs:

```powershell
python run_dataset_batch.py --limit 5 --no-skip-existing
python run_dataset_eval.py --run-bool-eval
python aggregate_results.py
```

## Resume and Backfill Patterns

Resume interrupted run (default behavior):

```powershell
python run_dataset_batch.py --skip-existing
```

Backfill a specific index range:

```powershell
python run_dataset_batch.py --start-idx 50 --end-idx 99 --skip-existing
```

Regenerate a small range even if rows already exist:

```powershell
python run_dataset_batch.py --start-idx 0 --end-idx 9 --no-skip-existing
```

Reset prediction output file before rerun:

```powershell
python run_dataset_batch.py --reset-predictions
```

## Strategy and Mode Recipes

Monolithic mode:

```powershell
python run_dataset_batch.py --mode monolithic
```

Pipeline with verifier enabled:

```powershell
python run_dataset_batch.py --no-skip-verifier
```

Pipeline with verifier and security disabled:

```powershell
python run_dataset_batch.py --skip-verifier --skip-security
```

## Evaluation Recipes

Boolean evaluation only:

```powershell
python run_dataset_eval.py --run-bool-eval
```

HumanEval official evaluator:

```powershell
python run_dataset_eval.py --run-eval
```

Keep all converted samples (no latest-per-task dedupe):

```powershell
python run_dataset_eval.py --keep-all-samples --run-bool-eval
```

Increase per-task timeout for slow tasks:

```powershell
python run_dataset_eval.py --run-bool-eval --timeout-s 60
```

## Post-Run Validation Checklist

For the active dataset/strategy, verify these exist:

- `pipeline/logs/<dataset>/<strategy>/predictions.jsonl`
- `pipeline/logs/<dataset>/<strategy>/predictions_executable.jsonl`
- `pipeline/logs/<dataset>/<strategy>/boolean_results.jsonl`

Then inspect:

- `pipeline/aggregated/strategy_summary.csv`
- `pipeline/aggregated/accuracy_metrics.csv`
- `pipeline/aggregated/data_quality_report.csv`

Sanity expectations:

1. `missing_vs_expected` should be near zero for intended full runs.
2. `rows_with_boolean` should match converted coverage.
3. mismatch/parse-error counters in `data_quality_report.csv` should be low.

## Common Failure Cases

`OPENROUTER_API_KEY is empty`

- set key in [config.py](/c:/VScode/pipeline/config.py:7)

`Dataset not found` / `Input predictions not found`

- confirm `ACTIVE_DATASET` and default paths
- or pass explicit `--dataset`, `--input`, `--output`

Many `TimeoutError` rows in `boolean_results.jsonl`

- increase `--timeout-s`
- inspect task-level failures in `boolean_results.jsonl`

Many parse errors (`SyntaxError`) in boolean eval

- rerun conversion to normalize artifacts:

```powershell
python run_dataset_eval.py --run-bool-eval
```

Coverage mismatch in summary tables

- rebuild all derived artifacts in strict order:

```powershell
python run_dataset_eval.py --run-bool-eval
python aggregate_results.py
python analyze_artifact_similarity_complexity.py
```

## Recovery Order (When Outputs Drift)

If outputs look inconsistent, run this sequence:

1. regenerate/resume predictions
2. regenerate executable predictions + boolean results
3. regenerate aggregated CSVs
4. regenerate similarity/complexity outputs

This keeps every downstream file derived from current upstream data.

## Operational Notes

- `run_dataset_batch.py` appends predictions; reruns can create multiple samples per task.
- evaluator dedupes to latest sample per task by default.
- aggregation also reports latest-per-task snapshots.
- schema details are in [data_contracts_and_file_schemas.md](/c:/VScode/pipeline/docs/data_contracts_and_file_schemas.md).
