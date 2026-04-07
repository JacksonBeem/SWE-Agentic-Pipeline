# Replication Guide

This guide is the paper-focused reproduction path for the `pipeline/` project.

The package is designed around two levels of reproducibility:

1. `Log-level replication`
   Rebuild every reported CSV and formatted table from the frozen raw artifacts in `pipeline/logs/`.
2. `Live rerun support`
   Re-execute generation against hosted APIs when credentials and model availability permit.

For publication and artifact evaluation, the primary reproducibility target should be `log-level replication`.

## Package Scope

The replication inputs are:

- `pipeline/data/*.jsonl`
- `pipeline/logs/<dataset>/<strategy>/...`

The derived outputs are:

- `pipeline/aggregated/*.csv`
- `pipeline/results_formatted*.csv`

If you are validating the package for the paper, you do **not** need to rerun model generation. Rebuilding the derived outputs from the archived logs is the authoritative path.

Authoritative vs auxiliary paths:

- authoritative: `data/`, `logs/`, `aggregated/`, `README.md`, `REPLICATION.md`
- auxiliary: `docs/`, `results_formatted*.csv`, the legacy nested `pipeline/` directory, and debug-only files such as `aggregated/final_metrics_report_debug.csv`

## Environment

Recommended:

- Python `3.11+`
- Windows PowerShell examples are used below because the project has been operated primarily from Windows

Install the minimal dependencies:

```powershell
cd c:\VScode\pipeline
python -m pip install -r requirements.txt
```

Optional:

- install `openai-human-eval` only if you plan to use the official HumanEval evaluator path from `run_dataset_eval.py --run-eval`

## API Key Setup

Generation scripts now read the OpenRouter key from either:

- the `OPENROUTER_API_KEY` environment variable, or
- `pipeline/.env`

Create a local env file from the example:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` and set:

```text
OPENROUTER_API_KEY=your-openrouter-api-key
```

If you are only reproducing reported metrics from archived logs, the API key is not required.

## Recommended Rebuild Order

Run from `c:\VScode\pipeline`:

```powershell
python aggregate_results.py
python aggregate_blame_data.py --include-non-verifier-strategies
python analyze_artifact_similarity_complexity.py
python build_final_metrics_report.py
python reformat_results.py
```

If your paper uses the pre-verifier checkpoint table:

```powershell
python build_pre_verifier_critic_metrics_report.py
python reformat_results.py aggregated\final_metrics_pre_verifier_critic_report.csv --output-path results_formatted_pre_verifier_critic.csv --strategy-filter agentic_plus_verifier --skip-strategy-files
```

## Validation

After rebuilding outputs, run:

```powershell
python validate_aggregation_consistency.py
python generate_manifest_checksums.py
```

Expected outcome:

- `VALIDATION PASSED`

This validator checks:

- dataset/strategy scope agreement between logs and aggregates
- row-count agreement between `task_level.csv`, `strategy_summary.csv`, and `data_quality_report.csv`
- blame summary consistency
- verifier help/harm invariants

The checksum generator writes `manifest_checksums.txt` for the current snapshot and excludes:

- `.git/`
- `.env`
- `__pycache__/`
- the legacy nested `pipeline/` directory
- `aggregated/final_metrics_report_debug.csv`

## Main Reproducibility Targets

The paper-facing report files are:

- `aggregated/final_metrics_report.csv`
- `results_formatted.csv`
- `results_formatted_monolithic.csv`
- `results_formatted_agentic_no_planner.csv`
- `results_formatted_agentic.csv`
- `results_formatted_agentic_plus_verifier.csv`
- `results_formatted_agentic_exec_gate_plus_verifier_loop3.csv`

The main audit files are:

- `aggregated/strategy_summary.csv`
- `aggregated/token_stats_by_agent.csv`
- `aggregated/blame_summary.csv`
- `aggregated/artifact_similarity_complexity_summary.csv`
- `aggregated/data_quality_report.csv`

## Live Reruns

Live reruns are supported through:

- `run_dataset_batch.py`
- `run_dataset_eval.py`

Example:

```powershell
python run_dataset_batch.py
python run_dataset_eval.py --run-bool-eval
```

Important caveat:

- live reruns depend on hosted model availability and provider behavior
- preview model names may change or disappear
- live rerun outputs may differ from archived logs even when the code is unchanged

For that reason, the archived logs should be treated as the canonical replication artifact for publication.

## What To Inspect If Numbers Drift

1. `pipeline/logs/<dataset>/<strategy>/boolean_results.jsonl`
2. `pipeline/logs/<dataset>/<strategy>/runs.csv`
3. `pipeline/aggregated/data_quality_report.csv`
4. `pipeline/aggregated/strategy_summary.csv`
5. `pipeline/aggregated/final_metrics_report.csv`

Typical causes of drift:

- a stale `final_metrics_report.csv` after raw logs changed
- multiple appended `run_id`s in `runs.csv`
- partial task coverage after interrupted reruns
- similarity analysis not yet rerun after boolean results changed

## Release Checklist

Before shipping the replication package:

1. Ensure `.env` is not committed with a real key.
2. Confirm `python validate_aggregation_consistency.py` passes.
3. Rebuild all aggregated outputs from the frozen logs once.
4. Rebuild formatted results after the final metrics report is refreshed.
5. Generate `manifest_checksums.txt` for the frozen release snapshot.
6. Keep the frozen `logs/` snapshot that matches the paper numbers.
