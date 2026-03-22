# Artifact Similarity and Complexity Workflow

This document explains `analyze_artifact_similarity_complexity.py` in plain language: what it compares, how it computes metrics, and how to interpret the outputs.

Main code:

- [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py)

## What Question This Script Answers

Accuracy tells you if a solution passed tests.

This script answers different questions:

- How similar is generated code to canonical code?
- Is generated code structurally simpler or more complex?
- Are there systematic differences by dataset or strategy?

## Inputs

Default inputs:

- `--logs-root pipeline/logs`
- `--output-dir pipeline/aggregated`
- `--dataset-root pipeline/data`

Optional filters:

- `--dataset {humaneval|mbpp|bigcodebench}`
- `--strategy {agentic|agentic_plus_verifier|monolithic}`

## End-to-End Flow (Function Map)

Inside [main](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:713), the script does:

1. Load dataset rows by task with [load_dataset_rows](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:140).
2. For each dataset/strategy folder, build per-task analysis rows via [build_task_rows](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:398).
3. Build summary rows with [build_summary_rows](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:573).
4. Write per-task and summary outputs via [write_analysis_bundle](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:690).

## How It Picks Generated Artifacts

For each task, artifact source priority is:

1. `predictions_executable.jsonl` (preferred)
2. `runs.csv` `RUN_SUMMARY.final_executable_code` (fallback)

If neither exists:

- `artifact_found=0`
- similarity/complexity values become missing (`None`) where appropriate

## How Canonical Code Is Built

Function: [build_canonical_code](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:157)

Dataset rules:

- MBPP: use dataset `code` directly
- HumanEval: combine `prompt + canonical_solution` with `compose_prompt_executable_code(...)`
- BigCodeBench: combine `code_prompt/complete_prompt + canonical_solution` with `compose_prompt_executable_code(...)`

## Normalization Before Comparison

Function: [normalize_code](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:235)

Steps:

1. remove trailing analysis text (`Critic`, `FINAL_CODE`, etc.)
2. extract code from diffs/fences/wrappers
3. try parse-and-unparse normalization (`ast.parse` + `ast.unparse`)
4. if parse fails, keep extracted text and record parse error

Result per side:

- normalized code string
- parse error marker (`None`, `"empty"`, or exception text)

## Similarity Metrics

Two cosine similarities are calculated per task:

1. `lexical_cosine_similarity`
- built from lowercase lexical terms (identifier/number regex)

2. `token_cosine_similarity`
- built from Python token stream (`tokenize`)
- ignores whitespace/comment-style token noise
- maps docstrings to `DOCSTRING`
- falls back to regex tokenization if tokenization fails

Computation function:

- [cosine_similarity](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:282)

## Complexity Metrics

Function: [analyze_complexity](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:337)

Produces per code blob:

- size: chars, lines, non-empty lines
- parse status: `parse_ok`, `parse_error`
- AST counts:
  - node count
  - function count
  - branch count
  - loop count
  - call count
  - comprehension count
  - return count
  - max control nesting depth
  - `cyclomatic_proxy`

`cyclomatic_proxy` comes from [ComplexityVisitor](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:295) and increases with branch/loop/boolean complexity.

## Delta and Ratio Features

Per task, artifact is compared to canonical with:

- deltas (`artifact - canonical`) for line count, AST nodes, branches, loops, calls, nesting depth, cyclomatic proxy
- ratios for line count, AST node count, and cyclomatic proxy

Helpers:

- [safe_delta](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:386)
- [safe_ratio](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:392)

## Summary Rows

Function: [build_summary_rows](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py:573)

Two scopes are generated:

- `dataset_strategy` (one row per dataset + strategy)
- `dataset_all_strategies` (one row per dataset across all strategies)

Summary metrics include:

- artifact found/parse-ok rates
- pass/fail counts (from attached boolean results)
- average and median cosine similarities
- average structural complexity and average deltas vs canonical

## Output Files

Per dataset/strategy output directory:

- `artifact_similarity_complexity_per_task.csv`
- `artifact_similarity_complexity_per_task.jsonl`
- `artifact_similarity_complexity_summary.csv`

When no filters are used, combined files are also written directly under `pipeline/aggregated/`.

## How to Read the Results

If similarity is low and pass rate is high:

- model found different but valid solutions

If similarity is high and pass rate is low:

- model is mimicking canonical shape but making implementation mistakes

If complexity deltas are strongly positive:

- generated artifacts are generally more complex than canonical solutions

If complexity deltas are strongly negative:

- generated artifacts are generally simpler/shorter than canonical solutions

## Practical Notes

- This script does not run tests; it uses existing boolean results when present.
- All expected tasks are included, even if artifacts are missing, so coverage gaps are explicit.
- MBPP task IDs are normalized to `MBPP/<id>` where needed.

