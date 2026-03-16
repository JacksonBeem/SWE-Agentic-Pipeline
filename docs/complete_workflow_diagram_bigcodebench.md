# Complete Workflow Diagram (BigCodeBench)

This diagram shows the full pipeline from generation to evaluation to aggregation to similarity/complexity analysis, using real column names from BigCodeBench artifacts.

## End-to-End Flow

```mermaid
flowchart TD
    A["Input Dataset
    pipeline/data/bigcodebench_200.jsonl
    key fields used:
    - task_id
    - complete_prompt / code_prompt
    - test"] --> B["Generation
    run_dataset_batch.py
    -> PipelineOrchestrator.run_task(...)"]

    B --> C["predictions.jsonl
    pipeline/logs/bigcodebench/<strategy>/predictions.jsonl
    columns:
    - task_id
    - completion
    - model"]

    B --> D["runs.csv
    pipeline/logs/bigcodebench/<strategy>/runs.csv
    columns (union schema):
    - event_type
    - run_id, task_id, pipeline_config
    - agent, model, messages, raw_output, clean_output
    - prompt_tokens, completion_tokens, total_tokens, latency_s
    - trigger_policy, verifier_invoked, verifier_decision, repair_attempted
    - run_total_tokens, end_to_end_latency_s
    - final_executable_code
    - parse_error_type, parse_error_text
    - architect_answer, developer_answer, qa_answer, verifier_answer
    - error_text / *_error_text"]

    C --> E["Conversion + Local Eval
    run_dataset_eval.py --dataset-type bigcodebench --run-bool-eval"]
    D --> E

    E --> F["predictions_executable.jsonl
    pipeline/logs/bigcodebench/<strategy>/predictions_executable.jsonl
    columns:
    - task_id
    - completion
    - model (optional passthrough)"]

    E --> G["boolean_results.jsonl
    pipeline/logs/bigcodebench/<strategy>/boolean_results.jsonl
    columns:
    - task_id
    - passed
    - error_type
    - error"]

    D --> H["Aggregation
    aggregate_results.py"]
    C --> H
    F --> H
    G --> H

    H --> I["task_level.csv
    columns:
    dataset,strategy,task_id,run_id,ts_unix,pipeline_config,trigger_policy,
    verifier_invoked,verifier_decision,repair_attempted,origin_stage,
    run_total_tokens,end_to_end_latency_s,parse_error_type,parse_error_text,
    architect,developer,qa,verifier,
    architect_total_tokens,developer_total_tokens,qa_total_tokens,verifier_total_tokens,
    architect_latency_s,developer_latency_s,qa_latency_s,verifier_latency_s,
    architect_error,developer_repair,developer_harm,verifier_repair,verifier_harm,
    run_index_for_task,run_count_for_task,is_latest_for_task,
    has_boolean_result,passed,error_type,error"]

    H --> J["strategy_summary.csv
    coverage, accuracy, CI95, token/latency stats,
    verifier/repair rates, parse error rates, top error types"]

    H --> K["accuracy_metrics.csv
    scope-based accuracy:
    - unique_runs
    - run_id"]

    H --> L["token_stats_by_agent.csv
    grouped by dataset,strategy,pipeline_config,agent:
    call counts + token/latency distribution stats"]

    H --> M["data_quality_report.csv
    file existence, row counts, duplicates,
    mismatches, expected-task coverage checks"]

    F --> N["Similarity + Complexity Analysis
    analyze_artifact_similarity_complexity.py"]
    G --> N
    A --> N
    D --> N

    N --> O["artifact_similarity_complexity_per_task.csv
    columns:
    dataset,strategy,task_id,expected_task_count,artifact_source,artifact_found,model,
    passed,error_type,canonical_parse_error,artifact_parse_error,
    canonical_* metrics, artifact_* metrics,
    lexical_cosine_similarity,token_cosine_similarity,
    *_delta fields, *_ratio fields"]

    N --> P["artifact_similarity_complexity_summary.csv
    scope,dataset,strategy,rows,artifact_found_count,artifact_found_rate_pct,
    artifact_parse_ok_count,artifact_parse_ok_rate_pct,pass_count,fail_count,
    lexical/token cosine avg+median,
    line/AST/cyclomatic/nesting aggregate comparisons"]
```

## BigCodeBench Artifact Headers (Current)

These are the actual current headers from your files.

### `predictions.jsonl` (row object keys)

- `task_id`
- `completion`
- `model`

### `predictions_executable.jsonl` (row object keys)

- `task_id`
- `completion`
- `model`

### `boolean_results.jsonl` (row object keys)

- `task_id`
- `passed`
- `error_type`
- `error`

### `runs.csv` header

- `event_type,ts_unix,run_id,task_id,pipeline_config,agent,model,messages,raw_output,clean_output,prompt_tokens,completion_tokens,total_tokens,latency_s,error_text,trigger_policy,verifier_invoked,verifier_decision,repair_attempted,final_correct,origin_stage,run_total_tokens,end_to_end_latency_s,final_executable_code,parse_error_type,parse_error_text,config,prompt,prompt_hash,architect,developer,qa,verifier,final_answer,architect_prompt_tokens,architect_completion_tokens,architect_total_tokens,architect_latency_s,developer_prompt_tokens,developer_completion_tokens,developer_total_tokens,developer_latency_s,qa_prompt_tokens,qa_completion_tokens,qa_total_tokens,qa_latency_s,verifier_prompt_tokens,verifier_completion_tokens,verifier_total_tokens,verifier_latency_s,architect_answer,developer_answer,qa_answer,verifier_answer,architect_error_text,developer_error_text,qa_error_text,verifier_error_text,correct_answer,architect_error,developer_repair,developer_harm,verifier_repair,verifier_harm`

## Script Mapping

- Generation/orchestration:
  - [run_dataset_batch.py](/c:/VScode/pipeline/run_dataset_batch.py)
  - [orchestrator.py](/c:/VScode/pipeline/orchestrator.py)
- Conversion/eval:
  - [run_dataset_eval.py](/c:/VScode/pipeline/run_dataset_eval.py)
- Aggregation:
  - [aggregate_results.py](/c:/VScode/pipeline/aggregate_results.py)
- Similarity/complexity:
  - [analyze_artifact_similarity_complexity.py](/c:/VScode/pipeline/analyze_artifact_similarity_complexity.py)
