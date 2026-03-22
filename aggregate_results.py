from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent

EXPECTED_TASKS: dict[str, int] = {
    "bigcodebench": 200,
    "mbpp": 200,
    "humaneval": 164,
}

def task_id_matches_dataset(dataset: str, task_id: str) -> bool:
    tid = (task_id or "").strip()
    ds = (dataset or "").strip().lower()
    if not tid:
        return False
    if ds == "mbpp":
        return tid.startswith("MBPP/") or tid.isdigit()
    if ds == "humaneval":
        return tid.startswith("HumanEval/")
    if ds == "bigcodebench":
        return tid.startswith("BigCodeBench/")
    return True

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate pipeline logs into normalized task-level, strategy summary, "
            "accuracy, and data-quality CSV outputs."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=SCRIPT_DIR / "logs",
        help="Root directory containing <dataset>/<strategy>/ artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "aggregated",
        help="Directory where aggregated CSV files will be written.",
    )
    return parser.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    txt = path.as_posix().lstrip("./")
    if txt.startswith("pipeline/"):
        return SCRIPT_DIR.parent / txt
    return SCRIPT_DIR / path


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def count_nonempty_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    low_val = ordered[low]
    high_val = ordered[high]
    return low_val + (high_val - low_val) * (rank - low)


def as_pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def wilson_ci_pct(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    phat = successes / total
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (phat + z2 / (2.0 * total)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
        / denom
    )
    low = max(0.0, center - margin) * 100.0
    high = min(1.0, center + margin) * 100.0
    return low, high


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out)


def write_scoped_csvs(
    output_dir: Path,
    file_name: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> int:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "") or "").strip().lower()
        strategy = str(row.get("strategy", "") or "").strip()
        if not dataset or not strategy:
            continue
        grouped.setdefault((dataset, strategy), []).append(row)

    for (dataset, strategy), scoped_rows in grouped.items():
        write_csv(output_dir / dataset / strategy / file_name, scoped_rows, fieldnames)

    return len(grouped)


def iter_strategy_dirs(logs_root: Path) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    if not logs_root.exists():
        return out

    for dataset_dir in sorted([p for p in logs_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        for strategy_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            out.append((dataset_dir.name.lower(), strategy_dir.name, strategy_dir))
    return out


def load_boolean_results(path: Path, dataset: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parse_error_rows = 0
    missing_task_id_rows = 0
    dataset_mismatch_rows = 0

    if not path.exists():
        return {
            "rows": rows,
            "by_task_id": {},
            "parse_error_rows": 0,
            "missing_task_id_rows": 0,
            "dataset_mismatch_rows": 0,
            "unique_task_ids": set(),
            "duplicate_rows": 0,
        }

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                parse_error_rows += 1
                continue

            task_id = str(obj.get("task_id", "") or "").strip()
            if not task_id:
                missing_task_id_rows += 1
                continue
            if not task_id_matches_dataset(dataset, task_id):
                dataset_mismatch_rows += 1
                continue

            rows.append(
                {
                    "task_id": task_id,
                    "passed": parse_bool(obj.get("passed")),
                    "error_type": str(obj.get("error_type", "") or "").strip() or None,
                    "error": str(obj.get("error", "") or "").strip() or None,
                    "line_no": line_no,
                }
            )

    by_task_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_task_id[row["task_id"]] = row

    unique_task_ids = set(by_task_id.keys())
    return {
        "rows": rows,
        "by_task_id": by_task_id,
        "parse_error_rows": parse_error_rows,
        "missing_task_id_rows": missing_task_id_rows,
        "dataset_mismatch_rows": dataset_mismatch_rows,
        "unique_task_ids": unique_task_ids,
        "duplicate_rows": len(rows) - len(unique_task_ids),
    }


def load_run_summaries(path: Path, dataset: str, strategy: str) -> dict[str, Any]:
    run_rows: list[dict[str, Any]] = []
    agent_calls: list[dict[str, Any]] = []
    total_rows = 0
    agent_call_rows = 0
    agent_call_dataset_mismatch_rows = 0
    unknown_event_rows = 0
    run_summary_missing_task_id_rows = 0
    run_summary_dataset_mismatch_rows = 0

    if not path.exists():
        return {
            "rows": run_rows,
            "agent_calls": agent_calls,
            "total_rows": 0,
            "agent_call_rows": 0,
            "agent_call_dataset_mismatch_rows": 0,
            "unknown_event_rows": 0,
            "run_summary_missing_task_id_rows": 0,
            "run_summary_dataset_mismatch_rows": 0,
            "distinct_run_ids": set(),
            "run_summary_unique_task_ids": set(),
        }

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            total_rows += 1
            event_type = str(row.get("event_type", "") or "").strip()
            if event_type == "AGENT_CALL":
                agent_call_rows += 1
                agent_name = str(row.get("agent", "") or "").strip()
                task_id_raw = str(row.get("task_id", "") or "").strip()
                if task_id_raw and not task_id_matches_dataset(dataset, task_id_raw):
                    agent_call_dataset_mismatch_rows += 1
                    continue
                agent_calls.append(
                    {
                        "dataset": dataset,
                        "strategy": strategy,
                        "run_id": str(row.get("run_id", "") or "").strip() or None,
                        "task_id": task_id_raw or None,
                        "pipeline_config": str(row.get("pipeline_config", "") or "").strip() or None,
                        "agent": agent_name or None,
                        "model": str(row.get("model", "") or "").strip() or None,
                        "ts_unix": parse_float(row.get("ts_unix")),
                        "prompt_tokens": parse_int(row.get("prompt_tokens")),
                        "completion_tokens": parse_int(row.get("completion_tokens")),
                        "total_tokens": parse_int(row.get("total_tokens")),
                        "latency_s": parse_float(row.get("latency_s")),
                        "error_text": str(row.get("error_text", "") or "").strip() or None,
                        "_input_order": idx,
                    }
                )
                continue
            if event_type != "RUN_SUMMARY":
                unknown_event_rows += 1
                continue

            task_id = str(row.get("task_id", "") or "").strip()
            if not task_id:
                run_summary_missing_task_id_rows += 1
            elif not task_id_matches_dataset(dataset, task_id):
                run_summary_dataset_mismatch_rows += 1
                continue

            run_rows.append(
                {
                    "dataset": dataset,
                    "strategy": strategy,
                    "task_id": task_id,
                    "run_id": str(row.get("run_id", "") or "").strip(),
                    "ts_unix": parse_float(row.get("ts_unix")),
                    "pipeline_config": str(row.get("pipeline_config", "") or "").strip() or None,
                    "trigger_policy": str(row.get("trigger_policy", "") or "").strip() or None,
                    "verifier_invoked": parse_int(row.get("verifier_invoked")),
                    "verifier_decision": str(row.get("verifier_decision", "") or "").strip() or None,
                    "repair_attempted": parse_int(row.get("repair_attempted")),
                    "origin_stage": str(row.get("origin_stage", "") or "").strip() or None,
                    "run_total_tokens": parse_int(row.get("run_total_tokens")),
                    "end_to_end_latency_s": parse_float(row.get("end_to_end_latency_s")),
                    "parse_error_type": str(row.get("parse_error_type", "") or "").strip() or None,
                    "parse_error_text": str(row.get("parse_error_text", "") or "").strip() or None,
                    "planner": str(row.get("planner", "") or "").strip() or None,
                    "executor": str(row.get("executor", "") or "").strip() or None,
                    "critic": str(row.get("critic", row.get("Critic", "")) or "").strip() or None,
                    "verifier": str(row.get("verifier", "") or "").strip() or None,
                    "planner_total_tokens": parse_int(row.get("planner_total_tokens")),
                    "executor_total_tokens": parse_int(row.get("executor_total_tokens")),
                    "critic_total_tokens": parse_int(row.get("critic_total_tokens")),
                    "verifier_total_tokens": parse_int(row.get("verifier_total_tokens")),
                    "planner_latency_s": parse_float(row.get("planner_latency_s")),
                    "executor_latency_s": parse_float(row.get("executor_latency_s")),
                    "critic_latency_s": parse_float(row.get("critic_latency_s")),
                    "verifier_latency_s": parse_float(row.get("verifier_latency_s")),
                    "planner_error": parse_int(row.get("planner_error")),
                    "executor_repair": parse_int(row.get("executor_repair")),
                    "executor_harm": parse_int(row.get("executor_harm")),
                    "verifier_repair": parse_int(row.get("verifier_repair")),
                    "verifier_harm": parse_int(row.get("verifier_harm")),
                    "_input_order": idx,
                }
            )

    task_to_indices: dict[str, list[int]] = {}
    for i, row in enumerate(run_rows):
        task_id = row["task_id"]
        if not task_id:
            continue
        task_to_indices.setdefault(task_id, []).append(i)

    for task_id, idxs in task_to_indices.items():
        sorted_idxs = sorted(
            idxs,
            key=lambda i: (
                run_rows[i]["ts_unix"] if run_rows[i]["ts_unix"] is not None else float("-inf"),
                run_rows[i]["_input_order"],
            ),
        )
        n = len(sorted_idxs)
        for rank, row_idx in enumerate(sorted_idxs, start=1):
            run_rows[row_idx]["run_index_for_task"] = rank
            run_rows[row_idx]["run_count_for_task"] = n
            run_rows[row_idx]["is_latest_for_task"] = 1 if rank == n else 0

    for row in run_rows:
        if "run_index_for_task" not in row:
            row["run_index_for_task"] = 1
            row["run_count_for_task"] = 1
            row["is_latest_for_task"] = 1

    distinct_run_ids = {
        row["run_id"] for row in run_rows if row.get("run_id")
    }
    run_summary_unique_task_ids = {
        row["task_id"] for row in run_rows if row.get("task_id")
    }

    return {
        "rows": run_rows,
        "agent_calls": agent_calls,
        "total_rows": total_rows,
        "agent_call_rows": agent_call_rows,
        "agent_call_dataset_mismatch_rows": agent_call_dataset_mismatch_rows,
        "unknown_event_rows": unknown_event_rows,
        "run_summary_missing_task_id_rows": run_summary_missing_task_id_rows,
        "run_summary_dataset_mismatch_rows": run_summary_dataset_mismatch_rows,
        "distinct_run_ids": distinct_run_ids,
        "run_summary_unique_task_ids": run_summary_unique_task_ids,
    }


def attach_boolean_results(
    run_rows: list[dict[str, Any]],
    bool_by_task_id: dict[str, dict[str, Any]],
    dataset: str,
    strategy: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for row in run_rows:
        task_id = row.get("task_id", "")
        bool_row = bool_by_task_id.get(task_id) if task_id else None
        if task_id:
            seen_task_ids.add(task_id)

        merged = dict(row)
        merged["has_boolean_result"] = 1 if bool_row else 0
        merged["passed"] = bool_row.get("passed") if bool_row else None
        merged["error_type"] = bool_row.get("error_type") if bool_row else None
        merged["error"] = bool_row.get("error") if bool_row else None
        out.append(merged)

    # Keep accuracy/evaluation visibility even when runs.csv is missing or incomplete.
    for task_id, bool_row in bool_by_task_id.items():
        if task_id in seen_task_ids:
            continue
        out.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "task_id": task_id,
                "run_id": None,
                "ts_unix": None,
                "pipeline_config": None,
                "trigger_policy": None,
                "verifier_invoked": None,
                "verifier_decision": None,
                "repair_attempted": None,
                "origin_stage": None,
                "run_total_tokens": None,
                "end_to_end_latency_s": None,
                "parse_error_type": None,
                "parse_error_text": None,
                "planner": None,
                "executor": None,
                "critic": None,
                "verifier": None,
                "planner_total_tokens": None,
                "executor_total_tokens": None,
                "critic_total_tokens": None,
                "verifier_total_tokens": None,
                "planner_latency_s": None,
                "executor_latency_s": None,
                "critic_latency_s": None,
                "verifier_latency_s": None,
                "planner_error": None,
                "executor_repair": None,
                "executor_harm": None,
                "verifier_repair": None,
                "verifier_harm": None,
                "run_index_for_task": 1,
                "run_count_for_task": 1,
                "is_latest_for_task": 1,
                "has_boolean_result": 1,
                "passed": bool_row.get("passed"),
                "error_type": bool_row.get("error_type"),
                "error": bool_row.get("error"),
            }
        )
    return out


def summarize_rows(
    dataset: str,
    strategy: str,
    rows: list[dict[str, Any]],
    view: str,
    expected_task_count: int | None,
) -> dict[str, Any]:
    if view == "latest_snapshot":
        scoped = [r for r in rows if int(r.get("is_latest_for_task", 0)) == 1]
    else:
        scoped = list(rows)

    unique_tasks = {r["task_id"] for r in scoped if r.get("task_id")}
    with_bool = [r for r in scoped if int(r.get("has_boolean_result", 0)) == 1]

    pass_count = sum(1 for r in with_bool if r.get("passed") is True)
    fail_count = sum(1 for r in with_bool if r.get("passed") is False)
    unknown_bool_count = sum(1 for r in with_bool if r.get("passed") is None)
    evaluated_count = len(with_bool)

    accuracy_ci95_low, accuracy_ci95_high = wilson_ci_pct(pass_count, evaluated_count)

    token_values = [float(r["run_total_tokens"]) for r in scoped if r.get("run_total_tokens") is not None]
    latency_values = [float(r["end_to_end_latency_s"]) for r in scoped if r.get("end_to_end_latency_s") is not None]

    verifier_values = [r["verifier_invoked"] for r in scoped if r.get("verifier_invoked") is not None]
    verifier_invoked_count = sum(1 for v in verifier_values if int(v) == 1)
    repair_values = [r["repair_attempted"] for r in scoped if r.get("repair_attempted") is not None]
    repair_attempted_count = sum(1 for v in repair_values if int(v) == 1)

    parse_error_rows = sum(1 for r in scoped if r.get("parse_error_type"))

    fail_error_counter = Counter(
        str(r.get("error_type"))
        for r in with_bool
        if r.get("passed") is False and r.get("error_type")
    )
    top_errors = fail_error_counter.most_common(3)

    expected_missing = None
    coverage_pct = None
    if expected_task_count is not None:
        expected_missing = max(expected_task_count - len(unique_tasks), 0)
        coverage_pct = as_pct(len(unique_tasks), expected_task_count)

    row: dict[str, Any] = {
        "dataset": dataset,
        "strategy": strategy,
        "view": view,
        "rows": len(scoped),
        "unique_task_ids": len(unique_tasks),
        "expected_task_count": expected_task_count,
        "missing_vs_expected": expected_missing,
        "coverage_vs_expected_pct": coverage_pct,
        "rows_with_boolean": len(with_bool),
        "rows_without_boolean": len(scoped) - len(with_bool),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unknown_bool_count": unknown_bool_count,
        "accuracy_eval_correct": pass_count,
        "accuracy_eval_total": evaluated_count,
        "accuracy_eval_pct": as_pct(pass_count, evaluated_count),
        "accuracy_eval_ci95_low_pct": accuracy_ci95_low,
        "accuracy_eval_ci95_high_pct": accuracy_ci95_high,
        "accuracy_expected_total": expected_task_count,
        "accuracy_expected_pct": (
            as_pct(pass_count, expected_task_count) if expected_task_count is not None else None
        ),
        "fail_rate_eval_pct": as_pct(fail_count, evaluated_count),
        "unknown_rate_eval_pct": as_pct(unknown_bool_count, evaluated_count),
        "pass_rate_pct": as_pct(pass_count, len(with_bool)),
        "run_total_tokens_avg": mean(token_values) if token_values else None,
        "run_total_tokens_median": percentile(token_values, 50.0),
        "run_total_tokens_p90": percentile(token_values, 90.0),
        "end_to_end_latency_s_avg": mean(latency_values) if latency_values else None,
        "end_to_end_latency_s_median": percentile(latency_values, 50.0),
        "end_to_end_latency_s_p90": percentile(latency_values, 90.0),
        "verifier_invoked_count": verifier_invoked_count,
        "verifier_invoked_rate_pct": as_pct(verifier_invoked_count, len(verifier_values)),
        "repair_attempted_count": repair_attempted_count,
        "repair_attempted_rate_pct": as_pct(repair_attempted_count, len(repair_values)),
        "parse_error_rows": parse_error_rows,
        "parse_error_rate_pct": as_pct(parse_error_rows, len(scoped)),
    }

    for i in range(3):
        if i < len(top_errors):
            row[f"top_error_type_{i + 1}"] = top_errors[i][0]
            row[f"top_error_count_{i + 1}"] = top_errors[i][1]
        else:
            row[f"top_error_type_{i + 1}"] = None
            row[f"top_error_count_{i + 1}"] = None

    return row


def summarize_accuracy_scope(
    dataset: str,
    strategy: str,
    scope: str,
    rows: list[dict[str, Any]],
    expected_task_count: int | None,
    run_id: str | None = None,
) -> dict[str, Any]:
    unique_tasks = {r["task_id"] for r in rows if r.get("task_id")}
    with_bool = [r for r in rows if int(r.get("has_boolean_result", 0)) == 1]

    pass_count = sum(1 for r in with_bool if r.get("passed") is True)
    fail_count = sum(1 for r in with_bool if r.get("passed") is False)
    unknown_count = sum(1 for r in with_bool if r.get("passed") is None)
    eval_total = len(with_bool)

    accuracy_eval_pct = as_pct(pass_count, eval_total)
    fail_rate_eval_pct = as_pct(fail_count, eval_total)
    unknown_rate_eval_pct = as_pct(unknown_count, eval_total)
    ci95_low, ci95_high = wilson_ci_pct(pass_count, eval_total)

    coverage_vs_expected_pct = None
    missing_vs_expected = None
    accuracy_expected_pct = None
    if expected_task_count is not None:
        coverage_vs_expected_pct = as_pct(len(unique_tasks), expected_task_count)
        missing_vs_expected = max(expected_task_count - len(unique_tasks), 0)
        accuracy_expected_pct = as_pct(pass_count, expected_task_count)

    return {
        "dataset": dataset,
        "strategy": strategy,
        "scope": scope,
        "run_id": run_id,
        "rows": len(rows),
        "unique_task_ids": len(unique_tasks),
        "rows_with_boolean": eval_total,
        "rows_without_boolean": len(rows) - eval_total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unknown_count": unknown_count,
        "accuracy_eval_pct": accuracy_eval_pct,
        "accuracy_eval_ci95_low_pct": ci95_low,
        "accuracy_eval_ci95_high_pct": ci95_high,
        "fail_rate_eval_pct": fail_rate_eval_pct,
        "unknown_rate_eval_pct": unknown_rate_eval_pct,
        "expected_task_count": expected_task_count,
        "accuracy_expected_pct": accuracy_expected_pct,
        "coverage_vs_expected_pct": coverage_vs_expected_pct,
        "missing_vs_expected": missing_vs_expected,
    }


def dedupe_agent_calls(
    agent_calls: list[dict[str, Any]],
    keep_run_task_keys: set[tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    """
    Keep one call per (dataset, strategy, run_id, task_id, pipeline_config, agent),
    restricted to the selected unique run-task rows.
    """
    latest_by_key: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    for row in agent_calls:
        dataset = str(row.get("dataset", "") or "").strip()
        strategy = str(row.get("strategy", "") or "").strip()
        run_id = str(row.get("run_id", "") or "").strip()
        task_id = str(row.get("task_id", "") or "").strip()
        pipeline_config = str(row.get("pipeline_config", "") or "").strip()
        agent = str(row.get("agent", "") or "").strip()

        if not dataset or not strategy or not run_id or not task_id or not agent:
            continue

        if keep_run_task_keys and (dataset, strategy, run_id, task_id) not in keep_run_task_keys:
            continue

        key = (dataset, strategy, run_id, task_id, pipeline_config, agent)
        prev = latest_by_key.get(key)
        if prev is None:
            latest_by_key[key] = row
            continue

        prev_ts = prev.get("ts_unix")
        cur_ts = row.get("ts_unix")
        prev_ord = int(prev.get("_input_order", 0) or 0)
        cur_ord = int(row.get("_input_order", 0) or 0)

        prev_sort = (
            float(prev_ts) if prev_ts is not None else float("-inf"),
            prev_ord,
        )
        cur_sort = (
            float(cur_ts) if cur_ts is not None else float("-inf"),
            cur_ord,
        )
        if cur_sort >= prev_sort:
            latest_by_key[key] = row

    deduped = list(latest_by_key.values())
    deduped.sort(
        key=lambda r: (
            str(r.get("dataset", "") or ""),
            str(r.get("strategy", "") or ""),
            str(r.get("pipeline_config", "") or ""),
            str(r.get("agent", "") or ""),
            str(r.get("run_id", "") or ""),
            str(r.get("task_id", "") or ""),
        )
    )
    return deduped


def summarize_token_stats(agent_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in agent_calls:
        dataset = str(row.get("dataset", "") or "").strip()
        strategy = str(row.get("strategy", "") or "").strip()
        pipeline_config = str(row.get("pipeline_config", "") or "").strip()
        agent = str(row.get("agent", "") or "").strip()
        key = (dataset, strategy, pipeline_config, agent)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        dataset, strategy, pipeline_config, agent = key
        rows = grouped[key]

        prompt_vals = [float(r["prompt_tokens"]) for r in rows if r.get("prompt_tokens") is not None]
        completion_vals = [float(r["completion_tokens"]) for r in rows if r.get("completion_tokens") is not None]
        total_vals = [float(r["total_tokens"]) for r in rows if r.get("total_tokens") is not None]
        latency_vals = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]

        run_ids = {r["run_id"] for r in rows if r.get("run_id")}
        task_ids = {r["task_id"] for r in rows if r.get("task_id")}
        calls_with_error = sum(1 for r in rows if r.get("error_text"))

        total_sum = sum(total_vals) if total_vals else 0.0
        run_count = len(run_ids)
        task_count = len(task_ids)

        out.append(
            {
                "dataset": dataset or None,
                "strategy": strategy or None,
                "pipeline_config": pipeline_config or None,
                "agent": agent or None,
                "call_count": len(rows),
                "call_count_with_prompt_tokens": len(prompt_vals),
                "call_count_with_completion_tokens": len(completion_vals),
                "call_count_with_total_tokens": len(total_vals),
                "calls_with_error": calls_with_error,
                "distinct_run_ids": run_count,
                "distinct_task_ids": task_count,
                "prompt_tokens_avg_per_call": mean(prompt_vals) if prompt_vals else None,
                "prompt_tokens_median_per_call": percentile(prompt_vals, 50.0),
                "prompt_tokens_p90_per_call": percentile(prompt_vals, 90.0),
                "completion_tokens_avg_per_call": mean(completion_vals) if completion_vals else None,
                "completion_tokens_median_per_call": percentile(completion_vals, 50.0),
                "completion_tokens_p90_per_call": percentile(completion_vals, 90.0),
                "total_tokens_avg_per_call": mean(total_vals) if total_vals else None,
                "total_tokens_median_per_call": percentile(total_vals, 50.0),
                "total_tokens_p90_per_call": percentile(total_vals, 90.0),
                "latency_s_avg_per_call": mean(latency_vals) if latency_vals else None,
                "latency_s_median_per_call": percentile(latency_vals, 50.0),
                "latency_s_p90_per_call": percentile(latency_vals, 90.0),
                "total_tokens_sum": total_sum if total_vals else None,
                "total_tokens_avg_per_run": (total_sum / run_count) if (total_vals and run_count > 0) else None,
                "total_tokens_avg_per_task": (total_sum / task_count) if (total_vals and task_count > 0) else None,
            }
        )

    return out


def build_quality_row(
    dataset: str,
    strategy: str,
    strategy_dir: Path,
    run_info: dict[str, Any],
    bool_info: dict[str, Any],
    task_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runs_path = strategy_dir / "runs.csv"
    bool_path = strategy_dir / "boolean_results.jsonl"
    pred_path = strategy_dir / "predictions.jsonl"
    pred_exec_path = strategy_dir / "predictions_executable.jsonl"

    run_summary_rows = run_info["rows"]
    run_unique = run_info["run_summary_unique_task_ids"]
    bool_unique = bool_info["unique_task_ids"]
    latest_rows = [
        r
        for r in task_rows
        if int(r.get("is_latest_for_task", 0)) == 1 and str(r.get("task_id", "") or "").strip()
    ]
    latest_unique = {r["task_id"] for r in latest_rows if r.get("task_id")}

    expected_task_count = EXPECTED_TASKS.get(dataset)
    latest_missing_vs_expected = None
    boolean_missing_vs_expected = None
    latest_coverage_vs_expected = None
    boolean_coverage_vs_expected = None
    if expected_task_count is not None:
        latest_missing_vs_expected = max(expected_task_count - len(latest_unique), 0)
        boolean_missing_vs_expected = max(expected_task_count - len(bool_unique), 0)
        latest_coverage_vs_expected = as_pct(len(latest_unique), expected_task_count)
        boolean_coverage_vs_expected = as_pct(len(bool_unique), expected_task_count)

    return {
        "dataset": dataset,
        "strategy": strategy,
        "runs_path": str(runs_path),
        "boolean_path": str(bool_path),
        "predictions_path": str(pred_path),
        "predictions_executable_path": str(pred_exec_path),
        "runs_file_exists": int(runs_path.exists()),
        "boolean_file_exists": int(bool_path.exists()),
        "predictions_file_exists": int(pred_path.exists()),
        "predictions_executable_file_exists": int(pred_exec_path.exists()),
        "runs_rows_total": run_info["total_rows"],
        "agent_call_rows": run_info["agent_call_rows"],
        "agent_call_dataset_mismatch_rows": run_info.get("agent_call_dataset_mismatch_rows", 0),
        "unknown_event_rows": run_info["unknown_event_rows"],
        "run_summary_rows": len(run_summary_rows),
        "run_summary_unique_task_ids": len(run_unique),
        "run_summary_duplicate_rows": len(run_summary_rows) - len(run_unique),
        "run_summary_missing_task_id_rows": run_info["run_summary_missing_task_id_rows"],
        "run_summary_dataset_mismatch_rows": run_info.get("run_summary_dataset_mismatch_rows", 0),
        "distinct_run_ids": len(run_info["distinct_run_ids"]),
        "run_id_list": ";".join(sorted(run_info["distinct_run_ids"])),
        "boolean_rows_total": len(bool_info["rows"]),
        "boolean_unique_task_ids": len(bool_unique),
        "boolean_duplicate_rows": bool_info["duplicate_rows"],
        "boolean_missing_task_id_rows": bool_info["missing_task_id_rows"],
        "boolean_dataset_mismatch_rows": bool_info.get("dataset_mismatch_rows", 0),
        "boolean_parse_error_rows": bool_info["parse_error_rows"],
        "run_summary_tasks_without_boolean": len(run_unique - bool_unique),
        "boolean_tasks_without_run_summary": len(bool_unique - run_unique),
        "predictions_rows": count_nonempty_lines(pred_path),
        "predictions_executable_rows": count_nonempty_lines(pred_exec_path),
        "expected_task_count": expected_task_count,
        "latest_snapshot_rows": len(latest_rows),
        "latest_snapshot_unique_task_ids": len(latest_unique),
        "latest_missing_vs_expected": latest_missing_vs_expected,
        "boolean_missing_vs_expected": boolean_missing_vs_expected,
        "latest_coverage_vs_expected_pct": latest_coverage_vs_expected,
        "boolean_coverage_vs_expected_pct": boolean_coverage_vs_expected,
    }


def round_numeric_fields(rows: list[dict[str, Any]], digits: int = 6) -> None:
    for row in rows:
        for key, val in list(row.items()):
            if isinstance(val, float):
                row[key] = round(val, digits)


def main() -> int:
    args = parse_args()
    logs_root: Path = resolve_input_path(args.logs_root)
    output_dir: Path = resolve_input_path(args.output_dir)

    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root not found: {logs_root.resolve()}")

    all_task_rows: list[dict[str, Any]] = []
    all_task_rows_all_runs: list[dict[str, Any]] = []
    all_agent_calls: list[dict[str, Any]] = []
    strategy_summary_rows: list[dict[str, Any]] = []
    strategy_summary_all_runs_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    accuracy_all_runs_rows: list[dict[str, Any]] = []
    selected_run_task_keys: set[tuple[str, str, str, str]] = set()

    strategy_dirs = iter_strategy_dirs(logs_root)
    if not strategy_dirs:
        raise ValueError(f"No dataset/strategy directories found under: {logs_root.resolve()}")

    for dataset, strategy, strategy_dir in strategy_dirs:
        runs_path = strategy_dir / "runs.csv"
        bool_path = strategy_dir / "boolean_results.jsonl"

        # Skip empty directories that have none of the expected artifacts.
        if not runs_path.exists() and not bool_path.exists():
            continue

        run_info = load_run_summaries(runs_path, dataset, strategy)
        bool_info = load_boolean_results(bool_path, dataset)
        all_agent_calls.extend(run_info["agent_calls"])
        task_rows = attach_boolean_results(
            run_rows=run_info["rows"],
            bool_by_task_id=bool_info["by_task_id"],
            dataset=dataset,
            strategy=strategy,
        )
        all_run_task_rows = [
            r
            for r in task_rows
            if str(r.get("task_id", "") or "").strip() and "_input_order" in r
        ]
        all_task_rows_all_runs.extend(all_run_task_rows)
        unique_task_rows = [
            r
            for r in task_rows
            if int(r.get("is_latest_for_task", 0)) == 1 and str(r.get("task_id", "") or "").strip()
        ]
        all_task_rows.extend(unique_task_rows)
        for row in unique_task_rows:
            run_id = str(row.get("run_id", "") or "").strip()
            task_id = str(row.get("task_id", "") or "").strip()
            if run_id and task_id:
                selected_run_task_keys.add((dataset, strategy, run_id, task_id))

        expected_task_count = EXPECTED_TASKS.get(dataset)
        strategy_summary_rows.append(
            summarize_rows(dataset, strategy, unique_task_rows, "unique_runs", expected_task_count)
        )
        strategy_summary_all_runs_rows.append(
            summarize_rows(dataset, strategy, all_run_task_rows, "all_runs", expected_task_count)
        )
        accuracy_rows.append(
            summarize_accuracy_scope(
                dataset=dataset,
                strategy=strategy,
                scope="unique_runs",
                rows=unique_task_rows,
                expected_task_count=expected_task_count,
            )
        )
        accuracy_all_runs_rows.append(
            summarize_accuracy_scope(
                dataset=dataset,
                strategy=strategy,
                scope="all_runs",
                rows=all_run_task_rows,
                expected_task_count=expected_task_count,
            )
        )
        by_run_id: dict[str, list[dict[str, Any]]] = {}
        for row in unique_task_rows:
            run_id = str(row.get("run_id", "") or "").strip()
            if not run_id:
                continue
            by_run_id.setdefault(run_id, []).append(row)
        for run_id in sorted(by_run_id.keys()):
            accuracy_rows.append(
                summarize_accuracy_scope(
                    dataset=dataset,
                    strategy=strategy,
                    scope="run_id",
                    rows=by_run_id[run_id],
                    expected_task_count=expected_task_count,
                    run_id=run_id,
                )
            )

        quality_rows.append(
            build_quality_row(dataset, strategy, strategy_dir, run_info, bool_info, task_rows)
        )

    deduped_agent_calls = dedupe_agent_calls(
        agent_calls=all_agent_calls,
        keep_run_task_keys=selected_run_task_keys,
    )
    deduped_agent_calls_all_runs = dedupe_agent_calls(
        agent_calls=all_agent_calls,
        keep_run_task_keys=set(),
    )
    token_stats_rows = summarize_token_stats(deduped_agent_calls)
    token_stats_all_runs_rows = summarize_token_stats(deduped_agent_calls_all_runs)

    task_fieldnames = [
        "dataset",
        "strategy",
        "task_id",
        "run_id",
        "ts_unix",
        "pipeline_config",
        "trigger_policy",
        "verifier_invoked",
        "verifier_decision",
        "repair_attempted",
        "origin_stage",
        "run_total_tokens",
        "end_to_end_latency_s",
        "parse_error_type",
        "parse_error_text",
        "planner",
        "executor",
        "critic",
        "verifier",
        "planner_total_tokens",
        "executor_total_tokens",
        "critic_total_tokens",
        "verifier_total_tokens",
        "planner_latency_s",
        "executor_latency_s",
        "critic_latency_s",
        "verifier_latency_s",
        "planner_error",
        "executor_repair",
        "executor_harm",
        "verifier_repair",
        "verifier_harm",
        "run_index_for_task",
        "run_count_for_task",
        "is_latest_for_task",
        "has_boolean_result",
        "passed",
        "error_type",
        "error",
    ]

    summary_fieldnames = [
        "dataset",
        "strategy",
        "view",
        "rows",
        "unique_task_ids",
        "expected_task_count",
        "missing_vs_expected",
        "coverage_vs_expected_pct",
        "rows_with_boolean",
        "rows_without_boolean",
        "pass_count",
        "fail_count",
        "unknown_bool_count",
        "accuracy_eval_correct",
        "accuracy_eval_total",
        "accuracy_eval_pct",
        "accuracy_eval_ci95_low_pct",
        "accuracy_eval_ci95_high_pct",
        "accuracy_expected_total",
        "accuracy_expected_pct",
        "fail_rate_eval_pct",
        "unknown_rate_eval_pct",
        "pass_rate_pct",
        "run_total_tokens_avg",
        "run_total_tokens_median",
        "run_total_tokens_p90",
        "end_to_end_latency_s_avg",
        "end_to_end_latency_s_median",
        "end_to_end_latency_s_p90",
        "verifier_invoked_count",
        "verifier_invoked_rate_pct",
        "repair_attempted_count",
        "repair_attempted_rate_pct",
        "parse_error_rows",
        "parse_error_rate_pct",
        "top_error_type_1",
        "top_error_count_1",
        "top_error_type_2",
        "top_error_count_2",
        "top_error_type_3",
        "top_error_count_3",
    ]

    quality_fieldnames = [
        "dataset",
        "strategy",
        "runs_path",
        "boolean_path",
        "predictions_path",
        "predictions_executable_path",
        "runs_file_exists",
        "boolean_file_exists",
        "predictions_file_exists",
        "predictions_executable_file_exists",
        "runs_rows_total",
        "agent_call_rows",
        "agent_call_dataset_mismatch_rows",
        "unknown_event_rows",
        "run_summary_rows",
        "run_summary_unique_task_ids",
        "run_summary_duplicate_rows",
        "run_summary_missing_task_id_rows",
        "run_summary_dataset_mismatch_rows",
        "distinct_run_ids",
        "run_id_list",
        "boolean_rows_total",
        "boolean_unique_task_ids",
        "boolean_duplicate_rows",
        "boolean_missing_task_id_rows",
        "boolean_dataset_mismatch_rows",
        "boolean_parse_error_rows",
        "run_summary_tasks_without_boolean",
        "boolean_tasks_without_run_summary",
        "predictions_rows",
        "predictions_executable_rows",
        "expected_task_count",
        "latest_snapshot_rows",
        "latest_snapshot_unique_task_ids",
        "latest_missing_vs_expected",
        "boolean_missing_vs_expected",
        "latest_coverage_vs_expected_pct",
        "boolean_coverage_vs_expected_pct",
    ]

    accuracy_fieldnames = [
        "dataset",
        "strategy",
        "scope",
        "run_id",
        "rows",
        "unique_task_ids",
        "rows_with_boolean",
        "rows_without_boolean",
        "pass_count",
        "fail_count",
        "unknown_count",
        "accuracy_eval_pct",
        "accuracy_eval_ci95_low_pct",
        "accuracy_eval_ci95_high_pct",
        "fail_rate_eval_pct",
        "unknown_rate_eval_pct",
        "expected_task_count",
        "accuracy_expected_pct",
        "coverage_vs_expected_pct",
        "missing_vs_expected",
    ]

    token_stats_fieldnames = [
        "dataset",
        "strategy",
        "pipeline_config",
        "agent",
        "call_count",
        "call_count_with_prompt_tokens",
        "call_count_with_completion_tokens",
        "call_count_with_total_tokens",
        "calls_with_error",
        "distinct_run_ids",
        "distinct_task_ids",
        "prompt_tokens_avg_per_call",
        "prompt_tokens_median_per_call",
        "prompt_tokens_p90_per_call",
        "completion_tokens_avg_per_call",
        "completion_tokens_median_per_call",
        "completion_tokens_p90_per_call",
        "total_tokens_avg_per_call",
        "total_tokens_median_per_call",
        "total_tokens_p90_per_call",
        "latency_s_avg_per_call",
        "latency_s_median_per_call",
        "latency_s_p90_per_call",
        "total_tokens_sum",
        "total_tokens_avg_per_run",
        "total_tokens_avg_per_task",
    ]

    round_numeric_fields(all_task_rows)
    round_numeric_fields(all_task_rows_all_runs)
    round_numeric_fields(strategy_summary_rows)
    round_numeric_fields(strategy_summary_all_runs_rows)
    round_numeric_fields(quality_rows)
    round_numeric_fields(accuracy_rows)
    round_numeric_fields(accuracy_all_runs_rows)
    round_numeric_fields(token_stats_rows)
    round_numeric_fields(token_stats_all_runs_rows)

    write_csv(output_dir / "task_level.csv", all_task_rows, task_fieldnames)
    write_csv(output_dir / "task_level_all_runs.csv", all_task_rows_all_runs, task_fieldnames)
    write_csv(output_dir / "strategy_summary.csv", strategy_summary_rows, summary_fieldnames)
    write_csv(
        output_dir / "strategy_summary_all_runs.csv",
        strategy_summary_all_runs_rows,
        summary_fieldnames,
    )
    write_csv(output_dir / "data_quality_report.csv", quality_rows, quality_fieldnames)
    write_csv(output_dir / "accuracy_metrics.csv", accuracy_rows, accuracy_fieldnames)
    write_csv(output_dir / "accuracy_metrics_all_runs.csv", accuracy_all_runs_rows, accuracy_fieldnames)
    write_csv(output_dir / "token_stats_by_agent.csv", token_stats_rows, token_stats_fieldnames)
    write_csv(
        output_dir / "token_stats_by_agent_all_runs.csv",
        token_stats_all_runs_rows,
        token_stats_fieldnames,
    )

    scoped_task = write_scoped_csvs(output_dir, "task_level.csv", all_task_rows, task_fieldnames)
    scoped_task_all_runs = write_scoped_csvs(
        output_dir,
        "task_level_all_runs.csv",
        all_task_rows_all_runs,
        task_fieldnames,
    )
    scoped_summary = write_scoped_csvs(output_dir, "strategy_summary.csv", strategy_summary_rows, summary_fieldnames)
    scoped_summary_all_runs = write_scoped_csvs(
        output_dir,
        "strategy_summary_all_runs.csv",
        strategy_summary_all_runs_rows,
        summary_fieldnames,
    )
    scoped_quality = write_scoped_csvs(output_dir, "data_quality_report.csv", quality_rows, quality_fieldnames)
    scoped_accuracy = write_scoped_csvs(output_dir, "accuracy_metrics.csv", accuracy_rows, accuracy_fieldnames)
    scoped_accuracy_all_runs = write_scoped_csvs(
        output_dir,
        "accuracy_metrics_all_runs.csv",
        accuracy_all_runs_rows,
        accuracy_fieldnames,
    )
    scoped_token = write_scoped_csvs(output_dir, "token_stats_by_agent.csv", token_stats_rows, token_stats_fieldnames)
    scoped_token_all_runs = write_scoped_csvs(
        output_dir,
        "token_stats_by_agent_all_runs.csv",
        token_stats_all_runs_rows,
        token_stats_fieldnames,
    )

    print(f"Wrote: {(output_dir / 'task_level.csv').as_posix()} ({len(all_task_rows)} rows)")
    print(f"Wrote: {(output_dir / 'task_level_all_runs.csv').as_posix()} ({len(all_task_rows_all_runs)} rows)")
    print(f"Wrote: {(output_dir / 'strategy_summary.csv').as_posix()} ({len(strategy_summary_rows)} rows)")
    print(
        f"Wrote: {(output_dir / 'strategy_summary_all_runs.csv').as_posix()} "
        f"({len(strategy_summary_all_runs_rows)} rows)"
    )
    print(f"Wrote: {(output_dir / 'data_quality_report.csv').as_posix()} ({len(quality_rows)} rows)")
    print(f"Wrote: {(output_dir / 'accuracy_metrics.csv').as_posix()} ({len(accuracy_rows)} rows)")
    print(f"Wrote: {(output_dir / 'accuracy_metrics_all_runs.csv').as_posix()} ({len(accuracy_all_runs_rows)} rows)")
    print(f"Wrote: {(output_dir / 'token_stats_by_agent.csv').as_posix()} ({len(token_stats_rows)} rows)")
    print(
        f"Wrote: {(output_dir / 'token_stats_by_agent_all_runs.csv').as_posix()} "
        f"({len(token_stats_all_runs_rows)} rows)"
    )
    print(f"Wrote scoped task_level.csv files for {scoped_task} dataset/strategy directories.")
    print(f"Wrote scoped task_level_all_runs.csv files for {scoped_task_all_runs} dataset/strategy directories.")
    print(f"Wrote scoped strategy_summary.csv files for {scoped_summary} dataset/strategy directories.")
    print(
        f"Wrote scoped strategy_summary_all_runs.csv files for {scoped_summary_all_runs} "
        "dataset/strategy directories."
    )
    print(f"Wrote scoped data_quality_report.csv files for {scoped_quality} dataset/strategy directories.")
    print(f"Wrote scoped accuracy_metrics.csv files for {scoped_accuracy} dataset/strategy directories.")
    print(
        f"Wrote scoped accuracy_metrics_all_runs.csv files for {scoped_accuracy_all_runs} "
        "dataset/strategy directories."
    )
    print(f"Wrote scoped token_stats_by_agent.csv files for {scoped_token} dataset/strategy directories.")
    print(
        f"Wrote scoped token_stats_by_agent_all_runs.csv files for {scoped_token_all_runs} "
        "dataset/strategy directories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

