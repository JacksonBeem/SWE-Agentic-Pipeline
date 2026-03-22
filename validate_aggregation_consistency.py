from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate consistency between aggregated CSV outputs and source logs "
            "for dataset/strategy scopes, row counts, and blame invariants."
        )
    )
    p.add_argument("--logs-root", type=Path, default=SCRIPT_DIR / "logs")
    p.add_argument("--aggregated-dir", type=Path, default=SCRIPT_DIR / "aggregated")
    return p.parse_args()


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    txt = path.as_posix().lstrip("./")
    if txt.startswith("pipeline/"):
        return SCRIPT_DIR.parent / txt
    return SCRIPT_DIR / path


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scope_of(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("dataset", "") or "").strip().lower(),
        str(row.get("strategy", "") or "").strip(),
    )


def to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except ValueError:
        return default


def is_bool_text(value: str) -> bool:
    return str(value).strip() in {"True", "False"}


def build_log_scopes(logs_root: Path) -> set[tuple[str, str]]:
    scopes: set[tuple[str, str]] = set()
    if not logs_root.exists():
        return scopes
    for ds_dir in sorted([p for p in logs_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        dataset = ds_dir.name.lower()
        for strategy_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            runs_path = strategy_dir / "runs.csv"
            bool_path = strategy_dir / "boolean_results.jsonl"
            if runs_path.exists() or bool_path.exists():
                scopes.add((dataset, strategy_dir.name))
    return scopes


def main() -> int:
    args = parse_args()
    logs_root = resolve_input_path(args.logs_root)
    agg_dir = resolve_input_path(args.aggregated_dir)

    required_files = {
        "task_level": agg_dir / "task_level.csv",
        "strategy_summary": agg_dir / "strategy_summary.csv",
        "data_quality_report": agg_dir / "data_quality_report.csv",
        "blame_task_level": agg_dir / "blame_task_level.csv",
        "blame_summary": agg_dir / "blame_summary.csv",
    }

    errors: list[str] = []
    warnings: list[str] = []

    for name, path in required_files.items():
        if not path.exists():
            errors.append(f"missing aggregated file: {name} -> {path}")

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print(f"ERROR: {e}")
        return 1

    task_rows = load_csv(required_files["task_level"])
    summary_rows = load_csv(required_files["strategy_summary"])
    quality_rows = load_csv(required_files["data_quality_report"])
    blame_task_rows = load_csv(required_files["blame_task_level"])
    blame_summary_rows = load_csv(required_files["blame_summary"])

    task_scopes = {scope_of(r) for r in task_rows if all(scope_of(r))}
    summary_scopes = {scope_of(r) for r in summary_rows if all(scope_of(r))}
    quality_scopes = {scope_of(r) for r in quality_rows if all(scope_of(r))}
    blame_task_scopes = {scope_of(r) for r in blame_task_rows if all(scope_of(r))}
    blame_summary_scopes = {scope_of(r) for r in blame_summary_rows if all(scope_of(r))}
    log_scopes = build_log_scopes(logs_root)

    if task_scopes != summary_scopes:
        errors.append(f"scope mismatch: task_level vs strategy_summary -> {sorted(task_scopes ^ summary_scopes)}")
    if task_scopes != quality_scopes:
        errors.append(f"scope mismatch: task_level vs data_quality_report -> {sorted(task_scopes ^ quality_scopes)}")
    if blame_task_scopes != blame_summary_scopes:
        errors.append(
            f"scope mismatch: blame_task_level vs blame_summary -> {sorted(blame_task_scopes ^ blame_summary_scopes)}"
        )
    if task_scopes != log_scopes:
        errors.append(f"scope mismatch: aggregated scopes vs logs scopes -> {sorted(task_scopes ^ log_scopes)}")

    task_count_by_scope: dict[tuple[str, str], int] = defaultdict(int)
    for r in task_rows:
        k = scope_of(r)
        if all(k):
            task_count_by_scope[k] += 1

    for r in summary_rows:
        k = scope_of(r)
        if not all(k):
            continue
        row_count = to_int(r.get("rows"))
        if row_count != task_count_by_scope[k]:
            errors.append(f"strategy_summary.rows mismatch for {k}: {row_count} vs {task_count_by_scope[k]}")

    for r in quality_rows:
        k = scope_of(r)
        if not all(k):
            continue
        latest_snapshot_rows = to_int(r.get("latest_snapshot_rows"))
        if latest_snapshot_rows != task_count_by_scope[k]:
            errors.append(
                f"data_quality_report.latest_snapshot_rows mismatch for {k}: "
                f"{latest_snapshot_rows} vs {task_count_by_scope[k]}"
            )

    blame_count_by_scope: dict[tuple[str, str], int] = defaultdict(int)
    blame_known_critic_by_scope: dict[tuple[str, str], int] = defaultdict(int)
    for r in blame_task_rows:
        k = scope_of(r)
        if not all(k):
            continue
        blame_count_by_scope[k] += 1
        if str(r.get("critic_verdict_first_pass", "")).strip().upper() != "UNKNOWN":
            blame_known_critic_by_scope[k] += 1

    for r in blame_summary_rows:
        k = scope_of(r)
        if not all(k):
            continue
        row_count = to_int(r.get("rows"))
        known_count = to_int(r.get("critic_known_count"))
        if row_count != blame_count_by_scope[k]:
            errors.append(f"blame_summary.rows mismatch for {k}: {row_count} vs {blame_count_by_scope[k]}")
        if known_count != blame_known_critic_by_scope[k]:
            errors.append(
                f"blame_summary.critic_known_count mismatch for {k}: "
                f"{known_count} vs {blame_known_critic_by_scope[k]}"
            )

    for r in blame_task_rows:
        dataset, strategy = scope_of(r)
        task_id = str(r.get("task_id", "")).strip()
        key = f"{dataset}/{strategy}/{task_id}"

        strict_eligible = to_int(r.get("verifier_strict_eligible"))
        verifier_help = to_int(r.get("verifier_help"))
        verifier_harm = to_int(r.get("verifier_harm"))
        if verifier_help and not strict_eligible:
            errors.append(f"verifier_help without verifier_strict_eligible: {key}")
        if verifier_harm and not strict_eligible:
            errors.append(f"verifier_harm without verifier_strict_eligible: {key}")
        if verifier_help and verifier_harm:
            errors.append(f"verifier_help and verifier_harm both set: {key}")

        blame_basis = str(r.get("blame_basis", "")).strip()
        pre_verifier_exec_passed = str(r.get("pre_verifier_exec_passed", "")).strip()
        if blame_basis == "checkpoint_exec_only" and not is_bool_text(pre_verifier_exec_passed):
            errors.append(f"checkpoint_exec_only without boolean pre_verifier_exec_passed: {key}")
        if blame_basis == "checkpoint_missing" and pre_verifier_exec_passed:
            warnings.append(f"checkpoint_missing but pre_verifier_exec_passed is populated: {key}")

        critic_verdict = str(r.get("critic_verdict_first_pass", "")).strip().upper()
        exec_blame_from_critic = str(r.get("executor_blame_from_critic", "")).strip()
        if critic_verdict == "UNKNOWN" and exec_blame_from_critic:
            errors.append(f"unknown critic verdict with executor_blame_from_critic set: {key}")

    for r in quality_rows:
        k = scope_of(r)
        if not all(k):
            continue
        mismatch_cols = [
            "agent_call_dataset_mismatch_rows",
            "run_summary_dataset_mismatch_rows",
            "boolean_dataset_mismatch_rows",
        ]
        for c in mismatch_cols:
            v = to_int(r.get(c))
            if v > 0:
                warnings.append(f"non-zero {c} for {k}: {v}")

    print("Validation Summary")
    print(f"- scopes in logs: {len(log_scopes)}")
    print(f"- scopes in task_level: {len(task_scopes)}")
    print(f"- task_level rows: {len(task_rows)}")
    print(f"- blame_task_level rows: {len(blame_task_rows)}")
    print(f"- strategy_summary rows: {len(summary_rows)}")
    print(f"- blame_summary rows: {len(blame_summary_rows)}")

    if warnings:
        print(f"- warnings: {len(warnings)}")
        for w in warnings:
            print(f"WARNING: {w}")
    else:
        print("- warnings: 0")

    if errors:
        print(f"- errors: {len(errors)}")
        for e in errors:
            print(f"ERROR: {e}")
        print("VALIDATION FAILED")
        return 1

    print("- errors: 0")
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

