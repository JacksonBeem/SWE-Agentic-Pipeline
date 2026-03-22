from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent

DATASETS = ("mbpp", "bigcodebench", "humaneval")
DEFAULT_STRATEGIES = ("agentic_plus_verifier", "agentic_no_planner_plus_verifier")
CRITIC_AGENT_ALIASES = {"critic", "qa", "quality assurance", "quality_assurance"}
TASK_FIELDNAMES = [
    "dataset",
    "strategy",
    "task_id",
    "run_id",
    "critic_verdict_first_pass",
    "verifier_invoked",
    "verifier_decision",
    "repair_attempted",
    "final_passed",
    "pre_verifier_exec_invoked",
    "pre_verifier_exec_passed",
    "verifier_pre_repair_exec_passed",
    "verifier_post_repair_exec_passed",
    "critic_true_positive",
    "critic_true_negative",
    "blame_executor",
    "blame_critic_false_fail",
    "blame_critic_false_pass",
    "verifier_strict_eligible",
    "verifier_help",
    "verifier_harm",
    "executor_blame_from_critic",
    "critic_blame_from_verifier",
    "blame_basis",
    "verifier_blame_basis",
]
SUMMARY_FIELDNAMES = [
    "dataset",
    "strategy",
    "rows",
    "critic_known_count",
    "final_known_count",
    "critic_true_positive_count",
    "critic_true_negative_count",
    "blame_executor_count",
    "blame_critic_false_fail_count",
    "blame_critic_false_pass_count",
    "verifier_strict_eligible_count",
    "verifier_help_count",
    "verifier_harm_count",
    "executor_blame_from_critic_count",
    "critic_blame_from_verifier_count",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate blame signals from verifier-enabled strategy logs.")
    p.add_argument("--logs-root", type=Path, default=SCRIPT_DIR / "logs", help="Root logs directory.")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "aggregated", help="Output directory for blame CSVs.")
    p.add_argument(
        "--strategies",
        default="",
        help=(
            "Optional comma-separated strategy directory names to include. "
            "If omitted, all strategies found under each dataset directory are included."
        ),
    )
    p.add_argument(
        "--include-non-verifier-strategies",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If false (default), only verifier-oriented strategies are included when --strategies "
            "is not provided."
        ),
    )
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


def parse_float(text: Any) -> float:
    try:
        return float(text)
    except Exception:
        return 0.0


def parse_bool_like(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_critic_verdict(text: str | None) -> str:
    if not text:
        return "UNKNOWN"

    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        return "UNKNOWN"

    # Primary: match first non-empty line, mirroring orchestrator first-line Critic semantics.
    first = lines[0]
    first_up = first.upper().strip()
    # Remove common markdown wrappers.
    first_up = first_up.strip("*_`#>- ").strip()
    m = re.match(r"^(?:FINAL\s+VERDICT\s*[:\-]\s*)?(PASS|FAIL)\b", first_up)
    if m:
        return m.group(1)

    # Secondary: explicit final-verdict markers anywhere in body.
    body_up = str(text).upper()
    m2 = re.search(r"\bFINAL\s+VERDICT\s*[:\-]\s*(PASS|FAIL)\b", body_up)
    if m2:
        return m2.group(1)
    return "UNKNOWN"


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


def is_verifier_strategy_name(strategy: str) -> bool:
    s = (strategy or "").strip().lower()
    return "plus_verifier" in s


def load_boolean_results(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            task_id = str(row.get("task_id", "")).strip()
            if not task_id:
                continue
            if not task_id_matches_dataset(dataset, task_id):
                continue
            latest[task_id] = row
    return latest


def choose_latest_run_per_task(run_summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in run_summary_rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        ts = parse_float(row.get("ts_unix"))
        prev = latest.get(task_id)
        if prev is None or ts >= parse_float(prev.get("ts_unix")):
            latest[task_id] = row
    return latest


def load_runs(path: Path, dataset: str) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_run_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_summaries: list[dict[str, Any]] = []
    if not path.exists():
        return by_run_id, run_summaries
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_type = str(row.get("event_type", "")).strip()
            run_id = str(row.get("run_id", "")).strip()
            task_id = str(row.get("task_id", "")).strip()

            if event_type == "AGENT_CALL" and run_id and task_id and task_id_matches_dataset(dataset, task_id):
                by_run_id[run_id].append(row)

            if event_type == "RUN_SUMMARY" and task_id and task_id_matches_dataset(dataset, task_id):
                run_summaries.append(row)
    return by_run_id, run_summaries


def compute_task_blame_row(
    dataset: str,
    strategy: str,
    task_id: str,
    run_summary: dict[str, Any],
    run_events: list[dict[str, Any]],
    final_bool: dict[str, Any] | None,
) -> dict[str, Any]:
    run_id = str(run_summary.get("run_id", "")).strip()
    verifier_invoked = parse_bool_like(run_summary.get("verifier_invoked"))
    verifier_decision = (run_summary.get("verifier_decision") or "").strip().upper() or None
    repair_attempted = parse_bool_like(run_summary.get("repair_attempted"))

    critic_rows = [
        r
        for r in run_events
        if r.get("event_type") == "AGENT_CALL"
        and (r.get("agent") or "").strip().lower() in CRITIC_AGENT_ALIASES
        and str(r.get("task_id", "")).strip() == task_id
    ]
    critic_rows.sort(key=lambda r: parse_float(r.get("ts_unix")))
    first_critic = critic_rows[0] if critic_rows else None
    critic_text = None
    if first_critic:
        critic_text = (first_critic.get("clean_output") or first_critic.get("raw_output") or "").strip()
    critic_verdict = parse_critic_verdict(critic_text)

    final_passed = parse_bool_like((final_bool or {}).get("passed"))
    pre_verifier_exec_passed_raw = run_summary.get("pre_verifier_exec_passed")
    pre_verifier_exec_passed = parse_bool_like(pre_verifier_exec_passed_raw)
    pre_verifier_exec_invoked = parse_bool_like(run_summary.get("pre_verifier_exec_invoked"))
    verifier_pre_repair_exec_passed = parse_bool_like(run_summary.get("verifier_pre_repair_exec_passed"))
    verifier_post_repair_exec_passed = parse_bool_like(run_summary.get("verifier_post_repair_exec_passed"))

    critic_fail = critic_verdict == "FAIL"
    critic_pass = critic_verdict == "PASS"
    verifier_used = verifier_invoked is True

    # Critic blame should be execution-backed: use pre-verifier execution only.
    if pre_verifier_exec_passed is not None:
        blame_basis = "checkpoint_exec_only"
        pre_fail = pre_verifier_exec_passed is False
        pre_ok = pre_verifier_exec_passed is True
        critic_true_positive = int(critic_fail and pre_fail)
        critic_true_negative = int(critic_pass and pre_ok)
        blame_executor = int(critic_fail and pre_fail)
        blame_critic_false_fail = int(critic_fail and pre_ok)
        blame_critic_false_pass = int(critic_pass and pre_fail)
    else:
        blame_basis = "checkpoint_missing"
        critic_true_positive = 0
        critic_true_negative = 0
        blame_executor = 0
        blame_critic_false_fail = 0
        blame_critic_false_pass = 0

    # Verifier blame should be strict before/after repair checkpoint delta.
    verifier_strict_eligible = int(
        verifier_used and verifier_decision == "REJECT" and repair_attempted is True
    )
    if verifier_strict_eligible:
        if verifier_pre_repair_exec_passed is not None and verifier_post_repair_exec_passed is not None:
            verifier_blame_basis = "repair_checkpoint_delta"
            verifier_help = int(
                (verifier_pre_repair_exec_passed is False)
                and (verifier_post_repair_exec_passed is True)
            )
            verifier_harm = int(
                (verifier_pre_repair_exec_passed is True)
                and (verifier_post_repair_exec_passed is False)
            )
        else:
            verifier_blame_basis = "repair_checkpoint_missing"
            verifier_help = 0
            verifier_harm = 0
    else:
        verifier_blame_basis = "not_applicable"
        verifier_help = 0
        verifier_harm = 0

    # Explicit requested attribution:
    # - Executor blame from Critic verdict alone.
    # - Critic blame from Verifier adjudication.
    if critic_verdict == "FAIL":
        executor_blame_from_critic = 1
    elif critic_verdict == "PASS":
        executor_blame_from_critic = 0
    else:
        executor_blame_from_critic = ""

    if verifier_decision in {"ACCEPT", "REJECT"} and critic_verdict in {"PASS", "FAIL"}:
        critic_blame_from_verifier = int(
            (critic_verdict == "FAIL" and verifier_decision == "ACCEPT")
            or (critic_verdict == "PASS" and verifier_decision == "REJECT")
        )
    else:
        critic_blame_from_verifier = ""

    return {
        "dataset": dataset,
        "strategy": strategy,
        "task_id": task_id,
        "run_id": run_id,
        "critic_verdict_first_pass": critic_verdict,
        "verifier_invoked": int(verifier_used),
        "verifier_decision": verifier_decision or "",
        "repair_attempted": int(repair_attempted is True),
        "final_passed": "" if final_passed is None else str(final_passed),
        "pre_verifier_exec_invoked": "" if pre_verifier_exec_invoked is None else str(pre_verifier_exec_invoked),
        "pre_verifier_exec_passed": "" if pre_verifier_exec_passed is None else str(pre_verifier_exec_passed),
        "verifier_pre_repair_exec_passed": (
            "" if verifier_pre_repair_exec_passed is None else str(verifier_pre_repair_exec_passed)
        ),
        "verifier_post_repair_exec_passed": (
            "" if verifier_post_repair_exec_passed is None else str(verifier_post_repair_exec_passed)
        ),
        "critic_true_positive": critic_true_positive,
        "critic_true_negative": critic_true_negative,
        "blame_executor": blame_executor,
        "blame_critic_false_fail": blame_critic_false_fail,
        "blame_critic_false_pass": blame_critic_false_pass,
        "verifier_strict_eligible": verifier_strict_eligible,
        "verifier_help": verifier_help,
        "verifier_harm": verifier_harm,
        "executor_blame_from_critic": executor_blame_from_critic,
        "critic_blame_from_verifier": critic_blame_from_verifier,
        "blame_basis": blame_basis,
        "verifier_blame_basis": verifier_blame_basis,
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["dataset"], r["strategy"])
        by_scope[key].append(r)

    out: list[dict[str, Any]] = []
    for dataset, strategy in sorted(by_scope.keys()):
        grp = by_scope[(dataset, strategy)]
        n = len(grp)
        critic_known = sum(1 for r in grp if r["critic_verdict_first_pass"] != "UNKNOWN")
        final_known = sum(1 for r in grp if r["final_passed"] in {"True", "False"})
        out.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "rows": n,
                "critic_known_count": critic_known,
                "final_known_count": final_known,
                "critic_true_positive_count": sum(int(r["critic_true_positive"]) for r in grp),
                "critic_true_negative_count": sum(int(r["critic_true_negative"]) for r in grp),
                "blame_executor_count": sum(int(r["blame_executor"]) for r in grp),
                "blame_critic_false_fail_count": sum(int(r["blame_critic_false_fail"]) for r in grp),
                "blame_critic_false_pass_count": sum(int(r["blame_critic_false_pass"]) for r in grp),
                "verifier_strict_eligible_count": sum(int(r["verifier_strict_eligible"]) for r in grp),
                "verifier_help_count": sum(int(r["verifier_help"]) for r in grp),
                "verifier_harm_count": sum(int(r["verifier_harm"]) for r in grp),
                "executor_blame_from_critic_count": sum(
                    int(r["executor_blame_from_critic"])
                    for r in grp
                    if str(r["executor_blame_from_critic"]).strip() != ""
                ),
                "critic_blame_from_verifier_count": sum(
                    int(r["critic_blame_from_verifier"])
                    for r in grp
                    if str(r["critic_blame_from_verifier"]).strip() != ""
                ),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_scoped_csvs(
    output_dir: Path,
    file_name: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> int:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = str(row.get("dataset", "") or "").strip().lower()
        strategy = str(row.get("strategy", "") or "").strip()
        if not dataset or not strategy:
            continue
        grouped[(dataset, strategy)].append(row)

    for (dataset, strategy), scoped_rows in grouped.items():
        write_csv(output_dir / dataset / strategy / file_name, scoped_rows, fieldnames)

    return len(grouped)


def iter_dataset_strategy_dirs(
    logs_root: Path,
    datasets: tuple[str, ...],
    strategy_filter: set[str] | None,
    include_non_verifier_strategies: bool,
) -> list[tuple[str, str, Path]]:
    out: list[tuple[str, str, Path]] = []
    for dataset in datasets:
        ds_dir = logs_root / dataset
        if not ds_dir.exists() or not ds_dir.is_dir():
            continue
        for strategy_dir in sorted([p for p in ds_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            strategy = strategy_dir.name
            if strategy_filter is not None and strategy not in strategy_filter:
                continue
            if strategy_filter is None and (not include_non_verifier_strategies) and (not is_verifier_strategy_name(strategy)):
                continue
            out.append((dataset, strategy, strategy_dir))
    return out


def main() -> None:
    args = parse_args()
    logs_root = resolve_input_path(args.logs_root)
    out_dir = resolve_input_path(args.out_dir)
    strategies = tuple(
        s.strip()
        for s in str(args.strategies or "").split(",")
        if s.strip()
    )
    strategy_filter = set(strategies) if strategies else None

    task_rows: list[dict[str, Any]] = []
    dataset_strategy_dirs = iter_dataset_strategy_dirs(
        logs_root,
        DATASETS,
        strategy_filter,
        include_non_verifier_strategies=bool(args.include_non_verifier_strategies),
    )
    if not dataset_strategy_dirs:
        # Backward-compatible fallback when folder discovery finds nothing.
        fallback_strategies = strategies or DEFAULT_STRATEGIES
        for dataset in DATASETS:
            for strategy in fallback_strategies:
                dataset_strategy_dirs.append((dataset, strategy, logs_root / dataset / strategy))

    for dataset, strategy, base in dataset_strategy_dirs:
        runs_path = base / "runs.csv"
        bool_path = base / "boolean_results.jsonl"
        by_run_id, run_summary_rows = load_runs(runs_path, dataset)
        final_by_task = load_boolean_results(bool_path, dataset)
        latest_by_task = choose_latest_run_per_task(run_summary_rows)

        for task_id, rs in latest_by_task.items():
            run_id = str(rs.get("run_id", "")).strip()
            run_events = by_run_id.get(run_id, [])
            task_rows.append(
                compute_task_blame_row(
                    dataset=dataset,
                    strategy=strategy,
                    task_id=task_id,
                    run_summary=rs,
                    run_events=run_events,
                    final_bool=final_by_task.get(task_id),
                )
            )

    task_rows.sort(key=lambda r: (r["dataset"], r["strategy"], r["task_id"]))
    summary_rows = summarize(task_rows)
    summary_rows.sort(key=lambda r: (r["dataset"], r["strategy"]))

    write_csv(out_dir / "blame_task_level.csv", task_rows, TASK_FIELDNAMES)
    write_csv(out_dir / "blame_summary.csv", summary_rows, SUMMARY_FIELDNAMES)
    scoped_task = write_scoped_csvs(out_dir, "blame_task_level.csv", task_rows, TASK_FIELDNAMES)
    scoped_summary = write_scoped_csvs(out_dir, "blame_summary.csv", summary_rows, SUMMARY_FIELDNAMES)

    print(f"Wrote {len(task_rows)} rows -> {out_dir / 'blame_task_level.csv'}")
    print(f"Wrote {len(summary_rows)} rows -> {out_dir / 'blame_summary.csv'}")
    print(f"Wrote scoped blame_task_level.csv files for {scoped_task} dataset/strategy directories.")
    print(f"Wrote scoped blame_summary.csv files for {scoped_summary} dataset/strategy directories.")


if __name__ == "__main__":
    main()

