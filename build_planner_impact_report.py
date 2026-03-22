from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build planner impact deltas by comparing agentic vs agentic_no_planner "
            "strategies per dataset."
        )
    )
    parser.add_argument(
        "--aggregated-dir",
        type=Path,
        default=SCRIPT_DIR / "aggregated",
        help="Directory containing aggregated CSV outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "aggregated" / "planner_impact_report.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    txt = path.as_posix().lstrip("./")
    if txt.startswith("pipeline/"):
        return SCRIPT_DIR.parent / txt
    return SCRIPT_DIR / path


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def pct(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * numerator / denominator


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def normalize_no_planner_to_with_planner(strategy: str) -> str:
    # agentic_no_planner -> agentic
    # agentic_no_planner_plus_verifier -> agentic_plus_verifier
    return strategy.replace("agentic_no_planner", "agentic", 1)


def round_numeric_fields(rows: list[dict[str, Any]], digits: int = 6) -> None:
    for row in rows:
        for key, val in list(row.items()):
            if isinstance(val, float):
                row[key] = round(val, digits)


def build_rows(aggregated_dir: Path) -> list[dict[str, Any]]:
    final_rows = read_csv(aggregated_dir / "final_metrics_report.csv")
    token_rows = read_csv(aggregated_dir / "token_stats_by_agent.csv")

    by_key = {
        (str(r.get("dataset", "")).strip(), str(r.get("strategy", "")).strip()): r
        for r in final_rows
    }

    planner_tokens_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in token_rows:
        dataset = str(row.get("dataset", "")).strip()
        strategy = str(row.get("strategy", "")).strip()
        agent = str(row.get("agent", "")).strip().lower()
        if agent != "planner":
            continue
        planner_tokens_by_key[(dataset, strategy)] = row

    out: list[dict[str, Any]] = []
    for r_no in sorted(final_rows, key=lambda r: (str(r.get("dataset", "")), str(r.get("strategy", "")))):
        dataset = str(r_no.get("dataset", "")).strip()
        strategy_no = str(r_no.get("strategy", "")).strip()
        if not strategy_no.startswith("agentic_no_planner"):
            continue

        strategy_with = normalize_no_planner_to_with_planner(strategy_no)
        r_with = by_key.get((dataset, strategy_with))
        if r_with is None:
            continue

        acc_with = parse_float(r_with.get("accuracy_eval_pct"))
        acc_no = parse_float(r_no.get("accuracy_eval_pct"))
        tok_with = parse_float(r_with.get("run_total_tokens_avg"))
        tok_no = parse_float(r_no.get("run_total_tokens_avg"))
        lat_with = parse_float(r_with.get("end_to_end_latency_s_avg"))
        lat_no = parse_float(r_no.get("end_to_end_latency_s_avg"))
        ver_with = parse_float(r_with.get("verifier_invoked_rate_pct"))
        ver_no = parse_float(r_no.get("verifier_invoked_rate_pct"))
        rep_with = parse_float(r_with.get("repair_attempted_rate_pct"))
        rep_no = parse_float(r_no.get("repair_attempted_rate_pct"))

        calls_with = parse_int(r_with.get("agent_call_count"))
        calls_no = parse_int(r_no.get("agent_call_count"))
        call_tokens_with = parse_float(r_with.get("agent_call_total_tokens_sum"))
        call_tokens_no = parse_float(r_no.get("agent_call_total_tokens_sum"))

        planner_row = planner_tokens_by_key.get((dataset, strategy_with), {})
        planner_calls = parse_int(planner_row.get("call_count"))
        planner_token_sum = parse_float(planner_row.get("total_tokens_sum"))
        planner_tokens_avg_per_call = parse_float(planner_row.get("total_tokens_avg_per_call"))
        planner_latency_avg_per_call = parse_float(planner_row.get("latency_s_avg_per_call"))

        row = {
            "dataset": dataset,
            "with_planner_strategy": strategy_with,
            "without_planner_strategy": strategy_no,
            "rows_with_planner": parse_int(r_with.get("rows")),
            "rows_without_planner": parse_int(r_no.get("rows")),
            "coverage_with_planner_pct": parse_float(r_with.get("coverage_vs_expected_pct")),
            "coverage_without_planner_pct": parse_float(r_no.get("coverage_vs_expected_pct")),
            "coverage_delta_pp": delta(
                parse_float(r_with.get("coverage_vs_expected_pct")),
                parse_float(r_no.get("coverage_vs_expected_pct")),
            ),
            "accuracy_with_planner_pct": acc_with,
            "accuracy_without_planner_pct": acc_no,
            "accuracy_delta_pp": delta(acc_with, acc_no),
            "run_total_tokens_avg_with_planner": tok_with,
            "run_total_tokens_avg_without_planner": tok_no,
            "run_total_tokens_avg_delta": delta(tok_with, tok_no),
            "run_total_tokens_reduction_without_planner_pct": (
                pct((tok_with - tok_no), tok_with) if tok_with is not None and tok_no is not None else None
            ),
            "end_to_end_latency_s_avg_with_planner": lat_with,
            "end_to_end_latency_s_avg_without_planner": lat_no,
            "end_to_end_latency_s_avg_delta": delta(lat_with, lat_no),
            "latency_reduction_without_planner_pct": (
                pct((lat_with - lat_no), lat_with) if lat_with is not None and lat_no is not None else None
            ),
            "verifier_invoked_rate_with_planner_pct": ver_with,
            "verifier_invoked_rate_without_planner_pct": ver_no,
            "verifier_invoked_rate_delta_pp": delta(ver_with, ver_no),
            "repair_attempted_rate_with_planner_pct": rep_with,
            "repair_attempted_rate_without_planner_pct": rep_no,
            "repair_attempted_rate_delta_pp": delta(rep_with, rep_no),
            "agent_call_count_with_planner": calls_with,
            "agent_call_count_without_planner": calls_no,
            "agent_call_count_delta": (
                calls_with - calls_no if calls_with is not None and calls_no is not None else None
            ),
            "agent_call_total_tokens_sum_with_planner": call_tokens_with,
            "agent_call_total_tokens_sum_without_planner": call_tokens_no,
            "agent_call_total_tokens_sum_delta": delta(call_tokens_with, call_tokens_no),
            "planner_call_count": planner_calls,
            "planner_total_tokens_sum": planner_token_sum,
            "planner_total_tokens_avg_per_call": planner_tokens_avg_per_call,
            "planner_latency_s_avg_per_call": planner_latency_avg_per_call,
            "planner_token_share_of_with_planner_agent_calls_pct": (
                pct(planner_token_sum, call_tokens_with)
                if planner_token_sum is not None and call_tokens_with is not None
                else None
            ),
            "planner_call_share_of_with_planner_agent_calls_pct": (
                pct(float(planner_calls), float(calls_with))
                if planner_calls is not None and calls_with is not None and calls_with > 0
                else None
            ),
        }
        out.append(row)

    return out


def main() -> int:
    args = parse_args()
    aggregated_dir = resolve_path(args.aggregated_dir)
    out_path = resolve_path(args.output)

    rows = build_rows(aggregated_dir)
    round_numeric_fields(rows)
    fieldnames = [
        "dataset",
        "with_planner_strategy",
        "without_planner_strategy",
        "rows_with_planner",
        "rows_without_planner",
        "coverage_with_planner_pct",
        "coverage_without_planner_pct",
        "coverage_delta_pp",
        "accuracy_with_planner_pct",
        "accuracy_without_planner_pct",
        "accuracy_delta_pp",
        "run_total_tokens_avg_with_planner",
        "run_total_tokens_avg_without_planner",
        "run_total_tokens_avg_delta",
        "run_total_tokens_reduction_without_planner_pct",
        "end_to_end_latency_s_avg_with_planner",
        "end_to_end_latency_s_avg_without_planner",
        "end_to_end_latency_s_avg_delta",
        "latency_reduction_without_planner_pct",
        "verifier_invoked_rate_with_planner_pct",
        "verifier_invoked_rate_without_planner_pct",
        "verifier_invoked_rate_delta_pp",
        "repair_attempted_rate_with_planner_pct",
        "repair_attempted_rate_without_planner_pct",
        "repair_attempted_rate_delta_pp",
        "agent_call_count_with_planner",
        "agent_call_count_without_planner",
        "agent_call_count_delta",
        "agent_call_total_tokens_sum_with_planner",
        "agent_call_total_tokens_sum_without_planner",
        "agent_call_total_tokens_sum_delta",
        "planner_call_count",
        "planner_total_tokens_sum",
        "planner_total_tokens_avg_per_call",
        "planner_latency_s_avg_per_call",
        "planner_token_share_of_with_planner_agent_calls_pct",
        "planner_call_share_of_with_planner_agent_calls_pct",
    ]
    write_csv(out_path, rows, fieldnames)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

