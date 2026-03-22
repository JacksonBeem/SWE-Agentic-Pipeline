from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a single per-dataset/per-strategy final metrics report by joining "
            "strategy, token, blame, and similarity aggregates."
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
        default=SCRIPT_DIR / "aggregated" / "final_metrics_report.csv",
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
    if denominator <= 0:
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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def round_numeric_fields(rows: list[dict[str, Any]], digits: int = 6) -> None:
    for row in rows:
        for key, val in list(row.items()):
            if isinstance(val, float):
                row[key] = round(val, digits)


def build_report_rows(aggregated_dir: Path) -> list[dict[str, Any]]:
    strategy_rows = [
        row
        for row in read_csv(aggregated_dir / "strategy_summary.csv")
        if str(row.get("view", "")).strip() in {"", "unique_runs"}
    ]

    blame_map = {
        (str(r.get("dataset", "")).strip(), str(r.get("strategy", "")).strip()): r
        for r in read_csv(aggregated_dir / "blame_summary.csv")
    }

    similarity_map = {
        (str(r.get("dataset", "")).strip(), str(r.get("strategy", "")).strip()): r
        for r in read_csv(aggregated_dir / "artifact_similarity_complexity_summary.csv")
        if str(r.get("scope", "")).strip() in {"", "dataset_strategy"}
    }

    token_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in read_csv(aggregated_dir / "token_stats_by_agent.csv"):
        key = (str(row.get("dataset", "")).strip(), str(row.get("strategy", "")).strip())
        token_grouped.setdefault(key, []).append(row)

    report_rows: list[dict[str, Any]] = []
    for sr in sorted(strategy_rows, key=lambda r: (str(r.get("dataset", "")), str(r.get("strategy", "")))):
        dataset = str(sr.get("dataset", "")).strip()
        strategy = str(sr.get("strategy", "")).strip()
        key = (dataset, strategy)

        bl = blame_map.get(key, {})
        sim = similarity_map.get(key, {})
        tok_rows = token_grouped.get(key, [])
        planner_tok_row = None
        for tr in tok_rows:
            if str(tr.get("agent", "")).strip().lower() == "planner":
                planner_tok_row = tr
                break

        token_call_count = sum(parse_int(r.get("call_count")) or 0 for r in tok_rows)
        token_sum = sum(parse_float(r.get("total_tokens_sum")) or 0.0 for r in tok_rows)

        weighted_token_avg_per_call = None
        if token_call_count > 0:
            weighted_token_avg_per_call = token_sum / token_call_count

        weighted_latency_num = 0.0
        weighted_latency_den = 0
        for r in tok_rows:
            cc = parse_int(r.get("call_count")) or 0
            lat = parse_float(r.get("latency_s_avg_per_call"))
            if cc > 0 and lat is not None:
                weighted_latency_num += lat * cc
                weighted_latency_den += cc
        weighted_latency_avg_per_call = (
            weighted_latency_num / weighted_latency_den if weighted_latency_den > 0 else None
        )
        planner_call_count_for_strategy = parse_int((planner_tok_row or {}).get("call_count"))
        planner_total_tokens_sum_for_strategy = parse_float((planner_tok_row or {}).get("total_tokens_sum"))
        planner_total_tokens_avg_per_call_for_strategy = parse_float(
            (planner_tok_row or {}).get("total_tokens_avg_per_call")
        )
        planner_latency_s_avg_per_call_for_strategy = parse_float(
            (planner_tok_row or {}).get("latency_s_avg_per_call")
        )
        planner_token_share_of_agent_calls_pct_for_strategy = (
            pct(planner_total_tokens_sum_for_strategy, token_sum)
            if (
                planner_total_tokens_sum_for_strategy is not None
                and token_call_count > 0
                and token_sum > 0
            )
            else None
        )
        planner_call_share_of_agent_calls_pct_for_strategy = (
            pct(float(planner_call_count_for_strategy), float(token_call_count))
            if planner_call_count_for_strategy is not None and token_call_count > 0
            else None
        )

        tp = float(parse_int(bl.get("critic_true_positive_count")) or 0)
        tn = float(parse_int(bl.get("critic_true_negative_count")) or 0)
        b_exec = float(parse_int(bl.get("blame_executor_count")) or 0)
        b_false_fail = float(parse_int(bl.get("blame_critic_false_fail_count")) or 0)
        b_false_pass = float(parse_int(bl.get("blame_critic_false_pass_count")) or 0)
        v_eligible = float(parse_int(bl.get("verifier_strict_eligible_count")) or 0)
        v_help = float(parse_int(bl.get("verifier_help_count")) or 0)
        v_harm = float(parse_int(bl.get("verifier_harm_count")) or 0)

        critic_fail_den = tp + b_false_fail
        critic_pass_den = tn + b_false_pass

        row = {
            "dataset": dataset,
            "strategy": strategy,
            "rows": parse_int(sr.get("rows")),
            "unique_task_ids": parse_int(sr.get("unique_task_ids")),
            "expected_task_count": parse_int(sr.get("expected_task_count")),
            "coverage_vs_expected_pct": parse_float(sr.get("coverage_vs_expected_pct")),
            "pass_count": parse_int(sr.get("pass_count")),
            "fail_count": parse_int(sr.get("fail_count")),
            "accuracy_eval_pct": parse_float(sr.get("accuracy_eval_pct")),
            "accuracy_eval_ci95_low_pct": parse_float(sr.get("accuracy_eval_ci95_low_pct")),
            "accuracy_eval_ci95_high_pct": parse_float(sr.get("accuracy_eval_ci95_high_pct")),
            "run_total_tokens_avg": parse_float(sr.get("run_total_tokens_avg")),
            "run_total_tokens_median": parse_float(sr.get("run_total_tokens_median")),
            "run_total_tokens_p90": parse_float(sr.get("run_total_tokens_p90")),
            "end_to_end_latency_s_avg": parse_float(sr.get("end_to_end_latency_s_avg")),
            "end_to_end_latency_s_median": parse_float(sr.get("end_to_end_latency_s_median")),
            "end_to_end_latency_s_p90": parse_float(sr.get("end_to_end_latency_s_p90")),
            "verifier_invoked_rate_pct": parse_float(sr.get("verifier_invoked_rate_pct")),
            "repair_attempted_rate_pct": parse_float(sr.get("repair_attempted_rate_pct")),
            "agent_count": len(tok_rows),
            "agent_call_count": token_call_count,
            "agent_call_total_tokens_sum": token_sum if token_call_count > 0 else None,
            "agent_call_total_tokens_avg_per_call": weighted_token_avg_per_call,
            "agent_call_latency_s_avg_per_call": weighted_latency_avg_per_call,
            "planner_call_count_for_strategy": planner_call_count_for_strategy,
            "planner_total_tokens_sum_for_strategy": planner_total_tokens_sum_for_strategy,
            "planner_total_tokens_avg_per_call_for_strategy": planner_total_tokens_avg_per_call_for_strategy,
            "planner_latency_s_avg_per_call_for_strategy": planner_latency_s_avg_per_call_for_strategy,
            "planner_token_share_of_agent_calls_pct_for_strategy": (
                planner_token_share_of_agent_calls_pct_for_strategy
            ),
            "planner_call_share_of_agent_calls_pct_for_strategy": (
                planner_call_share_of_agent_calls_pct_for_strategy
            ),
            "blame_rows": parse_int(bl.get("rows")),
            "critic_known_count": parse_int(bl.get("critic_known_count")),
            "final_known_count": parse_int(bl.get("final_known_count")),
            "critic_true_positive_count": parse_int(bl.get("critic_true_positive_count")),
            "critic_true_negative_count": parse_int(bl.get("critic_true_negative_count")),
            "blame_executor_count": parse_int(bl.get("blame_executor_count")),
            "blame_critic_false_fail_count": parse_int(bl.get("blame_critic_false_fail_count")),
            "blame_critic_false_pass_count": parse_int(bl.get("blame_critic_false_pass_count")),
            "verifier_strict_eligible_count": parse_int(bl.get("verifier_strict_eligible_count")),
            "verifier_help_count": parse_int(bl.get("verifier_help_count")),
            "verifier_harm_count": parse_int(bl.get("verifier_harm_count")),
            "executor_blame_from_critic_count": parse_int(bl.get("executor_blame_from_critic_count")),
            "critic_blame_from_verifier_count": parse_int(bl.get("critic_blame_from_verifier_count")),
            "executor_blame_rate_on_critic_fail_pct": pct(b_exec, critic_fail_den),
            "critic_false_fail_rate_pct": pct(b_false_fail, critic_fail_den),
            "critic_false_pass_rate_pct": pct(b_false_pass, critic_pass_den),
            "verifier_help_rate_pct": pct(v_help, v_eligible),
            "verifier_harm_rate_pct": pct(v_harm, v_eligible),
            "verifier_net_help_rate_pct": pct(v_help - v_harm, v_eligible),
            "artifact_found_rate_pct": parse_float(sim.get("artifact_found_rate_pct")),
            "artifact_parse_ok_rate_pct": parse_float(sim.get("artifact_parse_ok_rate_pct")),
            "lexical_cosine_similarity_avg": parse_float(sim.get("lexical_cosine_similarity_avg")),
            "token_cosine_similarity_avg": parse_float(sim.get("token_cosine_similarity_avg")),
            "artifact_line_count_avg": parse_float(sim.get("artifact_line_count_avg")),
            "canonical_line_count_avg": parse_float(sim.get("canonical_line_count_avg")),
            "line_count_delta_avg": parse_float(sim.get("line_count_delta_avg")),
            "artifact_cyclomatic_proxy_avg": parse_float(sim.get("artifact_cyclomatic_proxy_avg")),
            "canonical_cyclomatic_proxy_avg": parse_float(sim.get("canonical_cyclomatic_proxy_avg")),
            "cyclomatic_proxy_delta_avg": parse_float(sim.get("cyclomatic_proxy_delta_avg")),
            "artifact_max_control_nesting_depth_avg": parse_float(
                sim.get("artifact_max_control_nesting_depth_avg")
            ),
            "canonical_max_control_nesting_depth_avg": parse_float(
                sim.get("canonical_max_control_nesting_depth_avg")
            ),
            "max_control_nesting_depth_delta_avg": parse_float(sim.get("max_control_nesting_depth_delta_avg")),
        }
        report_rows.append(row)

    by_key = {
        (str(r.get("dataset", "")).strip(), str(r.get("strategy", "")).strip()): r
        for r in report_rows
    }

    for r_no in report_rows:
        dataset = str(r_no.get("dataset", "")).strip()
        strategy_no = str(r_no.get("strategy", "")).strip()
        if not strategy_no.startswith("agentic_no_planner"):
            continue

        strategy_with = normalize_no_planner_to_with_planner(strategy_no)
        r_with = by_key.get((dataset, strategy_with))
        if r_with is None:
            continue

        accuracy_delta_pp = delta(
            parse_float(r_with.get("accuracy_eval_pct")),
            parse_float(r_no.get("accuracy_eval_pct")),
        )
        run_total_tokens_avg_delta = delta(
            parse_float(r_with.get("run_total_tokens_avg")),
            parse_float(r_no.get("run_total_tokens_avg")),
        )
        end_to_end_latency_s_avg_delta = delta(
            parse_float(r_with.get("end_to_end_latency_s_avg")),
            parse_float(r_no.get("end_to_end_latency_s_avg")),
        )
        coverage_delta_pp = delta(
            parse_float(r_with.get("coverage_vs_expected_pct")),
            parse_float(r_no.get("coverage_vs_expected_pct")),
        )
        verifier_invoked_rate_delta_pp = delta(
            parse_float(r_with.get("verifier_invoked_rate_pct")),
            parse_float(r_no.get("verifier_invoked_rate_pct")),
        )
        repair_attempted_rate_delta_pp = delta(
            parse_float(r_with.get("repair_attempted_rate_pct")),
            parse_float(r_no.get("repair_attempted_rate_pct")),
        )
        agent_call_total_tokens_sum_delta = delta(
            parse_float(r_with.get("agent_call_total_tokens_sum")),
            parse_float(r_no.get("agent_call_total_tokens_sum")),
        )
        calls_with = parse_float(r_with.get("agent_call_count"))
        calls_no = parse_float(r_no.get("agent_call_count"))
        agent_call_count_delta = delta(calls_with, calls_no)

        tok_with = parse_float(r_with.get("run_total_tokens_avg"))
        tok_no = parse_float(r_no.get("run_total_tokens_avg"))
        lat_with = parse_float(r_with.get("end_to_end_latency_s_avg"))
        lat_no = parse_float(r_no.get("end_to_end_latency_s_avg"))
        run_tokens_reduction_without_planner_pct = (
            pct(tok_with - tok_no, tok_with)
            if tok_with is not None and tok_no is not None and tok_with > 0
            else None
        )
        latency_reduction_without_planner_pct = (
            pct(lat_with - lat_no, lat_with)
            if lat_with is not None and lat_no is not None and lat_with > 0
            else None
        )

        planner_token_share_pct = parse_float(
            r_with.get("planner_token_share_of_agent_calls_pct_for_strategy")
        )
        planner_call_share_pct = parse_float(
            r_with.get("planner_call_share_of_agent_calls_pct_for_strategy")
        )

        common_impact = {
            "planner_impact_pair_with_planner_strategy": strategy_with,
            "planner_impact_pair_without_planner_strategy": strategy_no,
            "planner_impact_accuracy_delta_pp_agentic_minus_no_planner": accuracy_delta_pp,
            "planner_impact_run_total_tokens_avg_delta_agentic_minus_no_planner": run_total_tokens_avg_delta,
            "planner_impact_end_to_end_latency_s_avg_delta_agentic_minus_no_planner": end_to_end_latency_s_avg_delta,
            "planner_impact_coverage_delta_pp_agentic_minus_no_planner": coverage_delta_pp,
            "planner_impact_verifier_invoked_rate_delta_pp_agentic_minus_no_planner": (
                verifier_invoked_rate_delta_pp
            ),
            "planner_impact_repair_attempted_rate_delta_pp_agentic_minus_no_planner": (
                repair_attempted_rate_delta_pp
            ),
            "planner_impact_agent_call_total_tokens_sum_delta_agentic_minus_no_planner": (
                agent_call_total_tokens_sum_delta
            ),
            "planner_impact_agent_call_count_delta_agentic_minus_no_planner": agent_call_count_delta,
            "planner_impact_run_tokens_reduction_without_planner_pct": run_tokens_reduction_without_planner_pct,
            "planner_impact_latency_reduction_without_planner_pct": latency_reduction_without_planner_pct,
            "planner_impact_planner_token_share_of_with_planner_agent_calls_pct": planner_token_share_pct,
            "planner_impact_planner_call_share_of_with_planner_agent_calls_pct": planner_call_share_pct,
        }
        r_with.update(common_impact)
        r_with["planner_impact_role"] = "with_planner"
        r_no.update(common_impact)
        r_no["planner_impact_role"] = "without_planner"

    return report_rows


def main() -> int:
    args = parse_args()
    aggregated_dir = resolve_path(args.aggregated_dir)
    out_path = resolve_path(args.output)

    rows = build_report_rows(aggregated_dir)
    round_numeric_fields(rows)
    fieldnames = [
        "dataset",
        "strategy",
        "rows",
        "unique_task_ids",
        "expected_task_count",
        "coverage_vs_expected_pct",
        "pass_count",
        "fail_count",
        "accuracy_eval_pct",
        "accuracy_eval_ci95_low_pct",
        "accuracy_eval_ci95_high_pct",
        "run_total_tokens_avg",
        "run_total_tokens_median",
        "run_total_tokens_p90",
        "end_to_end_latency_s_avg",
        "end_to_end_latency_s_median",
        "end_to_end_latency_s_p90",
        "verifier_invoked_rate_pct",
        "repair_attempted_rate_pct",
        "agent_count",
        "agent_call_count",
        "agent_call_total_tokens_sum",
        "agent_call_total_tokens_avg_per_call",
        "agent_call_latency_s_avg_per_call",
        "planner_call_count_for_strategy",
        "planner_total_tokens_sum_for_strategy",
        "planner_total_tokens_avg_per_call_for_strategy",
        "planner_latency_s_avg_per_call_for_strategy",
        "planner_token_share_of_agent_calls_pct_for_strategy",
        "planner_call_share_of_agent_calls_pct_for_strategy",
        "blame_rows",
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
        "executor_blame_rate_on_critic_fail_pct",
        "critic_false_fail_rate_pct",
        "critic_false_pass_rate_pct",
        "verifier_help_rate_pct",
        "verifier_harm_rate_pct",
        "verifier_net_help_rate_pct",
        "artifact_found_rate_pct",
        "artifact_parse_ok_rate_pct",
        "lexical_cosine_similarity_avg",
        "token_cosine_similarity_avg",
        "artifact_line_count_avg",
        "canonical_line_count_avg",
        "line_count_delta_avg",
        "artifact_cyclomatic_proxy_avg",
        "canonical_cyclomatic_proxy_avg",
        "cyclomatic_proxy_delta_avg",
        "artifact_max_control_nesting_depth_avg",
        "canonical_max_control_nesting_depth_avg",
        "max_control_nesting_depth_delta_avg",
        "planner_impact_role",
        "planner_impact_pair_with_planner_strategy",
        "planner_impact_pair_without_planner_strategy",
        "planner_impact_accuracy_delta_pp_agentic_minus_no_planner",
        "planner_impact_run_total_tokens_avg_delta_agentic_minus_no_planner",
        "planner_impact_end_to_end_latency_s_avg_delta_agentic_minus_no_planner",
        "planner_impact_coverage_delta_pp_agentic_minus_no_planner",
        "planner_impact_verifier_invoked_rate_delta_pp_agentic_minus_no_planner",
        "planner_impact_repair_attempted_rate_delta_pp_agentic_minus_no_planner",
        "planner_impact_agent_call_total_tokens_sum_delta_agentic_minus_no_planner",
        "planner_impact_agent_call_count_delta_agentic_minus_no_planner",
        "planner_impact_run_tokens_reduction_without_planner_pct",
        "planner_impact_latency_reduction_without_planner_pct",
        "planner_impact_planner_token_share_of_with_planner_agent_calls_pct",
        "planner_impact_planner_call_share_of_with_planner_agent_calls_pct",
    ]

    write_csv(out_path, rows, fieldnames)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
