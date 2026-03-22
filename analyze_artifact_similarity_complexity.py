from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import re
import tokenize
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from pipeline.aggregate_results import EXPECTED_TASKS, iter_strategy_dirs
    from pipeline.dataset_utils import task_id_for_row
    from pipeline.io_utils import iter_jsonl, write_jsonl
    from pipeline.utils.artifact_to_code import compose_prompt_executable_code, extract_code_from_artifact_text
except ModuleNotFoundError:
    from aggregate_results import EXPECTED_TASKS, iter_strategy_dirs
    from dataset_utils import task_id_for_row
    from io_utils import iter_jsonl, write_jsonl
    from utils.artifact_to_code import compose_prompt_executable_code, extract_code_from_artifact_text


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

TOKEN_SKIP_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.INDENT,
    tokenize.DEDENT,
}
CONTROL_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.IfExp,
    ast.Match,
    ast.comprehension,
)
BRANCH_NODES = (ast.If, ast.IfExp, ast.Try, ast.Match)
LOOP_NODES = (ast.For, ast.AsyncFor, ast.While, ast.comprehension)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity and static complexity metrics between final executable "
            "artifacts and canonical solutions across all datasets and strategies."
        )
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("pipeline/logs"),
        help="Root directory containing <dataset>/<strategy>/ logs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pipeline/aggregated"),
        help="Directory for analysis outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("pipeline/data"),
        help="Directory containing source dataset JSONL files.",
    )
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp", "bigcodebench"],
        help="Optional dataset filter.",
    )
    parser.add_argument(
        "--strategy",
        choices=["agentic", "agentic_plus_verifier", "monolithic"],
        help="Optional strategy/configuration filter.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def normalize_task_id(dataset: str, task_id: str) -> str:
    tid = (task_id or "").strip()
    if not tid:
        return ""
    if dataset == "mbpp" and "/" not in tid:
        return f"MBPP/{tid}"
    return tid


def load_dataset_rows(dataset_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    dataset_files = {
        "humaneval": dataset_root / "human_eval.jsonl",
        "mbpp": dataset_root / "mbpp_sanitized_200.jsonl",
        "bigcodebench": dataset_root / "bigcodebench_200.jsonl",
    }
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, path in dataset_files.items():
        by_task: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(path):
            task_id = task_id_for_row(row, dataset)
            if task_id:
                by_task[task_id] = row
        out[dataset] = by_task
    return out


def build_canonical_code(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "mbpp":
        return str(row.get("code", "") or "").strip()

    if dataset == "humaneval":
        prompt = str(row.get("prompt", "") or "")
        body = str(row.get("canonical_solution", "") or "")
        extracted = compose_prompt_executable_code(prompt=prompt, artifact_text=body)
        return (extracted.code or "").strip()

    prompt = str(row.get("code_prompt", "") or row.get("complete_prompt", "") or "")
    body = str(row.get("canonical_solution", "") or "")
    extracted = compose_prompt_executable_code(prompt=prompt, artifact_text=body)
    return (extracted.code or "").strip()


def load_prediction_map(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_task
    for row in iter_jsonl(path):
        task_id = normalize_task_id(dataset, str(row.get("task_id", "") or ""))
        if task_id:
            by_task[task_id] = row
    return by_task


def load_boolean_map(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_task
    for row in iter_jsonl(path):
        task_id = normalize_task_id(dataset, str(row.get("task_id", "") or ""))
        if task_id:
            by_task[task_id] = row
    return by_task


def load_runs_fallback(path: Path, dataset: str) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return by_task
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("event_type", "") or "").strip() != "RUN_SUMMARY":
                continue
            task_id = normalize_task_id(dataset, str(row.get("task_id", "") or ""))
            if not task_id:
                continue
            code = str(row.get("final_executable_code", "") or "").strip()
            if not code:
                continue
            by_task[task_id] = row
    return by_task


def strip_trailing_analysis(text: str) -> str:
    if not text:
        return ""
    markers = (
        "===CRITIC===",
        "===FINAL_CODE===",
        "CRITIC:",
        "FINAL_CODE",
        "FINAL CODE",
        "MENTAL SIMULATION:",
        "ANALYSIS:",
    )
    lines: list[str] = []
    for line in text.splitlines():
        upper = line.strip().upper()
        if any(marker in upper for marker in markers):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def normalize_code(text: str) -> tuple[str, str | None]:
    raw = strip_trailing_analysis(text or "")
    extracted = extract_code_from_artifact_text(raw)
    code = (extracted.code or raw or "").strip()
    if not code:
        return "", "empty"

    try:
        tree = ast.parse(code)
        normalized = ast.unparse(tree).strip()
        return normalized, None
    except Exception as exc:
        return code, f"{type(exc).__name__}: {exc}"


def tokenize_code(text: str) -> list[str]:
    if not text.strip():
        return []
    out: list[str] = []
    try:
        stream = io.StringIO(text)
        for tok in tokenize.generate_tokens(stream.readline):
            if tok.type in TOKEN_SKIP_TYPES:
                continue
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING and tok.string.startswith(('"""', "'''")):
                out.append("DOCSTRING")
                continue
            if tok.type == tokenize.NAME:
                out.append(tok.string)
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return re.findall(r"[A-Za-z_]\w+|\d+|[^\s]", text)
    return out


def lexical_terms(text: str) -> Counter[str]:
    tokens = re.findall(r"[A-Za-z_]\w+|\d+", text.lower())
    return Counter(tokens)


def token_terms(text: str) -> Counter[str]:
    return Counter(tokenize_code(text))


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float | None:
    if not left or not right:
        return None
    dot = sum(left[token] * right.get(token, 0) for token in left)
    if dot == 0:
        return 0.0
    left_mag = math.sqrt(sum(value * value for value in left.values()))
    right_mag = math.sqrt(sum(value * value for value in right.values()))
    if left_mag == 0.0 or right_mag == 0.0:
        return None
    return dot / (left_mag * right_mag)


class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.node_count = 0
        self.branch_count = 0
        self.loop_count = 0
        self.call_count = 0
        self.comprehension_count = 0
        self.return_count = 0
        self.function_count = 0
        self.max_depth = 0
        self._depth = 0
        self.cyclomatic = 1

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1

        entered_control = isinstance(node, CONTROL_NODES)
        if isinstance(node, BRANCH_NODES):
            self.branch_count += 1
            self.cyclomatic += 1
        if isinstance(node, LOOP_NODES):
            self.loop_count += 1
            self.cyclomatic += 1
        if isinstance(node, ast.BoolOp):
            self.cyclomatic += max(0, len(node.values) - 1)
        if isinstance(node, ast.Call):
            self.call_count += 1
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self.comprehension_count += 1
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            self.function_count += 1
        if isinstance(node, ast.Return):
            self.return_count += 1

        if entered_control:
            self._depth += 1
            self.max_depth = max(self.max_depth, self._depth)
        super().generic_visit(node)
        if entered_control:
            self._depth -= 1


def analyze_complexity(text: str) -> dict[str, Any]:
    stripped = text.strip()
    lines = stripped.splitlines() if stripped else []
    nonempty_lines = [line for line in lines if line.strip()]
    metrics = {
        "char_count": len(stripped),
        "line_count": len(lines),
        "nonempty_line_count": len(nonempty_lines),
        "parse_ok": 0,
        "parse_error": None,
        "ast_node_count": None,
        "function_count": None,
        "branch_count": None,
        "loop_count": None,
        "call_count": None,
        "comprehension_count": None,
        "return_count": None,
        "max_control_nesting_depth": None,
        "cyclomatic_proxy": None,
    }
    if not stripped:
        metrics["parse_error"] = "empty"
        return metrics

    try:
        tree = ast.parse(stripped)
    except Exception as exc:
        metrics["parse_error"] = f"{type(exc).__name__}: {exc}"
        return metrics

    visitor = ComplexityVisitor()
    visitor.visit(tree)
    metrics.update(
        {
            "parse_ok": 1,
            "ast_node_count": visitor.node_count,
            "function_count": visitor.function_count,
            "branch_count": visitor.branch_count,
            "loop_count": visitor.loop_count,
            "call_count": visitor.call_count,
            "comprehension_count": visitor.comprehension_count,
            "return_count": visitor.return_count,
            "max_control_nesting_depth": visitor.max_depth,
            "cyclomatic_proxy": visitor.cyclomatic,
        }
    )
    return metrics


def safe_delta(left: Any, right: Any) -> Any:
    if left is None or right is None:
        return None
    return left - right


def safe_ratio(left: Any, right: Any) -> float | None:
    if left is None or right in {None, 0}:
        return None
    return left / right


def build_task_rows(
    datasets_by_task: dict[str, dict[str, dict[str, Any]]],
    logs_root: Path,
    dataset_filter: str | None = None,
    strategy_filter: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, strategy, strategy_dir in iter_strategy_dirs(logs_root):
        if dataset_filter and dataset != dataset_filter:
            continue
        if strategy_filter and strategy != strategy_filter:
            continue
        pred_exec_path = strategy_dir / "predictions_executable.jsonl"
        bool_path = strategy_dir / "boolean_results.jsonl"
        runs_path = strategy_dir / "runs.csv"

        prediction_map = load_prediction_map(pred_exec_path, dataset)
        bool_map = load_boolean_map(bool_path, dataset)
        run_fallback_map = load_runs_fallback(runs_path, dataset)
        dataset_rows = datasets_by_task.get(dataset, {})

        for task_id, ds_row in sorted(dataset_rows.items()):
            canonical_raw = build_canonical_code(dataset, ds_row)
            canonical_code, canonical_parse_error = normalize_code(canonical_raw)

            prediction_row = prediction_map.get(task_id)
            run_row = run_fallback_map.get(task_id)

            artifact_source = None
            artifact_raw = ""
            model_name = None
            if prediction_row:
                artifact_source = "predictions_executable"
                artifact_raw = str(prediction_row.get("completion", "") or "")
                model_name = str(prediction_row.get("model", "") or "") or None
            elif run_row:
                artifact_source = "runs_csv_final_executable_code"
                artifact_raw = str(run_row.get("final_executable_code", "") or "")

            artifact_code, artifact_parse_error = normalize_code(artifact_raw)

            canonical_lexical = lexical_terms(canonical_code)
            artifact_lexical = lexical_terms(artifact_code)
            canonical_tokens = token_terms(canonical_code)
            artifact_tokens = token_terms(artifact_code)

            canonical_metrics = analyze_complexity(canonical_code)
            artifact_metrics = analyze_complexity(artifact_code)
            bool_row = bool_map.get(task_id, {})
            passed = parse_bool(bool_row.get("passed"))

            row = {
                "dataset": dataset,
                "strategy": strategy,
                "task_id": task_id,
                "expected_task_count": EXPECTED_TASKS.get(dataset),
                "artifact_source": artifact_source,
                "artifact_found": int(bool(artifact_source)),
                "model": model_name,
                "passed": passed,
                "error_type": str(bool_row.get("error_type", "") or "").strip() or None,
                "canonical_parse_error": canonical_parse_error,
                "artifact_parse_error": artifact_parse_error,
                "canonical_char_count": canonical_metrics["char_count"],
                "artifact_char_count": artifact_metrics["char_count"],
                "canonical_line_count": canonical_metrics["line_count"],
                "artifact_line_count": artifact_metrics["line_count"],
                "canonical_nonempty_line_count": canonical_metrics["nonempty_line_count"],
                "artifact_nonempty_line_count": artifact_metrics["nonempty_line_count"],
                "canonical_parse_ok": canonical_metrics["parse_ok"],
                "artifact_parse_ok": artifact_metrics["parse_ok"],
                "canonical_ast_node_count": canonical_metrics["ast_node_count"],
                "artifact_ast_node_count": artifact_metrics["ast_node_count"],
                "canonical_function_count": canonical_metrics["function_count"],
                "artifact_function_count": artifact_metrics["function_count"],
                "canonical_branch_count": canonical_metrics["branch_count"],
                "artifact_branch_count": artifact_metrics["branch_count"],
                "canonical_loop_count": canonical_metrics["loop_count"],
                "artifact_loop_count": artifact_metrics["loop_count"],
                "canonical_call_count": canonical_metrics["call_count"],
                "artifact_call_count": artifact_metrics["call_count"],
                "canonical_comprehension_count": canonical_metrics["comprehension_count"],
                "artifact_comprehension_count": artifact_metrics["comprehension_count"],
                "canonical_return_count": canonical_metrics["return_count"],
                "artifact_return_count": artifact_metrics["return_count"],
                "canonical_max_control_nesting_depth": canonical_metrics["max_control_nesting_depth"],
                "artifact_max_control_nesting_depth": artifact_metrics["max_control_nesting_depth"],
                "canonical_cyclomatic_proxy": canonical_metrics["cyclomatic_proxy"],
                "artifact_cyclomatic_proxy": artifact_metrics["cyclomatic_proxy"],
                "lexical_cosine_similarity": cosine_similarity(artifact_lexical, canonical_lexical),
                "token_cosine_similarity": cosine_similarity(artifact_tokens, canonical_tokens),
                "line_count_delta": safe_delta(
                    artifact_metrics["line_count"], canonical_metrics["line_count"]
                ),
                "nonempty_line_count_delta": safe_delta(
                    artifact_metrics["nonempty_line_count"], canonical_metrics["nonempty_line_count"]
                ),
                "ast_node_count_delta": safe_delta(
                    artifact_metrics["ast_node_count"], canonical_metrics["ast_node_count"]
                ),
                "branch_count_delta": safe_delta(
                    artifact_metrics["branch_count"], canonical_metrics["branch_count"]
                ),
                "loop_count_delta": safe_delta(
                    artifact_metrics["loop_count"], canonical_metrics["loop_count"]
                ),
                "call_count_delta": safe_delta(
                    artifact_metrics["call_count"], canonical_metrics["call_count"]
                ),
                "max_control_nesting_depth_delta": safe_delta(
                    artifact_metrics["max_control_nesting_depth"],
                    canonical_metrics["max_control_nesting_depth"],
                ),
                "cyclomatic_proxy_delta": safe_delta(
                    artifact_metrics["cyclomatic_proxy"], canonical_metrics["cyclomatic_proxy"]
                ),
                "line_count_ratio": safe_ratio(
                    artifact_metrics["line_count"], canonical_metrics["line_count"]
                ),
                "ast_node_count_ratio": safe_ratio(
                    artifact_metrics["ast_node_count"], canonical_metrics["ast_node_count"]
                ),
                "cyclomatic_proxy_ratio": safe_ratio(
                    artifact_metrics["cyclomatic_proxy"], canonical_metrics["cyclomatic_proxy"]
                ),
            }
            rows.append(row)
    return rows


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found_rows = [row for row in rows if row.get("artifact_found")]
    parse_ok_rows = [row for row in rows if row.get("artifact_parse_ok") == 1]

    def metric_mean(name: str, subset: list[dict[str, Any]]) -> float | None:
        values = [row[name] for row in subset if row.get(name) is not None]
        return mean(values) if values else None

    def metric_median(name: str, subset: list[dict[str, Any]]) -> float | None:
        values = [row[name] for row in subset if row.get(name) is not None]
        return median(values) if values else None

    return {
        "rows": len(rows),
        "artifact_found_count": len(found_rows),
        "artifact_found_rate_pct": (100.0 * len(found_rows) / len(rows)) if rows else None,
        "artifact_parse_ok_count": len(parse_ok_rows),
        "artifact_parse_ok_rate_pct": (100.0 * len(parse_ok_rows) / len(rows)) if rows else None,
        "pass_count": sum(1 for row in rows if row.get("passed") is True),
        "fail_count": sum(1 for row in rows if row.get("passed") is False),
        "lexical_cosine_similarity_avg": metric_mean("lexical_cosine_similarity", found_rows),
        "lexical_cosine_similarity_median": metric_median("lexical_cosine_similarity", found_rows),
        "token_cosine_similarity_avg": metric_mean("token_cosine_similarity", found_rows),
        "token_cosine_similarity_median": metric_median("token_cosine_similarity", found_rows),
        "artifact_line_count_avg": metric_mean("artifact_line_count", found_rows),
        "canonical_line_count_avg": metric_mean("canonical_line_count", rows),
        "line_count_delta_avg": metric_mean("line_count_delta", found_rows),
        "artifact_ast_node_count_avg": metric_mean("artifact_ast_node_count", parse_ok_rows),
        "canonical_ast_node_count_avg": metric_mean("canonical_ast_node_count", rows),
        "ast_node_count_delta_avg": metric_mean("ast_node_count_delta", parse_ok_rows),
        "artifact_cyclomatic_proxy_avg": metric_mean("artifact_cyclomatic_proxy", parse_ok_rows),
        "canonical_cyclomatic_proxy_avg": metric_mean("canonical_cyclomatic_proxy", rows),
        "cyclomatic_proxy_delta_avg": metric_mean("cyclomatic_proxy_delta", parse_ok_rows),
        "artifact_max_control_nesting_depth_avg": metric_mean(
            "artifact_max_control_nesting_depth", parse_ok_rows
        ),
        "canonical_max_control_nesting_depth_avg": metric_mean(
            "canonical_max_control_nesting_depth", rows
        ),
        "max_control_nesting_depth_delta_avg": metric_mean(
            "max_control_nesting_depth_delta", parse_ok_rows
        ),
    }


def build_summary_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    dataset_grouped: dict[str, list[dict[str, Any]]] = {}

    for row in task_rows:
        grouped.setdefault((row["dataset"], row["strategy"]), []).append(row)
        dataset_grouped.setdefault(row["dataset"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, strategy), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "scope": "dataset_strategy",
                "dataset": dataset,
                "strategy": strategy,
                **summarize_group(rows),
            }
        )

    for dataset, rows in sorted(dataset_grouped.items()):
        summary_rows.append(
            {
                "scope": "dataset_all_strategies",
                "dataset": dataset,
                "strategy": "all",
                **summarize_group(rows),
            }
        )
    return summary_rows


def task_fieldnames() -> list[str]:
    return [
        "dataset",
        "strategy",
        "task_id",
        "expected_task_count",
        "artifact_source",
        "artifact_found",
        "model",
        "passed",
        "error_type",
        "canonical_parse_error",
        "artifact_parse_error",
        "canonical_char_count",
        "artifact_char_count",
        "canonical_line_count",
        "artifact_line_count",
        "canonical_nonempty_line_count",
        "artifact_nonempty_line_count",
        "canonical_parse_ok",
        "artifact_parse_ok",
        "canonical_ast_node_count",
        "artifact_ast_node_count",
        "canonical_function_count",
        "artifact_function_count",
        "canonical_branch_count",
        "artifact_branch_count",
        "canonical_loop_count",
        "artifact_loop_count",
        "canonical_call_count",
        "artifact_call_count",
        "canonical_comprehension_count",
        "artifact_comprehension_count",
        "canonical_return_count",
        "artifact_return_count",
        "canonical_max_control_nesting_depth",
        "artifact_max_control_nesting_depth",
        "canonical_cyclomatic_proxy",
        "artifact_cyclomatic_proxy",
        "lexical_cosine_similarity",
        "token_cosine_similarity",
        "line_count_delta",
        "nonempty_line_count_delta",
        "ast_node_count_delta",
        "branch_count_delta",
        "loop_count_delta",
        "call_count_delta",
        "max_control_nesting_depth_delta",
        "cyclomatic_proxy_delta",
        "line_count_ratio",
        "ast_node_count_ratio",
        "cyclomatic_proxy_ratio",
    ]


def summary_fieldnames() -> list[str]:
    return [
        "scope",
        "dataset",
        "strategy",
        "rows",
        "artifact_found_count",
        "artifact_found_rate_pct",
        "artifact_parse_ok_count",
        "artifact_parse_ok_rate_pct",
        "pass_count",
        "fail_count",
        "lexical_cosine_similarity_avg",
        "lexical_cosine_similarity_median",
        "token_cosine_similarity_avg",
        "token_cosine_similarity_median",
        "artifact_line_count_avg",
        "canonical_line_count_avg",
        "line_count_delta_avg",
        "artifact_ast_node_count_avg",
        "canonical_ast_node_count_avg",
        "ast_node_count_delta_avg",
        "artifact_cyclomatic_proxy_avg",
        "canonical_cyclomatic_proxy_avg",
        "cyclomatic_proxy_delta_avg",
        "artifact_max_control_nesting_depth_avg",
        "canonical_max_control_nesting_depth_avg",
        "max_control_nesting_depth_delta_avg",
    ]


def write_analysis_bundle(
    output_dir: Path,
    task_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    file_prefix: str = "",
) -> None:
    prefix = str(file_prefix or "")
    write_csv(output_dir / f"{prefix}artifact_similarity_complexity_per_task.csv", task_rows, task_fieldnames())
    write_jsonl(output_dir / f"{prefix}artifact_similarity_complexity_per_task.jsonl", task_rows)
    write_csv(
        output_dir / f"{prefix}artifact_similarity_complexity_summary.csv",
        summary_rows,
        summary_fieldnames(),
    )


def select_summary_rows(
    summary_rows: list[dict[str, Any]],
    dataset: str | None = None,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in summary_rows:
        if dataset and row.get("dataset") != dataset:
            continue
        if strategy and row.get("strategy") != strategy:
            continue
        if strategy and row.get("scope") != "dataset_strategy":
            continue
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    logs_root = resolve_path(args.logs_root)
    output_dir = resolve_path(args.output_dir)
    dataset_root = resolve_path(args.dataset_root)

    datasets_by_task = load_dataset_rows(dataset_root)
    if args.dataset or args.strategy:
        task_rows = build_task_rows(
            datasets_by_task=datasets_by_task,
            logs_root=logs_root,
            dataset_filter=args.dataset,
            strategy_filter=args.strategy,
        )
        summary_rows = build_summary_rows(task_rows)
        if args.dataset and args.strategy:
            out_dir = output_dir / args.dataset / args.strategy
        else:
            parts = [part for part in [args.dataset, args.strategy] if part]
            out_dir = output_dir / "_".join(parts)
        write_analysis_bundle(
            out_dir,
            task_rows,
            select_summary_rows(summary_rows, dataset=args.dataset, strategy=args.strategy),
        )
        return

    combined_task_rows: list[dict[str, Any]] = []
    for dataset, strategy, _strategy_dir in iter_strategy_dirs(logs_root):
        task_rows = build_task_rows(
            datasets_by_task=datasets_by_task,
            logs_root=logs_root,
            dataset_filter=dataset,
            strategy_filter=strategy,
        )
        summary_rows = build_summary_rows(task_rows)
        write_analysis_bundle(
            output_dir / dataset / strategy,
            task_rows,
            select_summary_rows(summary_rows, dataset=dataset, strategy=strategy),
        )
        combined_task_rows.extend(task_rows)

    write_analysis_bundle(output_dir, combined_task_rows, build_summary_rows(combined_task_rows))


if __name__ == "__main__":
    main()

