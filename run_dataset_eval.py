from __future__ import annotations

import argparse
import ast
import io
import multiprocessing as mp
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

try:
    from pipeline.config import (
        get_active_dataset_path,
        get_active_dataset_type,
        get_default_paths_for_dataset,
        get_workflow_defaults,
    )
    from pipeline.dataset_utils import detect_dataset_type, mbpp_test_harness, task_id_for_row, task_prompt_for_dataset
    from pipeline.utils.artifact_to_code import compose_humaneval_executable_code, extract_code_from_artifact_text
    from pipeline.io_utils import iter_jsonl, strip_fences, write_jsonl
except ModuleNotFoundError:
    from config import (
        get_active_dataset_path,
        get_active_dataset_type,
        get_default_paths_for_dataset,
        get_workflow_defaults,
    )
    from dataset_utils import detect_dataset_type, mbpp_test_harness, task_id_for_row, task_prompt_for_dataset
    from utils.artifact_to_code import compose_humaneval_executable_code, extract_code_from_artifact_text
    from io_utils import iter_jsonl, strip_fences, write_jsonl


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKFLOW_DEFAULTS = get_workflow_defaults()


def resolve_path_from_config(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    txt = path_str.replace("\\", "/")
    if txt.startswith("pipeline/"):
        return PROJECT_ROOT / Path(txt)
    return SCRIPT_DIR / p


DEFAULT_DATASET_PATH = resolve_path_from_config(get_active_dataset_path())
_ACTIVE_DEFAULTS = get_default_paths_for_dataset(get_active_dataset_type())
DEFAULT_INPUT_PATH = resolve_path_from_config(_ACTIVE_DEFAULTS[0])
DEFAULT_OUTPUT_PATH = resolve_path_from_config(_ACTIVE_DEFAULTS[1])
DEFAULT_BOOL_EVAL_OUTPUT = resolve_path_from_config(_ACTIVE_DEFAULTS[2])

def load_dataset_task_ids(dataset_path: Path, dataset_type: str) -> set[str]:
    task_ids: set[str] = set()
    for row in iter_jsonl(dataset_path):
        task_id = task_id_for_row(row, dataset_type)
        if task_id:
            task_ids.add(task_id)
    return task_ids


def load_dataset_rows_by_task(dataset_path: Path, dataset_type: str) -> dict[str, dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(dataset_path):
        task_id = task_id_for_row(row, dataset_type)
        if task_id:
            by_task[task_id] = row
    return by_task


def to_executable_completion(raw_completion: str) -> tuple[str, str | None]:
    extracted = extract_code_from_artifact_text(raw_completion or "")
    if extracted.code:
        return strip_fences(extracted.code), extracted.error
    return strip_fences((raw_completion or "").strip()), extracted.error


def to_prompt_executable(prompt: str, raw_completion: str) -> tuple[str, str | None]:
    extracted = compose_humaneval_executable_code(prompt=prompt, artifact_text=raw_completion or "")
    return strip_fences(extracted.code or ""), extracted.error


def normalize_predictions(
    input_path: Path,
    dataset_by_task: dict[str, dict[str, Any]],
    dataset_type: str,
    valid_task_ids: set[str],
    keep_all_samples: bool,
    completion_field: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows_out: list[dict[str, Any]] = []
    latest_by_task: dict[str, dict[str, Any]] = {}

    stats = {
        "total_input_rows": 0,
        "rows_with_task_id": 0,
        "rows_converted": 0,
        "rows_skipped_not_in_dataset": 0,
        "rows_with_extraction_warning": 0,
    }

    for row in iter_jsonl(input_path):
        stats["total_input_rows"] += 1

        task_id = str(row.get("task_id", "")).strip()
        if dataset_type == "mbpp" and task_id and "/" not in task_id:
            task_id = f"MBPP/{task_id}"
        if not task_id:
            continue
        stats["rows_with_task_id"] += 1

        if valid_task_ids and task_id not in valid_task_ids:
            stats["rows_skipped_not_in_dataset"] += 1
            continue

        raw_completion = str(row.get(completion_field, "") or "")
        ds_row = dataset_by_task.get(task_id, {})
        if dataset_type == "humaneval":
            prompt = task_prompt_for_dataset(ds_row, dataset_type)
            completion, err = to_prompt_executable(prompt=prompt, raw_completion=raw_completion)
        else:
            completion, err = to_executable_completion(raw_completion=raw_completion)
        if err:
            stats["rows_with_extraction_warning"] += 1

        out = {
            "task_id": task_id,
            "completion": completion,
        }
        if row.get("model"):
            out["model"] = row["model"]
        if row.get("model_name_or_path"):
            out["model_name_or_path"] = row["model_name_or_path"]

        stats["rows_converted"] += 1
        if keep_all_samples:
            rows_out.append(out)
        else:
            latest_by_task[task_id] = out

    if keep_all_samples:
        return rows_out, stats

    deduped = [latest_by_task[k] for k in sorted(latest_by_task.keys())]
    return deduped, stats


def _eval_worker(payload: dict[str, Any], out_q: mp.Queue) -> None:
    try:
        ns: dict[str, Any] = {}
        if payload["dataset_type"] == "bigcodebench":
            exec(payload["prompt"], ns, ns)
            exec(payload["completion"], ns, ns)
            exec(payload["test"], ns, ns)
            test_case = ns.get("TestCases")
            if not test_case:
                raise RuntimeError("BigCodeBench test did not define TestCases.")
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
            stream = io.StringIO()
            result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
            if result.wasSuccessful():
                out_q.put({"passed": True, "error_type": None, "error": None})
            else:
                err_text = stream.getvalue().strip() or "unittest failures"
                out_q.put({"passed": False, "error_type": "AssertionError", "error": err_text})
        elif payload["dataset_type"] == "mbpp":
            exec(payload["completion"], ns, ns)
            if payload["test"]:
                exec(payload["test"], ns, ns)
            out_q.put({"passed": True, "error_type": None, "error": None})
        else:
            exec(payload["prompt"], ns, ns)
            exec(payload["completion"], ns, ns)
            exec(payload["test"], ns, ns)
            candidate = ns[payload["entry_point"]]
            ns["check"](candidate)
            out_q.put({"passed": True, "error_type": None, "error": None})
    except Exception as exc:
        out_q.put({"passed": False, "error_type": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}"})


def evaluate_boolean(
    predictions: list[dict[str, Any]],
    dataset_by_task: dict[str, dict[str, Any]],
    dataset_type: str,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    results: list[dict[str, Any]] = []
    stats = {
        "eval_total": 0,
        "eval_passed": 0,
        "eval_failed": 0,
        "eval_timeout": 0,
        "eval_missing_task": 0,
        "eval_parse_error": 0,
    }

    for pred in predictions:
        task_id = str(pred.get("task_id", "")).strip()
        if not task_id:
            continue

        stats["eval_total"] += 1
        task = dataset_by_task.get(task_id)
        if not task:
            stats["eval_missing_task"] += 1
            stats["eval_failed"] += 1
            results.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error_type": "MissingTaskError",
                    "error": "Task not found in dataset.",
                }
            )
            continue

        payload = {
            "prompt": task_prompt_for_dataset(task, dataset_type),
            "completion": str(pred.get("completion", "") or ""),
            "test": (
                mbpp_test_harness(task)
                if dataset_type == "mbpp"
                else str(task.get("test", "") or "")
            ),
            "entry_point": str(task.get("entry_point", "") or ""),
            "dataset_type": dataset_type,
        }
        try:
            ast.parse(payload["completion"])
        except Exception as exc:
            stats["eval_parse_error"] += 1
            stats["eval_failed"] += 1
            results.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue

        q: mp.Queue = mp.Queue()
        proc = mp.Process(target=_eval_worker, args=(payload, q))
        proc.start()
        proc.join(timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            stats["eval_timeout"] += 1
            stats["eval_failed"] += 1
            results.append(
                {
                    "task_id": task_id,
                    "passed": False,
                    "error_type": "TimeoutError",
                    "error": f"Timeout after {timeout_s:.1f}s",
                }
            )
            q.close()
            continue

        worker_out = {"passed": False, "error_type": "RuntimeError", "error": "No result returned."}
        if not q.empty():
            worker_out = q.get()
        q.close()

        passed = bool(worker_out.get("passed"))
        if passed:
            stats["eval_passed"] += 1
        else:
            stats["eval_failed"] += 1

        results.append(
            {
                "task_id": task_id,
                "passed": passed,
                "error_type": worker_out.get("error_type"),
                "error": worker_out.get("error"),
            }
        )

    return results, stats


def run_humaneval_eval(predictions_path: Path, dataset_path: Path) -> int:
    commands = [
        [
            "evaluate_functional_correctness",
            str(predictions_path),
            "--problem_file",
            str(dataset_path),
        ],
        [
            sys.executable,
            "-m",
            "human_eval.evaluate_functional_correctness",
            str(predictions_path),
            "--problem_file",
            str(dataset_path),
        ],
    ]

    for cmd in commands:
        try:
            print(f"Trying evaluator command: {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False)
            if proc.returncode == 0:
                return 0
        except FileNotFoundError:
            continue

    print(
        "Could not run HumanEval evaluator automatically. "
        "Install/openai-human-eval and run evaluate_functional_correctness manually.",
        file=sys.stderr,
    )
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert pipeline HumanEval outputs into executable predictions and "
            "optionally run full HumanEval evaluation."
        )
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument(
        "--dataset-type",
        choices=["auto", "humaneval", "bigcodebench", "mbpp"],
        default="auto",
        help="Dataset schema/evaluation style.",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    p.add_argument(
        "--completion-field",
        default="completion",
        help="Input JSONL field containing the pipeline artifact/completion text.",
    )
    p.add_argument(
        "--keep-all-samples",
        action="store_true",
        help="Keep all converted samples. Default keeps latest sample per task_id.",
    )
    p.add_argument(
        "--run-eval",
        action="store_true",
        help="Run HumanEval functional correctness after conversion.",
    )
    p.add_argument(
        "--run-bool-eval",
        action="store_true",
        help="Run local per-task execution and output passed=true/false.",
    )
    p.add_argument(
        "--bool-eval-output",
        type=Path,
        default=DEFAULT_BOOL_EVAL_OUTPUT,
        help="Where to write per-task true/false evaluation rows.",
    )
    p.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Per-task timeout in seconds for local boolean evaluation.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset.resolve()}")
    if not args.input.exists():
        raise FileNotFoundError(f"Input predictions not found: {args.input.resolve()}")

    if args.dataset_type == "auto":
        first_row = next(iter(iter_jsonl(args.dataset)), None)
        if not first_row:
            raise ValueError(f"No rows found in dataset: {args.dataset}")
        dataset_type = detect_dataset_type(first_row)
    else:
        dataset_type = args.dataset_type

    configured_default_input = DEFAULT_INPUT_PATH
    configured_default_output = DEFAULT_OUTPUT_PATH
    configured_default_bool = DEFAULT_BOOL_EVAL_OUTPUT
    ds_default_input, ds_default_output, ds_default_bool = get_default_paths_for_dataset(dataset_type)
    if args.input == configured_default_input:
        args.input = resolve_path_from_config(ds_default_input)
    if args.output == configured_default_output:
        args.output = resolve_path_from_config(ds_default_output)
    if args.bool_eval_output == configured_default_bool:
        args.bool_eval_output = resolve_path_from_config(ds_default_bool)

    if not args.input.exists():
        raise FileNotFoundError(f"Input predictions not found: {args.input.resolve()}")

    dataset_by_task = load_dataset_rows_by_task(args.dataset, dataset_type)
    if not dataset_by_task:
        raise ValueError(f"No rows found in dataset: {args.dataset}")
    task_ids = set(dataset_by_task.keys())
    converted, stats = normalize_predictions(
        input_path=args.input,
        dataset_by_task=dataset_by_task,
        dataset_type=dataset_type,
        valid_task_ids=task_ids,
        keep_all_samples=args.keep_all_samples,
        completion_field=args.completion_field,
    )
    write_jsonl(args.output, converted)

    converted_task_ids = {r["task_id"] for r in converted}
    missing = len(task_ids - converted_task_ids)

    print(f"Wrote executable predictions: {args.output}")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"output_rows: {len(converted)}")
    print(f"dataset_task_count: {len(task_ids)}")
    print(f"missing_task_count: {missing}")

    if args.run_eval:
        rc = run_humaneval_eval(args.output, args.dataset)
        if rc != 0:
            return rc

    if args.run_bool_eval:
        bool_results, bool_stats = evaluate_boolean(
            predictions=converted,
            dataset_by_task=dataset_by_task,
            dataset_type=dataset_type,
            timeout_s=args.timeout_s,
        )
        write_jsonl(args.bool_eval_output, bool_results)
        print(f"Wrote boolean results: {args.bool_eval_output}")
        for k, v in bool_stats.items():
            print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
