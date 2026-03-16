from __future__ import annotations

import ast
import io
import multiprocessing as mp
import re
import unittest
from typing import Any

from .schemas import TaskInput
from .utils.artifact_to_code import compose_humaneval_executable_code, extract_code_from_artifact_text


def infer_dataset_type_from_task_id(task_id: str) -> str:
    t = (task_id or "").strip().lower()
    if t.startswith("mbpp/"):
        return "mbpp"
    if t.startswith("bigcodebench/"):
        return "bigcodebench"
    return "humaneval"


def infer_humaneval_entry_point(problem_prompt: str) -> str | None:
    m = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", problem_prompt or "", flags=re.MULTILINE)
    return m.group(1) if m else None


def artifact_to_executable_for_task(task: TaskInput, artifact_text: str) -> str:
    dataset_type = infer_dataset_type_from_task_id(task.task_id)
    if dataset_type == "humaneval":
        return (compose_humaneval_executable_code(task.problem or "", artifact_text or "").code or "").strip()
    extracted = extract_code_from_artifact_text(artifact_text or "")
    return (extracted.code or artifact_text or "").strip()


def _eval_worker(payload: dict[str, Any], out_q: mp.Queue) -> None:
    try:
        ns: dict[str, Any] = {}
        dataset_type = payload["dataset_type"]
        if dataset_type == "bigcodebench":
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
            return

        if dataset_type == "mbpp":
            exec(payload["completion"], ns, ns)
            if payload["test"]:
                exec(payload["test"], ns, ns)
            out_q.put({"passed": True, "error_type": None, "error": None})
            return

        exec(payload["prompt"], ns, ns)
        exec(payload["completion"], ns, ns)
        exec(payload["test"], ns, ns)
        entry_point = payload.get("entry_point")
        if not entry_point:
            raise RuntimeError("Could not infer HumanEval entry_point.")
        candidate = ns[entry_point]
        ns["check"](candidate)
        out_q.put({"passed": True, "error_type": None, "error": None})
    except Exception as exc:
        out_q.put({"passed": False, "error_type": type(exc).__name__, "error": f"{type(exc).__name__}: {exc}"})


def evaluate_pre_verifier_checkpoint(
    task: TaskInput,
    artifact_text: str,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    if not task.test_harness:
        return {
            "invoked": False,
            "passed": None,
            "error_type": None,
            "error": None,
            "completion": artifact_to_executable_for_task(task, artifact_text),
        }

    dataset_type = infer_dataset_type_from_task_id(task.task_id)
    completion = artifact_to_executable_for_task(task, artifact_text)
    try:
        ast.parse(completion or "")
    except Exception as exc:
        return {
            "invoked": True,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completion": completion,
        }

    payload = {
        "prompt": task.problem or "",
        "completion": completion,
        "test": task.test_harness or "",
        "entry_point": infer_humaneval_entry_point(task.problem or ""),
        "dataset_type": dataset_type,
    }
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_eval_worker, args=(payload, q))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {
            "invoked": True,
            "passed": False,
            "error_type": "TimeoutError",
            "error": f"Timeout after {timeout_s:.1f}s",
            "completion": completion,
        }
    try:
        result = q.get_nowait()
    except Exception:
        result = {"passed": False, "error_type": "NoResultError", "error": "Checkpoint worker returned no result."}
    return {
        "invoked": True,
        "passed": bool(result.get("passed")) if result.get("passed") is not None else None,
        "error_type": result.get("error_type"),
        "error": result.get("error"),
        "completion": completion,
    }
