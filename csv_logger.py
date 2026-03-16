from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable


def now_ts() -> float:
    return time.time()


@dataclass
class AgentCallRow:
    ts_unix: float
    run_id: str
    task_id: str
    pipeline_config: str
    agent: str
    model: str
    messages: str
    raw_output: str
    clean_output: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_s: float
    error_text: str | None


@dataclass
class RunRow:
    ts_unix: float
    run_id: str
    task_id: str
    pipeline_config: str
    trigger_policy: str
    verifier_invoked: int
    verifier_decision: str | None
    repair_attempted: int
    final_correct: int | None
    origin_stage: str | None
    total_tokens: int | None
    end_to_end_latency_s: float
    final_executable_code: str | None = None
    parse_error_type: str | None = None
    parse_error_text: str | None = None
    config: str | None = None
    prompt: str | None = None
    prompt_hash: str | None = None
    architect: str | None = None
    developer: str | None = None
    qa: str | None = None
    verifier: str | None = None
    final_answer: str | None = None
    architect_prompt_tokens: int | None = None
    architect_completion_tokens: int | None = None
    architect_total_tokens: int | None = None
    architect_latency_s: float | None = None
    developer_prompt_tokens: int | None = None
    developer_completion_tokens: int | None = None
    developer_total_tokens: int | None = None
    developer_latency_s: float | None = None
    qa_prompt_tokens: int | None = None
    qa_completion_tokens: int | None = None
    qa_total_tokens: int | None = None
    qa_latency_s: float | None = None
    verifier_prompt_tokens: int | None = None
    verifier_completion_tokens: int | None = None
    verifier_total_tokens: int | None = None
    verifier_latency_s: float | None = None
    architect_answer: str | None = None
    developer_answer: str | None = None
    qa_answer: str | None = None
    verifier_answer: str | None = None
    architect_error_text: str | None = None
    developer_error_text: str | None = None
    qa_error_text: str | None = None
    verifier_error_text: str | None = None
    correct_answer: str | None = None
    architect_error: int | None = None
    developer_repair: int | None = None
    developer_harm: int | None = None
    verifier_repair: int | None = None
    verifier_harm: int | None = None
    pre_verifier_exec_invoked: int | None = None
    pre_verifier_exec_passed: int | None = None
    pre_verifier_exec_error_type: str | None = None
    pre_verifier_exec_error: str | None = None
    verifier_pre_repair_exec_invoked: int | None = None
    verifier_pre_repair_exec_passed: int | None = None
    verifier_pre_repair_exec_error_type: str | None = None
    verifier_pre_repair_exec_error: str | None = None
    verifier_post_repair_exec_invoked: int | None = None
    verifier_post_repair_exec_passed: int | None = None
    verifier_post_repair_exec_error_type: str | None = None
    verifier_post_repair_exec_error: str | None = None


class CSVLogger:
    """
    Single-file logger:
      pipeline/logs/runs.csv

    It stores:
      - AGENT_CALL rows (per agent invocation)
      - RUN_SUMMARY row (per task run)

    Use event_type column to filter.
    """

    # Union schema (superset of both agent + run fields)
    FIELDNAMES = [
        # common
        "event_type",  # "AGENT_CALL" | "RUN_SUMMARY"
        "ts_unix",
        "run_id",
        "task_id",
        "pipeline_config",

        # agent-call fields
        "agent",
        "model",
        "messages",
        "raw_output",
        "clean_output",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_s",
        "error_text",

        # run-summary fields
        "trigger_policy",
        "verifier_invoked",
        "verifier_decision",
        "repair_attempted",
        "final_correct",
        "origin_stage",
        "run_total_tokens",        # alias to avoid confusion w/ per-call total_tokens
        "end_to_end_latency_s",
        "final_executable_code",
        "parse_error_type",
        "parse_error_text",
        "config",
        "prompt",
        "prompt_hash",
        "architect",
        "developer",
        "qa",
        "verifier",
        "final_answer",
        "architect_prompt_tokens",
        "architect_completion_tokens",
        "architect_total_tokens",
        "architect_latency_s",
        "developer_prompt_tokens",
        "developer_completion_tokens",
        "developer_total_tokens",
        "developer_latency_s",
        "qa_prompt_tokens",
        "qa_completion_tokens",
        "qa_total_tokens",
        "qa_latency_s",
        "verifier_prompt_tokens",
        "verifier_completion_tokens",
        "verifier_total_tokens",
        "verifier_latency_s",
        "architect_answer",
        "developer_answer",
        "qa_answer",
        "verifier_answer",
        "architect_error_text",
        "developer_error_text",
        "qa_error_text",
        "verifier_error_text",
        "correct_answer",
        "architect_error",
        "developer_repair",
        "developer_harm",
        "verifier_repair",
        "verifier_harm",
        "pre_verifier_exec_invoked",
        "pre_verifier_exec_passed",
        "pre_verifier_exec_error_type",
        "pre_verifier_exec_error",
        "verifier_pre_repair_exec_invoked",
        "verifier_pre_repair_exec_passed",
        "verifier_pre_repair_exec_error_type",
        "verifier_pre_repair_exec_error",
        "verifier_post_repair_exec_invoked",
        "verifier_post_repair_exec_passed",
        "verifier_post_repair_exec_error_type",
        "verifier_post_repair_exec_error",
    ]

    def __init__(self, out_dir: str = "logs"):
        # Anchor to pipeline/ directory so it always writes to C:\VScode\pipeline\logs
        pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        self.out_dir = os.path.join(pipeline_dir, out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.runs_path = os.path.join(self.out_dir, "runs.csv")
        self._ensure_csv(self.runs_path, self.FIELDNAMES)

    def _ensure_csv(self, path: str, fieldnames: Iterable[str]) -> None:
        needs_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if needs_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(fieldnames))
                w.writeheader()
                f.flush()
                os.fsync(f.fileno())
            return

        # Schema evolution: if header changed, rewrite file with new header while preserving data.
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])

        expected = list(fieldnames)
        if header == expected:
            return

        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=expected)
            writer.writeheader()
            for row in rows:
                out = {k: row.get(k, "") for k in expected}
                writer.writerow(out)
            f.flush()
            os.fsync(f.fileno())

    def _write_row(self, row: Dict[str, Any]) -> None:
        self._ensure_csv(self.runs_path, self.FIELDNAMES)

        # Fill missing columns with blanks so the CSV is consistent
        full = {k: "" for k in self.FIELDNAMES}
        full.update(row)

        with open(self.runs_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            w.writerow(full)
            f.flush()
            os.fsync(f.fileno())

    def log_agent_call(self, row: AgentCallRow) -> None:
        d = asdict(row)
        self._write_row({
            "event_type": "AGENT_CALL",
            **d,
            # run-summary-only fields left blank
        })

    def log_run(self, row: RunRow) -> None:
        d = asdict(row)
        # store run total tokens in a dedicated column to avoid confusion with per-call total_tokens
        run_total_tokens = d.get("total_tokens", "")

        self._write_row({
            "event_type": "RUN_SUMMARY",
            "ts_unix": d["ts_unix"],
            "run_id": d["run_id"],
            "task_id": d["task_id"],
            "pipeline_config": d["pipeline_config"],
            "trigger_policy": d["trigger_policy"],
            "verifier_invoked": d["verifier_invoked"],
            "verifier_decision": d["verifier_decision"],
            "repair_attempted": d["repair_attempted"],
            "final_correct": d["final_correct"],
            "origin_stage": d["origin_stage"],
            "run_total_tokens": run_total_tokens,
            "end_to_end_latency_s": d["end_to_end_latency_s"],
            "final_executable_code": d.get("final_executable_code"),
            "parse_error_type": d.get("parse_error_type"),
            "parse_error_text": d.get("parse_error_text"),
            "config": d.get("config"),
            "prompt": d.get("prompt"),
            "prompt_hash": d.get("prompt_hash"),
            "architect": d.get("architect"),
            "developer": d.get("developer"),
            "qa": d.get("qa"),
            "verifier": d.get("verifier"),
            "final_answer": d.get("final_answer"),
            "architect_prompt_tokens": d.get("architect_prompt_tokens"),
            "architect_completion_tokens": d.get("architect_completion_tokens"),
            "architect_total_tokens": d.get("architect_total_tokens"),
            "architect_latency_s": d.get("architect_latency_s"),
            "developer_prompt_tokens": d.get("developer_prompt_tokens"),
            "developer_completion_tokens": d.get("developer_completion_tokens"),
            "developer_total_tokens": d.get("developer_total_tokens"),
            "developer_latency_s": d.get("developer_latency_s"),
            "qa_prompt_tokens": d.get("qa_prompt_tokens"),
            "qa_completion_tokens": d.get("qa_completion_tokens"),
            "qa_total_tokens": d.get("qa_total_tokens"),
            "qa_latency_s": d.get("qa_latency_s"),
            "verifier_prompt_tokens": d.get("verifier_prompt_tokens"),
            "verifier_completion_tokens": d.get("verifier_completion_tokens"),
            "verifier_total_tokens": d.get("verifier_total_tokens"),
            "verifier_latency_s": d.get("verifier_latency_s"),
            "architect_answer": d.get("architect_answer"),
            "developer_answer": d.get("developer_answer"),
            "qa_answer": d.get("qa_answer"),
            "verifier_answer": d.get("verifier_answer"),
            "architect_error_text": d.get("architect_error_text"),
            "developer_error_text": d.get("developer_error_text"),
            "qa_error_text": d.get("qa_error_text"),
            "verifier_error_text": d.get("verifier_error_text"),
            "correct_answer": d.get("correct_answer"),
            "architect_error": d.get("architect_error"),
            "developer_repair": d.get("developer_repair"),
            "developer_harm": d.get("developer_harm"),
            "verifier_repair": d.get("verifier_repair"),
            "verifier_harm": d.get("verifier_harm"),
            "pre_verifier_exec_invoked": d.get("pre_verifier_exec_invoked"),
            "pre_verifier_exec_passed": d.get("pre_verifier_exec_passed"),
            "pre_verifier_exec_error_type": d.get("pre_verifier_exec_error_type"),
            "pre_verifier_exec_error": d.get("pre_verifier_exec_error"),
            "verifier_pre_repair_exec_invoked": d.get("verifier_pre_repair_exec_invoked"),
            "verifier_pre_repair_exec_passed": d.get("verifier_pre_repair_exec_passed"),
            "verifier_pre_repair_exec_error_type": d.get("verifier_pre_repair_exec_error_type"),
            "verifier_pre_repair_exec_error": d.get("verifier_pre_repair_exec_error"),
            "verifier_post_repair_exec_invoked": d.get("verifier_post_repair_exec_invoked"),
            "verifier_post_repair_exec_passed": d.get("verifier_post_repair_exec_passed"),
            "verifier_post_repair_exec_error_type": d.get("verifier_post_repair_exec_error_type"),
            "verifier_post_repair_exec_error": d.get("verifier_post_repair_exec_error"),
        })
