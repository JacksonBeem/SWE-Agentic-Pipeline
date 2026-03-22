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
    planner: str | None = None
    executor: str | None = None
    critic: str | None = None
    verifier: str | None = None
    final_answer: str | None = None
    planner_prompt_tokens: int | None = None
    planner_completion_tokens: int | None = None
    planner_total_tokens: int | None = None
    planner_latency_s: float | None = None
    executor_prompt_tokens: int | None = None
    executor_completion_tokens: int | None = None
    executor_total_tokens: int | None = None
    executor_latency_s: float | None = None
    critic_prompt_tokens: int | None = None
    critic_completion_tokens: int | None = None
    critic_total_tokens: int | None = None
    critic_latency_s: float | None = None
    verifier_prompt_tokens: int | None = None
    verifier_completion_tokens: int | None = None
    verifier_total_tokens: int | None = None
    verifier_latency_s: float | None = None
    planner_answer: str | None = None
    executor_answer: str | None = None
    critic_answer: str | None = None
    verifier_answer: str | None = None
    planner_error_text: str | None = None
    executor_error_text: str | None = None
    critic_error_text: str | None = None
    verifier_error_text: str | None = None
    correct_answer: str | None = None
    planner_error: int | None = None
    executor_repair: int | None = None
    executor_harm: int | None = None
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
        "planner",
        "executor",
        "critic",
        "verifier",
        "final_answer",
        "planner_prompt_tokens",
        "planner_completion_tokens",
        "planner_total_tokens",
        "planner_latency_s",
        "executor_prompt_tokens",
        "executor_completion_tokens",
        "executor_total_tokens",
        "executor_latency_s",
        "critic_prompt_tokens",
        "critic_completion_tokens",
        "critic_total_tokens",
        "critic_latency_s",
        "verifier_prompt_tokens",
        "verifier_completion_tokens",
        "verifier_total_tokens",
        "verifier_latency_s",
        "planner_answer",
        "executor_answer",
        "critic_answer",
        "verifier_answer",
        "planner_error_text",
        "executor_error_text",
        "critic_error_text",
        "verifier_error_text",
        "correct_answer",
        "planner_error",
        "executor_repair",
        "executor_harm",
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
            "planner": d.get("planner"),
            "executor": d.get("executor"),
            "critic": d.get("critic"),
            "verifier": d.get("verifier"),
            "final_answer": d.get("final_answer"),
            "planner_prompt_tokens": d.get("planner_prompt_tokens"),
            "planner_completion_tokens": d.get("planner_completion_tokens"),
            "planner_total_tokens": d.get("planner_total_tokens"),
            "planner_latency_s": d.get("planner_latency_s"),
            "executor_prompt_tokens": d.get("executor_prompt_tokens"),
            "executor_completion_tokens": d.get("executor_completion_tokens"),
            "executor_total_tokens": d.get("executor_total_tokens"),
            "executor_latency_s": d.get("executor_latency_s"),
            "critic_prompt_tokens": d.get("critic_prompt_tokens"),
            "critic_completion_tokens": d.get("critic_completion_tokens"),
            "critic_total_tokens": d.get("critic_total_tokens"),
            "critic_latency_s": d.get("critic_latency_s"),
            "verifier_prompt_tokens": d.get("verifier_prompt_tokens"),
            "verifier_completion_tokens": d.get("verifier_completion_tokens"),
            "verifier_total_tokens": d.get("verifier_total_tokens"),
            "verifier_latency_s": d.get("verifier_latency_s"),
            "planner_answer": d.get("planner_answer"),
            "executor_answer": d.get("executor_answer"),
            "critic_answer": d.get("critic_answer"),
            "verifier_answer": d.get("verifier_answer"),
            "planner_error_text": d.get("planner_error_text"),
            "executor_error_text": d.get("executor_error_text"),
            "critic_error_text": d.get("critic_error_text"),
            "verifier_error_text": d.get("verifier_error_text"),
            "correct_answer": d.get("correct_answer"),
            "planner_error": d.get("planner_error"),
            "executor_repair": d.get("executor_repair"),
            "executor_harm": d.get("executor_harm"),
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

