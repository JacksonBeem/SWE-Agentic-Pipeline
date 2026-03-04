from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import uuid
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    ModelConfig,
    PipelineConfig,
    get_active_dataset_path,
    get_active_dataset_type,
    get_default_paths_for_dataset,
    get_workflow_defaults,
    load_config,
)
from pipeline.csv_logger import AgentCallRow, CSVLogger, RunRow, now_ts
from pipeline.dataset_utils import (
    detect_dataset_type,
    mbpp_test_harness,
    task_entry_point_for_dataset,
    task_id_for_row,
    task_prompt_for_dataset,
)
from pipeline.io_utils import append_prediction, iter_jsonl
from pipeline.openrouter_client import OpenRouterClient
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.schemas import TaskInput
from pipeline.utils.artifact_to_code import extract_code_from_artifact_text

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
DEFAULT_PREDICTIONS_PATH = resolve_path_from_config(
    get_default_paths_for_dataset(get_active_dataset_type())[0]
)

def load_completed_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    done: set[str] = set()
    for row in iter_jsonl(path):
        task_id = str(row.get("task_id", "")).strip()
        if task_id:
            done.add(task_id)
    return done


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pipeline over HumanEval tasks in batch.")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=None, help="Inclusive end index.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip tasks already present in predictions file.",
    )
    p.add_argument(
        "--reset-predictions",
        action="store_true",
        help="Delete predictions output before run.",
    )
    p.add_argument(
        "--mode",
        choices=["pipeline", "monolithic"],
        default=WORKFLOW_DEFAULTS.mode,
        help="pipeline=multi-agent orchestrator, monolithic=single LLM call per task.",
    )
    p.add_argument(
        "--skip-security",
        action=argparse.BooleanOptionalAction,
        default=(not WORKFLOW_DEFAULTS.enable_security),
        help="When mode=pipeline, skip the Security agent stage.",
    )
    p.add_argument(
        "--skip-verifier",
        action=argparse.BooleanOptionalAction,
        default=(not WORKFLOW_DEFAULTS.enable_verifier),
        help="When mode=pipeline, skip the Verifier stage.",
    )
    return p.parse_args()


MONOLITHIC_SYSTEM_PROMPT = """You are a disciplined software engineering system operating in three internal roles:

1) ARCHITECT
2) DEVELOPER
3) QA

You must execute these roles sequentially and strictly.

ROLE DEFINITIONS:

ARCHITECT:
- Interpret the problem.
- Define intent.
- Identify constraints.
- Identify edge cases.
- Define acceptance criteria.
- Do NOT write code.

DEVELOPER:
- Implement executable Python code that satisfies the ARCHITECT specification.
- Follow constraints strictly.
- Avoid unnecessary complexity.
- Do NOT reference ground truth solutions.
- Output must define exactly one function matching the ENTRY_POINT.

QA:
- Simulate execution of the function mentally.
- Test the function against acceptance criteria.
- Check edge cases.
- Look for logical errors, off-by-one errors, sorting assumptions, mutation side effects, incorrect inequality conditions, etc.
- If the implementation is incorrect, minimally repair it.
- If repaired, re-evaluate mentally.
- Repeat until the solution satisfies the acceptance criteria.
- QA may only modify code to fix correctness errors.
- QA must not redesign the algorithm.

CRITICAL RULES:

- Do not mention the canonical solution.
- Do not use external references.
- Do not add explanations outside structured sections.
- Do not include docstrings or narrative text in the code.
- FINAL_CODE must contain only valid executable Python with no markdown fences.
- No additional commentary after FINAL_CODE.
"""


def build_monolithic_messages(prompt: str, entry_point: str | None = None) -> list[dict[str, str]]:
    ep = (entry_point or "").strip()
    ep_line = f"ENTRY_POINT: {ep}\n\n" if ep else ""
    return [
        {
            "role": "system",
            "content": MONOLITHIC_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"{ep_line}"
                "Problem:\n\n"
                f"{prompt}\n\n"
                "Now execute ARCHITECT -> DEVELOPER -> QA internally.\n"
                "Output only FINAL_CODE as raw executable Python (no markdown fences, no extra text)."
            ),
        },
    ]


def normalize_generated_code(text: str) -> str:
    extracted = extract_code_from_artifact_text(text or "")
    return (extracted.code or text or "").strip()


def main() -> int:
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset.resolve()}")

    if args.reset_predictions and args.predictions.exists():
        args.predictions.unlink()

    rows = list(iter_jsonl(args.dataset))
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")
    if args.end_idx is not None and args.end_idx < args.start_idx:
        raise ValueError("--end-idx must be >= --start-idx")

    completed = load_completed_task_ids(args.predictions) if args.skip_existing else set()
    if rows:
        inferred_type = detect_dataset_type(rows[0])
        default_pred, _, _ = get_default_paths_for_dataset(inferred_type)
        configured_default_pred = DEFAULT_PREDICTIONS_PATH
        if args.predictions == configured_default_pred:
            args.predictions = resolve_path_from_config(default_pred)
            completed = load_completed_task_ids(args.predictions) if args.skip_existing else set()

    run_id = f"run-{uuid.uuid4().hex[:8]}"
    app = load_config(run_id=run_id)
    pipe_cfg = None
    logger = CSVLogger(out_dir="logs")
    orch = None
    mono_client = None
    if args.mode == "pipeline":
        pipe_cfg = PipelineConfig(
            run_id=run_id,
            traceable=True,
            trigger_policy="always",
            allow_single_repair=True,
            enable_security=(not args.skip_security),
            enable_verifier=(not args.skip_verifier),
        )
        orch = PipelineOrchestrator(app, pipe_cfg, logger)
    else:
        mono_client = OpenRouterClient(app.openrouter)

    total = len(rows)
    processed = 0
    skipped_existing = 0
    failures = 0

    for idx, row in enumerate(rows):
        if idx < args.start_idx:
            continue
        if args.end_idx is not None and idx > args.end_idx:
            break
        if args.limit is not None and processed >= args.limit:
            break

        dataset_type = detect_dataset_type(row)
        task_id = task_id_for_row(row, dataset_type)
        if not task_id:
            failures += 1
            print(f"[{idx + 1}/{total}] FAILED <missing-task-id>: ValueError: missing task id")
            continue
        if task_id in completed:
            skipped_existing += 1
            continue

        prompt = task_prompt_for_dataset(row, dataset_type)
        entry_point = task_entry_point_for_dataset(row, dataset_type)
        test = mbpp_test_harness(row) if dataset_type == "mbpp" else row.get("test", "")

        task = TaskInput(
            task_id=task_id,
            problem=prompt,
            repo_context=(None if args.mode == "pipeline" else (f"entry_point={entry_point}" if entry_point else None)),
            test_harness=(test or None),
        )

        print(f"[{idx + 1}/{total}] Running {task_id} ...")
        t0 = time.time()
        try:
            if args.mode == "pipeline":
                if args.skip_security and args.skip_verifier:
                    pipe_name = f"{dataset_type}_no_security_no_verifier"
                elif args.skip_security:
                    pipe_name = f"{dataset_type}_no_security"
                elif args.skip_verifier:
                    pipe_name = f"{dataset_type}_no_verifier"
                else:
                    pipe_name = dataset_type
                out = orch.run_task(task, pipeline_config=pipe_name)
                completion = out.get("final_executable_code", "") or out.get("final_artifact", "") or ""
            else:
                messages = build_monolithic_messages(prompt=prompt, entry_point=entry_point)
                # BigCodeBench completions can be long; give monolithic extra budget to avoid truncation.
                mono_cfg = ModelConfig(
                    model=app.developer_model.model,
                    temperature=app.developer_model.temperature,
                    max_tokens=6000,
                )
                llm = mono_client.chat(mono_cfg, messages)
                completion = normalize_generated_code(llm.text or "")
                parse_error_type = None
                parse_error_text = None
                try:
                    ast.parse(completion or "")
                except Exception as e:
                    parse_error_type = type(e).__name__
                    parse_error_text = str(e)
                logger.log_agent_call(
                    AgentCallRow(
                        ts_unix=now_ts(),
                        run_id=run_id,
                        task_id=task_id,
                        pipeline_config="humaneval_monolithic",
                        agent="Monolithic",
                        model=app.developer_model.model,
                        messages=json.dumps(messages, ensure_ascii=False),
                        raw_output=json.dumps(llm.raw, ensure_ascii=False),
                        clean_output=completion,
                        prompt_tokens=llm.prompt_tokens,
                        completion_tokens=llm.completion_tokens,
                        total_tokens=llm.total_tokens,
                        latency_s=llm.latency_s,
                        error_text=None,
                    )
                )
            append_prediction(
                path=args.predictions,
                task_id=task_id,
                completion=completion,
                model_name=app.developer_model.model,
            )
            if args.mode == "monolithic":
                logger.log_run(
                    RunRow(
                        ts_unix=now_ts(),
                        run_id=run_id,
                        task_id=task_id,
                        pipeline_config="humaneval_monolithic",
                        trigger_policy="monolithic",
                        verifier_invoked=0,
                        verifier_decision=None,
                        repair_attempted=0,
                        final_correct=None,
                        origin_stage="Monolithic",
                        total_tokens=llm.total_tokens,
                        end_to_end_latency_s=time.time() - t0,
                        final_executable_code=completion,
                        parse_error_type=parse_error_type,
                        parse_error_text=parse_error_text,
                    )
                )
            processed += 1
            print(f"[{idx + 1}/{total}] Done {task_id}")
        except Exception as exc:
            failures += 1
            if args.mode == "monolithic":
                messages = build_monolithic_messages(prompt=prompt, entry_point=entry_point)
                logger.log_agent_call(
                    AgentCallRow(
                        ts_unix=now_ts(),
                        run_id=run_id,
                        task_id=task_id,
                        pipeline_config="humaneval_monolithic",
                        agent="Monolithic",
                        model=app.developer_model.model,
                        messages=json.dumps(messages, ensure_ascii=False),
                        raw_output="",
                        clean_output="",
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        latency_s=0.0,
                        error_text=f"{type(exc).__name__}: {exc}",
                    )
                )
                logger.log_run(
                    RunRow(
                        ts_unix=now_ts(),
                        run_id=run_id,
                        task_id=task_id,
                        pipeline_config="humaneval_monolithic",
                        trigger_policy="monolithic",
                        verifier_invoked=0,
                        verifier_decision=None,
                        repair_attempted=0,
                        final_correct=0,
                        origin_stage="Monolithic",
                        total_tokens=None,
                        end_to_end_latency_s=time.time() - t0,
                    )
                )
            print(f"[{idx + 1}/{total}] FAILED {task_id}: {type(exc).__name__}: {exc}")

    print("Batch run complete.")
    print(f"processed: {processed}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"failures: {failures}")
    print(f"predictions_file: {args.predictions}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
