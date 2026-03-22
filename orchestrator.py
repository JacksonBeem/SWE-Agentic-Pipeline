from __future__ import annotations

import ast
import time
import json
import hashlib
from typing import Any

from .csv_logger import CSVLogger, AgentCallRow, RunRow, now_ts
from .schemas import TaskInput
from .openrouter_client import OpenRouterClient
from .config import AppConfig, PipelineConfig

from .agents.planner import PlannerAgent
from .agents.executor import ExecutorAgent
from .agents.critic import CriticAgent
from .agents.verifier import VerifierAgent
from .utils.artifact_to_code import compose_humaneval_executable_code, extract_code_from_artifact_text
from .checkpoint_eval import evaluate_pre_verifier_checkpoint


def parse_critic_passfail(text: str) -> tuple[bool | None, str]:
    t = (text or "").strip()
    first = t.splitlines()[0].strip().upper() if t else ""
    if first.startswith("PASS"):
        return True, text
    if first.startswith("FAIL"):
        return False, text
    return None, text


def parse_verifier(text: str) -> tuple[str | None, str | None]:
    t = (text or "").strip()
    if t == "ACCEPT":
        return "ACCEPT", None
    if t.startswith("REJECT"):
        parts = t.split(":", 1)
        if len(parts) == 2:
            return "REJECT", parts[1].strip()
        lines = t.splitlines()
        req = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
        return "REJECT", (req or None)
    return None, None


def patch_format_summary(text: str) -> dict:
    t = (text or "").strip()
    return {
        "has_diff_git": ("diff --git" in t),
        "has_file_markers": ("--- " in t and "+++ " in t),
        "has_hunks": ("@@" in t),
        "has_markdown_fence": ("```" in t),
        "looks_like_patch": ("diff --git" in t and "--- " in t and "+++ " in t and "@@" in t),
    }


def infer_artifact_mode(task: TaskInput) -> str:
    """
    code: standalone executable Python artifact (HumanEval/BigCodeBench style)
    patch: unified diff artifact (repo task style)
    """
    rc = (task.repo_context or "").lower()
    if "repo=" in rc:
        return "patch"
    return "code"


def normalize_executable_artifact(artifact_text: str) -> str:
    extracted = extract_code_from_artifact_text(artifact_text or "")
    return (extracted.code or artifact_text or "").strip()


class PipelineOrchestrator:
    def __init__(self, app: AppConfig, pipe: PipelineConfig, logger: CSVLogger):
        self.app = app
        self.pipe = pipe
        self.logger = logger

        client = OpenRouterClient(app.openrouter)

        self.planner = PlannerAgent(client, app.planner_model)
        self.executor = ExecutorAgent(client, app.executor_model)
        self.critic = CriticAgent(client, app.critic_model)
        self.verifier = VerifierAgent(client, app.verifier_model)

    # ---------------------------
    # Logging helpers (LOG EVERYTHING)
    # ---------------------------
    def _json_dumps_safe(self, obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)

    def _extract_messages(self, agent_result: Any, fallback_messages: Any) -> str:
        msgs = getattr(agent_result, "messages", None)
        if msgs is None:
            msgs = fallback_messages
        return self._json_dumps_safe(msgs)

    def _extract_raw_output(self, agent_result: Any) -> str:
        raw = getattr(agent_result, "raw_output", None)
        if raw is None:
            raw = getattr(agent_result, "output_text", "") or ""
        return raw

    def _extract_clean_output(self, agent_result: Any) -> str:
        return getattr(agent_result, "output_text", "") or ""

    def _log_call_success(
        self,
        task_id: str,
        pipeline_config: str,
        agent: str,
        model: str,
        messages: Any,
        agent_result: Any,
    ) -> None:
        llm = getattr(agent_result, "llm", None)

        self.logger.log_agent_call(
            AgentCallRow(
                ts_unix=now_ts(),
                run_id=self.pipe.run_id,
                task_id=task_id,
                pipeline_config=pipeline_config,
                agent=agent,
                model=model,
                messages=self._extract_messages(agent_result, messages),
                raw_output=self._extract_raw_output(agent_result),
                clean_output=self._extract_clean_output(agent_result),
                prompt_tokens=getattr(llm, "prompt_tokens", None),
                completion_tokens=getattr(llm, "completion_tokens", None),
                total_tokens=getattr(llm, "total_tokens", None),
                latency_s=getattr(llm, "latency_s", 0.0) or 0.0,
                error_text=None,
            )
        )

    def _log_call_error(
        self,
        task_id: str,
        pipeline_config: str,
        agent: str,
        model: str,
        messages: Any,
        error_text: str,
    ) -> None:
        self.logger.log_agent_call(
            AgentCallRow(
                ts_unix=now_ts(),
                run_id=self.pipe.run_id,
                task_id=task_id,
                pipeline_config=pipeline_config,
                agent=agent,
                model=model,
                messages=self._json_dumps_safe(messages),
                raw_output="",
                clean_output="",
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                latency_s=0.0,
                error_text=error_text,
            )
        )

    # ---------------------------
    # Main pipeline execution
    # ---------------------------
    def run_task(self, task: TaskInput, pipeline_config: str = "agentic+verifier") -> dict:
        t0 = time.time()
        total_tokens = 0
        verifier_invoked = 0
        verifier_decision = None
        repair_attempted = 0
        origin_stage = None
        is_humaneval = "humaneval" in (pipeline_config or "").lower()
        prompt_text = task.problem or ""
        prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

        planner_answer = None
        executor_answer = None
        critic_answer = None
        verifier_answer = None

        planner_error_text = None
        executor_error_text = None
        critic_error_text = None
        verifier_error_text = None

        planner_prompt_tokens = None
        planner_completion_tokens = None
        planner_total_tokens = None
        planner_latency_s = None

        executor_prompt_tokens_sum = 0
        executor_completion_tokens_sum = 0
        executor_total_tokens_sum = 0
        executor_latency_s_sum = 0.0
        executor_seen = False

        critic_prompt_tokens_sum = 0
        critic_completion_tokens_sum = 0
        critic_total_tokens_sum = 0
        critic_latency_s_sum = 0.0
        critic_seen = False

        verifier_prompt_tokens = None
        verifier_completion_tokens = None
        verifier_total_tokens = None
        verifier_latency_s = None
        pre_verifier_exec_invoked = 0
        pre_verifier_exec_passed: int | None = None
        pre_verifier_exec_error_type = None
        pre_verifier_exec_error = None
        pre_verifier_executable_code = None
        verifier_pre_repair_exec_invoked = 0
        verifier_pre_repair_exec_passed: int | None = None
        verifier_pre_repair_exec_error_type = None
        verifier_pre_repair_exec_error = None
        verifier_post_repair_exec_invoked = 0
        verifier_post_repair_exec_passed: int | None = None
        verifier_post_repair_exec_error_type = None
        verifier_post_repair_exec_error = None
        planner_spec_text = task.problem or ""

        # 1) Planner
        if self.pipe.enable_planner:
            a_messages = None
            try:
                a_messages = self.planner.build_messages(problem=task.problem, repo_context=task.repo_context)
                a = self.planner.run(problem=task.problem, repo_context=task.repo_context)
                self._log_call_success(task.task_id, pipeline_config, "Planner", self.app.planner_model.model, a_messages, a)
                total_tokens += (a.llm.total_tokens or 0)
                planner_answer = a.output_text
                planner_spec_text = a.output_text
                planner_prompt_tokens = a.llm.prompt_tokens
                planner_completion_tokens = a.llm.completion_tokens
                planner_total_tokens = a.llm.total_tokens
                planner_latency_s = a.llm.latency_s
            except Exception as e:
                planner_error_text = str(e)
                self._log_call_error(task.task_id, pipeline_config, "Planner", self.app.planner_model.model, a_messages, str(e))
                raise
        else:
            planner_answer = "PLANNER_BYPASSED: direct prompt sent to Executor."

        # 2) Executor
        d_messages = None
        try:
            d_messages = self.executor.build_messages(problem=task.problem, planner_spec=planner_spec_text, repo_context=task.repo_context)
            d = self.executor.run(problem=task.problem, planner_spec=planner_spec_text, repo_context=task.repo_context)
            self._log_call_success(task.task_id, pipeline_config, "Executor", self.app.executor_model.model, d_messages, d)
            total_tokens += (d.llm.total_tokens or 0)
            executor_answer = d.output_text
            executor_seen = True
            executor_prompt_tokens_sum += int(d.llm.prompt_tokens or 0)
            executor_completion_tokens_sum += int(d.llm.completion_tokens or 0)
            executor_total_tokens_sum += int(d.llm.total_tokens or 0)
            executor_latency_s_sum += float(d.llm.latency_s or 0.0)
        except Exception as e:
            executor_error_text = str(e)
            self._log_call_error(task.task_id, pipeline_config, "Executor", self.app.executor_model.model, d_messages, str(e))
            raise

        artifact_mode = infer_artifact_mode(task)

        # Format checks for verifier gating (especially useful for patch-mode tasks)
        fmt = patch_format_summary(d.output_text)
        format_checks_json = self._json_dumps_safe(fmt)

        # 3) Critic (SKIP if no harness to avoid fake FAILs on SWE tasks)
        q = None
        critic_pass: bool | None = None
        critic_summary: str = "Critic skipped (no test harness provided)."

        critic_artifact = (
            (compose_humaneval_executable_code(task.problem, d.output_text).code or "").strip()
            if is_humaneval
            else d.output_text
        )

        if task.test_harness:
            q_messages = None
            try:
                q_messages = self.critic.build_messages(code_artifact=critic_artifact, test_harness=task.test_harness)
                q = self.critic.run(code_artifact=critic_artifact, test_harness=task.test_harness)
                self._log_call_success(task.task_id, pipeline_config, "Critic", self.app.critic_model.model, q_messages, q)
                total_tokens += (q.llm.total_tokens or 0)
                critic_answer = q.output_text
                critic_seen = True
                critic_prompt_tokens_sum += int(q.llm.prompt_tokens or 0)
                critic_completion_tokens_sum += int(q.llm.completion_tokens or 0)
                critic_total_tokens_sum += int(q.llm.total_tokens or 0)
                critic_latency_s_sum += float(q.llm.latency_s or 0.0)
            except Exception as e:
                critic_error_text = str(e)
                self._log_call_error(task.task_id, pipeline_config, "Critic", self.app.critic_model.model, q_messages, str(e))
                raise

            critic_pass, critic_summary = parse_critic_passfail(q.output_text)

        # 3b) Pre-verifier execution checkpoint (real executable test before verifier/repair)
        if self.pipe.enable_verifier and self.pipe.enable_pre_verifier_checkpoint:
            try:
                checkpoint = evaluate_pre_verifier_checkpoint(task=task, artifact_text=d.output_text, timeout_s=20.0)
                pre_verifier_exec_invoked = 1 if checkpoint.get("invoked") else 0
                ck_pass = checkpoint.get("passed")
                if ck_pass is True:
                    pre_verifier_exec_passed = 1
                elif ck_pass is False:
                    pre_verifier_exec_passed = 0
                else:
                    pre_verifier_exec_passed = None
                pre_verifier_exec_error_type = checkpoint.get("error_type")
                pre_verifier_exec_error = checkpoint.get("error")
                pre_verifier_executable_code = checkpoint.get("completion")
            except Exception as e:
                # Checkpoint logging should not abort generation.
                pre_verifier_exec_invoked = 0
                pre_verifier_exec_passed = None
                pre_verifier_exec_error_type = type(e).__name__
                pre_verifier_exec_error = str(e)

        # --- Trigger policy ---
        disagreement = False

        # Only treat Critic as disagreement if we ran it and got an explicit FAIL.
        if critic_pass is False:
            disagreement = True

        # Optional: treat invalid patch format as "disagreement" so verifier runs under disagreement policy.
        # This keeps verifier central on SWE-Bench even before real execution.
        if fmt.get("has_markdown_fence") or (artifact_mode == "patch" and not fmt.get("looks_like_patch")):
            disagreement = True

        policy = (self.pipe.trigger_policy or "").strip().lower()
        if policy not in {"always", "disagreement", "critic_fail"}:
            policy = "disagreement"

        should_invoke_verifier = self.pipe.enable_planner and self.pipe.enable_verifier and (
            (policy == "always")
            or (policy == "disagreement" and disagreement)
            or (policy == "critic_fail" and critic_pass is False)
        )

        final_artifact = d.output_text

        # 5) Verifier + optional single repair loop
        if should_invoke_verifier:
            verifier_invoked = 1
            v_messages = None
            try:
                v_messages = self.verifier.build_messages(
                    critic_summary=critic_summary,
                    disagreement=disagreement,
                    artifact_mode=artifact_mode,
                    format_checks=format_checks_json,
                )
                v = self.verifier.run(
                    critic_summary=critic_summary,
                    disagreement=disagreement,
                    artifact_mode=artifact_mode,
                    format_checks=format_checks_json,
                )
                self._log_call_success(task.task_id, pipeline_config, "Verifier", self.app.verifier_model.model, v_messages, v)
                total_tokens += (v.llm.total_tokens or 0)
                verifier_answer = v.output_text
                verifier_prompt_tokens = v.llm.prompt_tokens
                verifier_completion_tokens = v.llm.completion_tokens
                verifier_total_tokens = v.llm.total_tokens
                verifier_latency_s = v.llm.latency_s
            except Exception as e:
                verifier_error_text = str(e)
                self._log_call_error(task.task_id, pipeline_config, "Verifier", self.app.verifier_model.model, v_messages, str(e))
                raise

            verifier_decision, repair_request = parse_verifier(v.output_text)

            if verifier_decision == "REJECT" and self.pipe.allow_single_repair and repair_request:
                repair_attempted = 1
                if self.pipe.enable_verifier_repair_checkpoints:
                    try:
                        pre_rep = evaluate_pre_verifier_checkpoint(task=task, artifact_text=final_artifact, timeout_s=20.0)
                        verifier_pre_repair_exec_invoked = 1 if pre_rep.get("invoked") else 0
                        pre_rep_pass = pre_rep.get("passed")
                        if pre_rep_pass is True:
                            verifier_pre_repair_exec_passed = 1
                        elif pre_rep_pass is False:
                            verifier_pre_repair_exec_passed = 0
                        else:
                            verifier_pre_repair_exec_passed = None
                        verifier_pre_repair_exec_error_type = pre_rep.get("error_type")
                        verifier_pre_repair_exec_error = pre_rep.get("error")
                    except Exception as e:
                        verifier_pre_repair_exec_invoked = 0
                        verifier_pre_repair_exec_passed = None
                        verifier_pre_repair_exec_error_type = type(e).__name__
                        verifier_pre_repair_exec_error = str(e)

                d2_messages = None
                try:
                    d2_messages = self.executor.build_messages(
                        problem=task.problem,
                        planner_spec=planner_spec_text + "\n\nREPAIR REQUEST FROM VERIFIER:\n" + repair_request,
                        repo_context=task.repo_context,
                    )
                    d2 = self.executor.run(
                        problem=task.problem,
                        planner_spec=planner_spec_text + "\n\nREPAIR REQUEST FROM VERIFIER:\n" + repair_request,
                        repo_context=task.repo_context,
                    )
                    self._log_call_success(task.task_id, pipeline_config, "Executor", self.app.executor_model.model, d2_messages, d2)
                    total_tokens += (d2.llm.total_tokens or 0)
                    executor_answer = d2.output_text
                    executor_seen = True
                    executor_prompt_tokens_sum += int(d2.llm.prompt_tokens or 0)
                    executor_completion_tokens_sum += int(d2.llm.completion_tokens or 0)
                    executor_total_tokens_sum += int(d2.llm.total_tokens or 0)
                    executor_latency_s_sum += float(d2.llm.latency_s or 0.0)
                    final_artifact = d2.output_text
                    if self.pipe.enable_verifier_repair_checkpoints:
                        try:
                            post_rep = evaluate_pre_verifier_checkpoint(task=task, artifact_text=final_artifact, timeout_s=20.0)
                            verifier_post_repair_exec_invoked = 1 if post_rep.get("invoked") else 0
                            post_rep_pass = post_rep.get("passed")
                            if post_rep_pass is True:
                                verifier_post_repair_exec_passed = 1
                            elif post_rep_pass is False:
                                verifier_post_repair_exec_passed = 0
                            else:
                                verifier_post_repair_exec_passed = None
                            verifier_post_repair_exec_error_type = post_rep.get("error_type")
                            verifier_post_repair_exec_error = post_rep.get("error")
                        except Exception as e:
                            verifier_post_repair_exec_invoked = 0
                            verifier_post_repair_exec_passed = None
                            verifier_post_repair_exec_error_type = type(e).__name__
                            verifier_post_repair_exec_error = str(e)
                    if is_humaneval and task.test_harness:
                        # Re-run Critic on the repaired executable artifact so Critic and external eval align.
                        repaired_exec = (compose_humaneval_executable_code(task.problem, final_artifact).code or "").strip()
                        q2_messages = self.critic.build_messages(code_artifact=repaired_exec, test_harness=task.test_harness)
                        q2 = self.critic.run(code_artifact=repaired_exec, test_harness=task.test_harness)
                        self._log_call_success(task.task_id, pipeline_config, "Critic", self.app.critic_model.model, q2_messages, q2)
                        total_tokens += (q2.llm.total_tokens or 0)
                        critic_answer = q2.output_text
                        critic_seen = True
                        critic_prompt_tokens_sum += int(q2.llm.prompt_tokens or 0)
                        critic_completion_tokens_sum += int(q2.llm.completion_tokens or 0)
                        critic_total_tokens_sum += int(q2.llm.total_tokens or 0)
                        critic_latency_s_sum += float(q2.llm.latency_s or 0.0)
                except Exception as e:
                    executor_error_text = str(e)
                    self._log_call_error(task.task_id, pipeline_config, "Executor", self.app.executor_model.model, d2_messages, str(e))
                    raise

        final_executable_code = (
            (compose_humaneval_executable_code(task.problem, final_artifact).code or "").strip()
            if is_humaneval
            else (final_artifact or "").strip()
        )
        parse_error_type = None
        parse_error_text = None
        try:
            ast.parse(final_executable_code or "")
        except Exception as e:
            parse_error_type = type(e).__name__
            parse_error_text = str(e)

        # HumanEval hardening: one syntax-only repair pass to prevent malformed artifacts
        # (e.g., unterminated triple-quoted fragments) from becoming final outputs.
        if is_humaneval and parse_error_type:
            syntax_spec = (
                "SYNTAX REPAIR REQUEST (HumanEval):\n"
                "Return ONLY complete executable Python code for this task.\n"
                "Do not output diff format, markdown fences, or explanations.\n"
                f"Current parse error: {parse_error_type}: {parse_error_text}"
            )
            d3_messages = None
            try:
                d3_messages = self.executor.build_messages(
                    problem=task.problem,
                    planner_spec=syntax_spec,
                    repo_context=None,
                )
                d3 = self.executor.run(
                    problem=task.problem,
                    planner_spec=syntax_spec,
                    repo_context=None,
                )
                self._log_call_success(
                    task.task_id,
                    pipeline_config,
                    "Executor",
                    self.app.executor_model.model,
                    d3_messages,
                    d3,
                )
                total_tokens += (d3.llm.total_tokens or 0)
                executor_answer = d3.output_text
                executor_seen = True
                executor_prompt_tokens_sum += int(d3.llm.prompt_tokens or 0)
                executor_completion_tokens_sum += int(d3.llm.completion_tokens or 0)
                executor_total_tokens_sum += int(d3.llm.total_tokens or 0)
                executor_latency_s_sum += float(d3.llm.latency_s or 0.0)

                repaired_exec = (compose_humaneval_executable_code(task.problem, d3.output_text).code or "").strip()
                ast.parse(repaired_exec or "")
                final_artifact = d3.output_text
                final_executable_code = repaired_exec
                parse_error_type = None
                parse_error_text = None
                repair_attempted = 1
            except Exception as e:
                executor_error_text = str(e)
                self._log_call_error(
                    task.task_id,
                    pipeline_config,
                    "Executor",
                    self.app.executor_model.model,
                    d3_messages,
                    str(e),
                )

        end_to_end = time.time() - t0

        self.logger.log_run(
            RunRow(
                ts_unix=now_ts(),
                run_id=self.pipe.run_id,
                task_id=task.task_id,
                pipeline_config=pipeline_config,
                trigger_policy=self.pipe.trigger_policy,
                verifier_invoked=verifier_invoked,
                verifier_decision=verifier_decision,
                repair_attempted=repair_attempted,
                final_correct=None,
                origin_stage=origin_stage,
                total_tokens=total_tokens,
                end_to_end_latency_s=end_to_end,
                final_executable_code=final_executable_code,
                parse_error_type=parse_error_type,
                parse_error_text=parse_error_text,
                config=pipeline_config,
                prompt=prompt_text,
                prompt_hash=prompt_hash,
                planner=(self.app.planner_model.model if self.pipe.enable_planner else None),
                executor=self.app.executor_model.model,
                critic=self.app.critic_model.model,
                verifier=self.app.verifier_model.model if verifier_invoked else None,
                final_answer=final_executable_code,
                planner_prompt_tokens=planner_prompt_tokens,
                planner_completion_tokens=planner_completion_tokens,
                planner_total_tokens=planner_total_tokens,
                planner_latency_s=planner_latency_s,
                executor_prompt_tokens=(executor_prompt_tokens_sum if executor_seen else None),
                executor_completion_tokens=(executor_completion_tokens_sum if executor_seen else None),
                executor_total_tokens=(executor_total_tokens_sum if executor_seen else None),
                executor_latency_s=(executor_latency_s_sum if executor_seen else None),
                critic_prompt_tokens=(critic_prompt_tokens_sum if critic_seen else None),
                critic_completion_tokens=(critic_completion_tokens_sum if critic_seen else None),
                critic_total_tokens=(critic_total_tokens_sum if critic_seen else None),
                critic_latency_s=(critic_latency_s_sum if critic_seen else None),
                verifier_prompt_tokens=verifier_prompt_tokens,
                verifier_completion_tokens=verifier_completion_tokens,
                verifier_total_tokens=verifier_total_tokens,
                verifier_latency_s=verifier_latency_s,
                planner_answer=planner_answer,
                executor_answer=executor_answer,
                critic_answer=critic_answer,
                verifier_answer=verifier_answer,
                planner_error_text=planner_error_text,
                executor_error_text=executor_error_text,
                critic_error_text=critic_error_text,
                verifier_error_text=verifier_error_text,
                correct_answer=None,
                planner_error=(1 if planner_error_text else 0),
                executor_repair=repair_attempted,
                executor_harm=(1 if parse_error_type else 0),
                verifier_repair=(1 if verifier_decision == "REJECT" else 0),
                verifier_harm=(1 if (verifier_invoked and parse_error_type) else 0),
                pre_verifier_exec_invoked=pre_verifier_exec_invoked,
                pre_verifier_exec_passed=pre_verifier_exec_passed,
                pre_verifier_exec_error_type=pre_verifier_exec_error_type,
                pre_verifier_exec_error=pre_verifier_exec_error,
                verifier_pre_repair_exec_invoked=verifier_pre_repair_exec_invoked,
                verifier_pre_repair_exec_passed=verifier_pre_repair_exec_passed,
                verifier_pre_repair_exec_error_type=verifier_pre_repair_exec_error_type,
                verifier_pre_repair_exec_error=verifier_pre_repair_exec_error,
                verifier_post_repair_exec_invoked=verifier_post_repair_exec_invoked,
                verifier_post_repair_exec_passed=verifier_post_repair_exec_passed,
                verifier_post_repair_exec_error_type=verifier_post_repair_exec_error_type,
                verifier_post_repair_exec_error=verifier_post_repair_exec_error,
            )
        )

        return {
            "task_id": task.task_id,
            "verifier_invoked": bool(verifier_invoked),
            "verifier_decision": verifier_decision,
            "repair_attempted": bool(repair_attempted),
            "final_artifact": final_artifact,
            "final_executable_code": final_executable_code,
            "parse_error_type": parse_error_type,
            "parse_error_text": parse_error_text,
            "total_tokens": total_tokens,
            "end_to_end_latency_s": end_to_end,
            "pre_verifier_exec_invoked": bool(pre_verifier_exec_invoked),
            "pre_verifier_exec_passed": (None if pre_verifier_exec_passed is None else bool(pre_verifier_exec_passed)),
            "pre_verifier_exec_error_type": pre_verifier_exec_error_type,
            "pre_verifier_exec_error": pre_verifier_exec_error,
            "pre_verifier_executable_code": pre_verifier_executable_code,
            "pre_verifier_artifact": d.output_text,
        }

