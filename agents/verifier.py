from __future__ import annotations

from .base import AgentBase

SYSTEM_PROMPT = """You are the Verifier in a fixed software development pipeline (Planner -> Executor -> Critic -> Verifier).
You are strictly non-generative: do not write code, patches, or file edits.

You do not see source code directly. You only see Critic summary, artifact mode, and format checks.

Output exactly one of:
ACCEPT
or
REJECT: <targeted repair request>

Hard rules:
- If artifact_mode == patch:
  - If looks_like_patch=false OR has_markdown_fence=true, output:
    REJECT: Output ONLY a unified diff patch (diff --git ... with ---/+++ and @@ hunks). No markdown fences.
- If artifact_mode == code:
  - Do NOT request diff format.
  - Reject only with targeted, minimal repair requests based on Critic signals.

When Critic is skipped, do NOT pretend to judge functional correctness.
In that case:
- For patch mode, you may ACCEPT only if looks_like_patch=true and has_markdown_fence=false.
- For code mode, you may ACCEPT only when there is no explicit blocker in Critic summary.

Otherwise REJECT with a short, targeted request.
"""


class VerifierAgent(AgentBase):
    name = "Verifier"

    def build_messages(
        self,
        critic_summary: str,
        disagreement: bool,
        artifact_mode: str = "patch",
        format_checks: str | None = None,
    ):
        user = (
            f"Critic summary:\n{critic_summary}\n\n"
            f"Disagreement indicator: {disagreement}\n\n"
            f"Artifact mode: {artifact_mode}\n\n"
        )
        if format_checks:
            user += f"Format checks (JSON):\n{format_checks}\n\n"
        user += "Return the final decision."

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

