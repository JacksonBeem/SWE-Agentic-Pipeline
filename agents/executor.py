from __future__ import annotations

from .base import AgentBase


SYSTEM_PROMPT = """You are the Executor in a fixed software development pipeline (Planner -> Executor -> Critic -> Verifier).
Implement exactly the Planner spec.

Hard output rules:
- If repo context exists, output ONLY a unified diff patch (git-style).
- Do NOT include markdown fences (no ```).
- Do NOT output standalone code blocks.
- The very first line must start with: diff --git

If repo context is empty, output full code (no markdown fences).
"""


class ExecutorAgent(AgentBase):
    name = "Executor"

    def build_messages(self, problem: str, planner_spec: str, repo_context: str | None = None):
        repo_present = bool(repo_context and repo_context.strip())
        sys = SYSTEM_PROMPT

        user = (
            f"Problem:\n{problem}\n\n"
            f"Planner spec:\n{planner_spec}\n\n"
            f"Repo context (optional):\n{repo_context or ''}\n\n"
        )

        if repo_present:
            user += (
                "Return ONLY a unified diff patch now. "
                "No markdown fences. Start with 'diff --git'."
            )
        else:
            user += "Return ONLY the code artifact now. No markdown fences."

        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ]

