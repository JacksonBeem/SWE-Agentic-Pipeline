from __future__ import annotations

from .base import AgentBase

SYSTEM_PROMPT = """You are the Developer in a fixed software development pipeline (Architect → Developer → Security → QA → Verifier).
Implement exactly the Architect spec.

Hard output rules:
- If repo context exists, output ONLY a unified diff patch (git-style).
- Do NOT include markdown fences (no ```).
- Do NOT output standalone code blocks.
- The very first line must start with: diff --git

If repo context is empty, output full code (no markdown fences).
"""

class DeveloperAgent(AgentBase):
    name = "Developer"

    def build_messages(self, problem: str, architect_spec: str, repo_context: str | None = None):
        repo_present = bool(repo_context and repo_context.strip())
        sys = SYSTEM_PROMPT

        user = (
            f"Problem:\n{problem}\n\n"
            f"Architect spec:\n{architect_spec}\n\n"
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
