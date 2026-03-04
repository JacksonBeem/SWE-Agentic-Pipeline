from __future__ import annotations
from .base import AgentBase

ARCHITECT_PROMPT = """You are the Architect in a fixed software development pipeline (Architect → Developer → Security → QA → Verifier).

Your responsibility is to define intent, structure, and constraints WITHOUT producing executable artifacts.

Hard rules:
- Do NOT write code, pseudocode, tests, or patches.
- Do NOT use code formatting or code syntax such as: def, class, import, ``` or "diff --git".
- When describing interfaces, use plain English signatures like:
  - function_name(arg1, arg2) -> return_type
  NOT Python syntax like: def function_name(...):

Output a structured specification:
1) Functional requirements (numbered)
2) Constraints/assumptions
3) Interfaces (signatures / I/O)
4) Acceptance criteria (testable conditions)
"""

class ArchitectAgent(AgentBase):
    name = "Architect"

    def build_messages(self, problem: str, repo_context: str | None = None) -> list[dict[str, str]]:
        ctx = repo_context.strip() if repo_context else ""
        user = f"Problem:\n{problem}\n\nRepo context (if any):\n{ctx}"
        return [
            {"role": "system", "content": ARCHITECT_PROMPT},
            {"role": "user", "content": user},
        ]
