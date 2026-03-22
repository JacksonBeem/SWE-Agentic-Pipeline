from __future__ import annotations

from .base import AgentBase


CRITIC_PROMPT = """You are the Critic evaluator in a fixed software development pipeline (Planner -> Executor -> Critic -> Verifier).
You must evaluate functional correctness via the provided test harness.
You must NOT propose fixes. Output:
- PASS or FAIL (single line)
- Short failure localization / summary (if FAIL)
- Concise run summary
"""


class CriticAgent(AgentBase):
    name = "Critic"

    def build_messages(self, code_artifact: str, test_harness: str | None = None) -> list[dict[str, str]]:
        harness = test_harness.strip() if test_harness else "(no harness provided in this demo)"
        return [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": f"Test harness:\n{harness}\n\nCode artifact:\n{code_artifact}"},
        ]

