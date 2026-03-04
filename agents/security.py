from __future__ import annotations
from .base import AgentBase

SECURITY_PROMPT = """You are the Security reviewer in a fixed software development pipeline (Architect → Developer → Security → QA → Verifier).
Assess the code artifact for vulnerabilities and unsafe patterns. Do NOT modify code. Output a structured report:
- Findings (each with CWE if applicable)
- Severity (LOW/MED/HIGH/CRITICAL)
- Evidence (short)
If none: say 'no issues identified'."""

class SecurityAgent(AgentBase):
    name = "Security"

    def build_messages(self, code_artifact: str) -> list[dict[str,str]]:
        return [
            {"role": "system", "content": SECURITY_PROMPT},
            {"role": "user", "content": f"Code artifact:\n{code_artifact}"},
        ]
