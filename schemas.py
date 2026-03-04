from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass
class TaskInput:
    task_id: str
    problem: str
    repo_context: str | None = None
    test_harness: str | None = None

@dataclass
class ArchitectSpec:
    text: str

@dataclass
class CodeArtifact:
    text: str

@dataclass
class SecurityFindings:
    text: str

@dataclass
class QAResult:
    passed: bool
    summary: str

@dataclass
class VerifierDecision:
    decision: str
    repair_request: str | None

# --- Leakage barriers (simple v1 heuristics; tighten later) ---

PROHIBITED = {
    # Architect must never emit code or patch syntax.
    "Architect": ["diff --git", "def "],

    # Developer shouldn't claim QA/security outcomes.
    "Developer": ["REJECT", "unit test", "qa result", "security finding"],

    # Security must not propose or apply fixes.
    "Security":  ["apply patch", "here is the fix", "I changed"],

    # Verifier is allowed to mention patch format (diff --git) but still must not emit code.
    "Verifier":  ["import ", "def ", "class "],
}

def sanitize_output(agent: str, text: str) -> str:
    # Strip markdown code fences for non-verifier agents that should not emit fenced blocks.
    if agent in {"Architect", "Developer", "QA", "Security"}:
        lines = text.splitlines()
        out = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            out.append(line)
        return "\n".join(out).strip()
    return text

def assert_no_prohibited(agent: str, text: str) -> None:
    for pat in PROHIBITED.get(agent, []):
        if pat.lower() in (text or "").lower():
            raise ValueError(f"Leakage/prohibition hit for {agent}: contains '{pat}'")
