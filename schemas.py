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
class PlannerSpec:
    text: str

@dataclass
class CodeArtifact:
    text: str

@dataclass
class CriticResult:
    passed: bool
    summary: str

@dataclass
class VerifierDecision:
    decision: str
    repair_request: str | None

# --- Leakage barriers (simple v1 heuristics; tighten later) ---

PROHIBITED = {
    # Planner must never emit code or patch syntax.
    "Planner": ["diff --git", "def "],

    # Executor shouldn't claim Critic outcomes.
    "Executor": ["REJECT", "unit test", "Critic result"],

    # Verifier is allowed to mention patch format (diff --git) but still must not emit code.
    "Verifier":  ["import ", "def ", "class "],
}

def sanitize_output(agent: str, text: str) -> str:
    # Strip markdown code fences for non-verifier agents that should not emit fenced blocks.
    if agent in {"Planner", "Executor", "Critic"}:
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

