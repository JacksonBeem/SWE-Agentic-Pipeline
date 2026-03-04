# pipeline/utils/runs_csv_to_code.py
from __future__ import annotations

import ast
import csv
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ExtractedCode:
    code: Optional[str]
    mode: str                 # "diff" or "raw"
    error: Optional[str] = None
    row: Optional[Dict[str, Any]] = None  # optional: the row we extracted from


# ----------------------------
# Diff detection + parsing
# ----------------------------

_DIFF_HEADER_RE = re.compile(r"^diff --git ", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@", re.MULTILINE)


def _looks_like_unified_diff(text: str) -> bool:
    if not text:
        return False
    return bool(_DIFF_HEADER_RE.search(text)) or ("--- " in text and "+++ " in text and _HUNK_RE.search(text))


def _strip_markdown_fences(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    out: List[str] = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        out.append(line)
    return "\n".join(out).strip()


def _trim_to_python_start(text: str) -> str:
    """
    Drop role/prose wrappers and keep from the first likely Python line.
    """
    lines = text.splitlines()
    if not lines:
        return text
    starters = ("def ", "class ", "import ", "from ", "@")
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith(starters):
            return "\n".join(lines[i:]).strip()
    return text.strip()


def _truncate_at_noncode_section(text: str) -> str:
    """
    Keep the leading code block and drop common analysis/trailer sections
    emitted by instruction-following models (QA:, FINAL_CODE, XML role tags, etc.).
    """
    markers = (
        "QA:",
        "ARCHITECT:",
        "DEVELOPER:",
        "VERIFIER:",
        "FINAL_CODE",
        "FINAL CODE",
        "# FINAL_CODE",
        "<FINAL_CODE>",
        "</FINAL_CODE>",
        "QA PHASE",
        "# QA",
        "## QA",
        "**MENTAL EXECUTION",
        "MENTAL EXECUTION:",
        "</DEVELOPER>",
        "</ARCHITECT>",
        "<QA>",
        "</QA>",
        "</VERIFIER>",
    )
    out: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        su = s.upper()
        # Common markdown/delimiter trailers that are not Python code.
        if s in {"---", "___", "***"}:
            break
        if s and set(s) == {"-"} and len(s) >= 3:
            break
        if any(su.startswith(m) for m in markers):
            break
        out.append(line)
    return "\n".join(out).strip()


def extract_code_from_unified_diff(diff_text: str) -> ExtractedCode:
    """
    Extract a plausible final-file python source from a unified diff:
      - drop metadata lines and hunk headers
      - ignore removed lines (-)
      - keep context lines (space prefix) and added lines (+) without the prefix
    Works well when your diff contains the whole file/function (common in your demo output).
    """
    if not diff_text or not _looks_like_unified_diff(diff_text):
        return ExtractedCode(code=None, mode="diff", error="Input does not look like a unified diff.")

    kept: List[str] = []

    for line in diff_text.splitlines():
        if line.startswith(("diff --git", "index ", "--- ", "+++ ", "new file mode", "deleted file mode")):
            continue
        if line.startswith("@@"):
            continue

        if line.startswith("-"):
            continue
        if line.startswith("+"):
            kept.append(line[1:])
            continue
        if line.startswith(" "):
            kept.append(line[1:])
            continue

        # Rare: bare lines in diff context
        kept.append(line)

    code = "\n".join(kept).strip()
    if not code:
        return ExtractedCode(code=None, mode="diff", error="Diff parsed but produced empty code.")

    if "def " not in code:
        return ExtractedCode(code=code, mode="diff", error="Extracted code has no `def` (may be incomplete).")

    return ExtractedCode(code=code, mode="diff", error=None)


def extract_code_from_artifact_text(artifact_text: str) -> ExtractedCode:
    """
    Main extraction: handles either unified diff or raw python code.
    """
    if not artifact_text or not artifact_text.strip():
        return ExtractedCode(code=None, mode="raw", error="Empty artifact text.")

    text = artifact_text.strip()

    text = _strip_markdown_fences(text)

    if _looks_like_unified_diff(text):
        return extract_code_from_unified_diff(text)

    text = _trim_to_python_start(text)
    text = _truncate_at_noncode_section(text)

    # treat as raw python
    if "def " not in text and "class " not in text:
        return ExtractedCode(code=text, mode="raw", error="Artifact is not a diff, but has no `def` (may be incomplete).")
    return ExtractedCode(code=text, mode="raw", error=None)


def compose_prompt_executable_code(prompt: str, artifact_text: str) -> ExtractedCode:
    """
    Build executable Python code from prompt + artifact.
    Strategy:
      1) Extract candidate code from raw/diff artifact.
      2) If candidate is already a full, parseable function/module (contains `def`), use it.
      3) Otherwise treat candidate as body/tail fragment and append under the given prompt.
    """
    extracted = extract_code_from_artifact_text(artifact_text or "")
    candidate = (extracted.code or artifact_text or "").strip()
    candidate = _strip_markdown_fences(candidate)

    if candidate:
        try:
            ast.parse(candidate)
            if "def " in candidate:
                return ExtractedCode(code=candidate, mode=extracted.mode, error=extracted.error)
        except Exception:
            pass

    frag_lines = candidate.splitlines()
    cut_idx = 0
    for i, ln in enumerate(frag_lines):
        s = ln.strip()
        if s.startswith(">>>") or s.startswith("...") or '"""' in s:
            cut_idx = i + 1

    body_lines = frag_lines[cut_idx:] if cut_idx < len(frag_lines) else frag_lines

    normalized_body: List[str] = []
    for ln in body_lines:
        if not ln.strip():
            normalized_body.append("")
            continue
        if ln.startswith((" ", "\t")):
            normalized_body.append(ln)
        else:
            normalized_body.append("    " + ln)

    full_code = (prompt.rstrip() + "\n" + "\n".join(normalized_body).rstrip() + "\n").strip() + "\n"

    try:
        ast.parse(full_code)
        return ExtractedCode(code=full_code, mode=extracted.mode, error=extracted.error)
    except Exception as e:
        return ExtractedCode(
            code=full_code,
            mode=extracted.mode,
            error=f"Composed prompt+artifact code not parseable: {type(e).__name__}: {e}",
        )


def compose_humaneval_executable_code(prompt: str, artifact_text: str) -> ExtractedCode:
    # Backwards-compatible alias.
    return compose_prompt_executable_code(prompt, artifact_text)


# ----------------------------
# runs.csv reading helpers
# ----------------------------

def _read_runs_csv(csv_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"runs.csv not found at: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path} is empty.")
    return rows


def _pick_row(rows: List[Dict[str, Any]], task_id: Optional[str], run_id: Optional[str]) -> Dict[str, Any]:
    """
    Selection rules:
      - if run_id provided: pick first matching row
      - else if task_id provided: pick last matching row for that task_id
      - else: pick last row (most recent)
    """
    if run_id:
        for r in rows:
            if str(r.get("run_id", "")).strip() == run_id:
                return r
        raise ValueError(f"No row found with run_id={run_id}")

    if task_id:
        matches = [r for r in rows if str(r.get("task_id", "")).strip() == task_id]
        if not matches:
            raise ValueError(f"No row found with task_id={task_id}")
        return matches[-1]

    return rows[-1]


def extract_code_from_runs_csv(
    csv_path: str = "runs.csv",
    artifact_column: str = "final_artifact",
    task_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> ExtractedCode:
    """
    Reads runs.csv, selects a row, extracts executable code from the artifact column.

    Parameters:
      csv_path: path to runs.csv
      artifact_column: which column contains the artifact text (default: "final_artifact")
      task_id: optional selector
      run_id: optional selector

    Returns:
      ExtractedCode(code=..., mode=..., error=..., row=selected_row)
    """
    rows = _read_runs_csv(csv_path)
    row = _pick_row(rows, task_id=task_id, run_id=run_id)

    if artifact_column not in row:
        return ExtractedCode(
            code=None,
            mode="raw",
            error=f"Column '{artifact_column}' not found in runs.csv. Available columns: {list(row.keys())}",
            row=row,
        )

    artifact_text = row.get(artifact_column) or ""
    extracted = extract_code_from_artifact_text(str(artifact_text))
    extracted.row = row
    return extracted
