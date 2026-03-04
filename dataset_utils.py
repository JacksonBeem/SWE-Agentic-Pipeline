from __future__ import annotations

import re
from typing import Any

_MBPP_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.MULTILINE)
_MBPP_ASSERT_CALL_RE = re.compile(r"\bassert\s+([A-Za-z_]\w*)\s*\(")


def _mbpp_signature_from_code(row: dict[str, Any]) -> str:
    code = str(row.get("code", "") or "")
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("def "):
            return s
    return ""


def mbpp_entry_point(row: dict[str, Any]) -> str:
    code = str(row.get("code", "") or "")
    m = _MBPP_DEF_RE.search(code)
    if m:
        return m.group(1)

    for test in row.get("test_list", []) or []:
        m = _MBPP_ASSERT_CALL_RE.search(str(test or ""))
        if m:
            return m.group(1)
    return ""


def detect_dataset_type(row: dict[str, Any]) -> str:
    if "test_list" in row and ("prompt" in row or "text" in row):
        return "mbpp"
    if "complete_prompt" in row and "code_prompt" in row:
        return "bigcodebench"
    return "humaneval"


def task_prompt_for_dataset(row: dict[str, Any], dataset_type: str) -> str:
    if dataset_type == "bigcodebench":
        return str(row.get("complete_prompt", "") or row.get("code_prompt", "") or "")
    if dataset_type == "mbpp":
        prompt = str(row.get("prompt", "") or row.get("text", "") or "")
        signature = _mbpp_signature_from_code(row)
        entry_point = mbpp_entry_point(row)
        if signature:
            prompt += f"\n\nRequired function signature:\n{signature}"
        elif entry_point:
            prompt += f"\n\nRequired function name: {entry_point}"
        return prompt
    return str(row.get("prompt", "") or "")


def task_entry_point_for_dataset(row: dict[str, Any], dataset_type: str) -> str:
    if dataset_type == "mbpp":
        return mbpp_entry_point(row)
    return str(row.get("entry_point", "") or "")


def task_id_for_row(row: dict[str, Any], dataset_type: str) -> str:
    raw = row.get("task_id", row.get("ask_id", ""))
    tid = str(raw).strip()
    if not tid:
        return ""
    if "/" in tid:
        return tid
    if dataset_type == "mbpp":
        return f"MBPP/{tid}"
    return tid


def mbpp_test_harness(row: dict[str, Any]) -> str:
    lines: list[str] = []
    for imp in row.get("test_imports", []) or []:
        s = str(imp).strip()
        if s:
            lines.append(s)
    setup = str(row.get("test_setup_code", "") or "").strip()
    if setup:
        lines.append(setup)
    for test in row.get("test_list", []) or []:
        s = str(test).strip()
        if s:
            lines.append(s)
    return "\n".join(lines).strip()
