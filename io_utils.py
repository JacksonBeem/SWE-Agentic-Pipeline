from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_fences(s: str) -> str:
    if not s:
        return s
    lines = s.splitlines()
    out = []
    in_fence = False
    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        out.append(line)
    return "\n".join(out).strip()


def append_prediction(path: Path, task_id: str, completion: str, model_name: str | None = None) -> None:
    rec = {
        "task_id": task_id,
        "completion": strip_fences(completion),
    }
    if model_name:
        rec["model"] = model_name

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
