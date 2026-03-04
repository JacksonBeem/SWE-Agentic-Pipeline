from __future__ import annotations
import time
import requests
from dataclasses import dataclass
from typing import Any

from .config import OpenRouterConfig, ModelConfig

@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any]
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

class OpenRouterClient:
    def __init__(self, cfg: OpenRouterConfig, timeout_s: float = 120.0):
        self.cfg = cfg
        self.timeout_s = timeout_s

    def chat(self, model_cfg: ModelConfig, messages: list[dict[str, str]]) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }
        # Optional attribution headers (safe to omit)
        if self.cfg.http_referer:
            headers["HTTP-Referer"] = self.cfg.http_referer
        if self.cfg.x_title:
            headers["X-Title"] = self.cfg.x_title

        payload = {
            "model": model_cfg.model,
            "messages": messages,
            "temperature": model_cfg.temperature,
            "max_tokens": model_cfg.max_tokens,
        }

        t0 = time.time()
        r = requests.post(
            f"{self.cfg.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_s,
        )
        latency = time.time() - t0
        r.raise_for_status()
        data = r.json()

        # OpenRouter is compatible with OpenAI-style responses
        text = data["choices"][0]["message"]["content"]

        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            raw=data,
            latency_s=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
