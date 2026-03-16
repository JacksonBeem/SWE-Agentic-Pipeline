from __future__ import annotations
import random
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
    def __init__(
        self,
        cfg: OpenRouterConfig,
        timeout_s: float = 120.0,
        max_retries: int = 6,
        backoff_base_s: float = 2.0,
        backoff_cap_s: float = 60.0,
    ):
        self.cfg = cfg
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s

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

        url = f"{self.cfg.base_url}/chat/completions"
        last_response: requests.Response | None = None
        last_exc: Exception | None = None
        attempt = 0

        while attempt <= self.max_retries:
            attempt += 1
            t0 = time.time()
            try:
                r = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                latency = time.time() - t0
                last_response = r
                if r.ok:
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

                retryable = r.status_code in {429, 500, 502, 503, 504}
                if (not retryable) or (attempt > self.max_retries):
                    r.raise_for_status()

                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = self.backoff_base_s
                else:
                    sleep_s = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
                sleep_s = sleep_s + random.uniform(0.0, 0.5)
                time.sleep(sleep_s)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt > self.max_retries:
                    break
                sleep_s = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
                sleep_s = sleep_s + random.uniform(0.0, 0.5)
                time.sleep(sleep_s)

        if last_response is not None:
            body = (last_response.text or "").strip()
            if len(body) > 300:
                body = body[:300] + "..."
            raise RuntimeError(
                f"OpenRouter request failed after retries: HTTP {last_response.status_code}; body={body}"
            )
        if last_exc is not None:
            raise RuntimeError(f"OpenRouter request failed after retries: {last_exc}") from last_exc
        raise RuntimeError("OpenRouter request failed after retries: unknown error")
