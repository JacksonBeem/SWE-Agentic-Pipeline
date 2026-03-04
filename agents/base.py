from __future__ import annotations

from dataclasses import dataclass

from ..openrouter_client import OpenRouterClient, LLMResponse
from ..config import ModelConfig
from ..schemas import assert_no_prohibited, sanitize_output


@dataclass
class AgentResult:
    output_text: str
    llm: LLMResponse
    raw_output: str
    messages: list[dict[str, str]]
    error: str | None


class AgentBase:
    name: str = "Agent"

    def __init__(self, client: OpenRouterClient, model_cfg: ModelConfig):
        self.client = client
        self.model_cfg = model_cfg

    def build_messages(self, **kwargs) -> list[dict[str, str]]:
        raise NotImplementedError

    def run(self, **kwargs) -> AgentResult:
        messages = self.build_messages(**kwargs)

        try:
            resp = self.client.chat(self.model_cfg, messages)
            raw = resp.text

            clean = sanitize_output(self.name, raw)
            assert_no_prohibited(self.name, clean)

            return AgentResult(
                output_text=clean,
                llm=resp,
                raw_output=raw,
                messages=messages,
                error=None,
            )

        except Exception as e:
            # raise with context; orchestrator will still log the prompt it built
            raise RuntimeError(f"{self.name} failed after LLM call: {e}") from e
