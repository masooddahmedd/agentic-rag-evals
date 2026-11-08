"""Chat calls to any OpenAI-compatible endpoint, with timing and token counts for the trace."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import settings


@dataclass
class Call:
    text: str
    ms: int
    prompt_tokens: int
    completion_tokens: int


class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or settings.llm_model
        self.client = OpenAI(base_url=base_url or settings.llm_base_url, api_key=api_key or settings.llm_api_key,
                             timeout=300)

    def chat(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 500,
             temperature: float = 0.1) -> Call:
        t = time.perf_counter()
        qwen3 = "qwen3" in self.model.lower()
        if qwen3:
            system += " /no_think"          # Qwen3 would otherwise think out loud before the JSON
        # Ollama's JSON mode makes Qwen3 return an empty object; it follows "reply with JSON" fine without it.
        kwargs = {"response_format": {"type": "json_object"}} if json_mode and not qwen3 else {}
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)
        u = r.usage
        text = re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        return Call(text, round((time.perf_counter() - t) * 1000),
                    getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)


def parse_json(text: str) -> dict:
    """Small models sometimes wrap JSON in prose or code fences; take the first object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
