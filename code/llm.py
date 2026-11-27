"""Thin, backend-agnostic client for open-weight LLMs with structured output.

* ollama : POST /api/chat with `format=<JSON schema>` → grammar-constrained
           decoding, output is guaranteed schema-valid JSON.
* openai : any OpenAI-compatible /chat/completions endpoint (Groq, HF router,
           OpenRouter, vLLM, llama.cpp ...). Uses `json_schema` response_format
           where supported, falls back to `json_object`.
Either way the result is validated with pydantic before being accepted.
"""
from __future__ import annotations

import json
import logging

import requests
from pydantic import BaseModel, ValidationError

log = logging.getLogger("llm")


class EarlyStop:
    """Generation deliberately stopped early; `fields` holds what was parsed."""
    def __init__(self, fields: dict, raw: str):
        self.fields, self.raw = fields, raw


class LLMClient:
    def __init__(self, cfg: dict):
        self.backend = cfg["backend"]
        self.model = cfg["model"]
        self.base_url = cfg["base_url"].rstrip("/")
        self.api_key = cfg.get("api_key")
        self.temperature = cfg.get("temperature", 0)
        self.num_ctx = cfg.get("num_ctx", 8192)
        self.num_predict = cfg.get("num_predict", 700)
        self.max_retries = cfg.get("max_retries", 2)

    # -- backends ---------------------------------------------------------
    def _ollama(self, messages: list[dict], schema: dict, early_stop=None):
        """Streamed call. If `early_stop(partial_text)` returns a dict, the
        connection is closed (Ollama then stops generating) and that dict is
        returned as an EarlyStop instead of the full JSON text."""
        with requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "format": schema,
                "stream": True,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                            "num_predict": self.num_predict, "seed": 42},
            },
            stream=True,
            timeout=600,
        ) as r:
            r.raise_for_status()
            parts: list[str] = []
            for line in r.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                parts.append(chunk.get("message", {}).get("content", ""))
                if early_stop is not None:
                    hit = early_stop("".join(parts))
                    if hit is not None:
                        return EarlyStop(hit, "".join(parts))
                if chunk.get("done"):
                    break
        return "".join(parts)

    def _openai(self, messages: list[dict], schema: dict, early_stop=None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.num_predict,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "extraction", "schema": schema}},
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=300)
        if r.status_code == 400:  # provider without json_schema support
            body["response_format"] = {"type": "json_object"}
            r = requests.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # -- public -------------------------------------------------------------
    def runtime_version(self) -> str | None:
        """Recorded with every extraction, for provenance/reproducibility."""
        if self.backend != "ollama":
            return self.base_url
        try:
            return "ollama " + requests.get(f"{self.base_url}/api/version", timeout=10).json()["version"]
        except Exception:
            return None

    def structured(self, system: str, user: str, model_cls: type[BaseModel], early_stop=None):
        """Validated `model_cls` instance, or an EarlyStop (see _ollama)."""
        schema = model_cls.model_json_schema()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        call = self._ollama if self.backend == "ollama" else self._openai
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            raw = call(messages, schema, early_stop)
            if isinstance(raw, EarlyStop):
                return raw
            try:
                return model_cls.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                log.warning("invalid output (attempt %d): %s", attempt + 1, str(e)[:300])
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That output was invalid: {e}. Return corrected JSON only."},
                ]
        raise RuntimeError(f"LLM failed to return valid JSON: {last_err}")
