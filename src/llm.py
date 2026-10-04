"""Minimal LLM client (same shape as the sibling projects)."""
import requests

import config


class LLMClient:
    def __init__(self, provider: str = config.LLM_PROVIDER):
        self.provider = provider
        if provider == "mistral":
            self.url = "https://api.mistral.ai/v1/chat/completions"
            self.key = config.MISTRAL_API_KEY
            self.model = config.MISTRAL_MODEL
        elif provider == "openai":
            self.url = "https://api.openai.com/v1/chat/completions"
            self.key = config.OPENAI_API_KEY
            self.model = config.OPENAI_MODEL
        elif provider == "none":
            self.url = self.key = self.model = ""
        else:
            raise ValueError(f"unknown LLM provider: {provider}")

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and bool(self.key)

    def complete(self, system: str, user: str,
                 temperature: float = 0.0) -> tuple[str, int, int]:
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": temperature},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return (data["choices"][0]["message"]["content"],
                usage.get("prompt_tokens", 0),
                usage.get("completion_tokens", 0))

    def estimate_cost(self, pt: int, ct: int) -> float:
        prices = config.TOKEN_PRICES.get(self.model, {"input": 0, "output": 0})
        return pt / 1000 * prices["input"] + ct / 1000 * prices["output"]
