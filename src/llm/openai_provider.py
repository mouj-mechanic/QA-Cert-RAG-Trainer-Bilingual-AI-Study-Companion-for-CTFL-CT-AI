"""OpenAI chat completion provider."""

from __future__ import annotations

import logging

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.llm.provider import LLMProvider, MissingAPIKeyError

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Initial LLM backend for Chat_ISTQB."""

    def __init__(self, api_key: str | None = None, model: str = OPENAI_MODEL) -> None:
        self.model = model
        self._api_key = (api_key if api_key is not None else OPENAI_API_KEY).strip()
        self._client: OpenAI | None = None
        if self.is_available():
            self._client = OpenAI(api_key=self._api_key)

    def is_available(self) -> bool:
        key = self._api_key
        return bool(key) and key != "sk-your-key-here"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if not self.is_available() or self._client is None:
            raise MissingAPIKeyError(
                "OPENAI_API_KEY is missing. Copy .env.example to .env and set your key."
            )

        logger.debug("Calling OpenAI model=%s", self.model)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()
