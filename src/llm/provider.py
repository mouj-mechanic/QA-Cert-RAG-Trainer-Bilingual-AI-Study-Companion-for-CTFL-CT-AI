"""
LLM provider abstraction — swap OpenAI later without touching RAG logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Minimal generation interface."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        """Generate a completion from system + user prompts."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the provider can be used."""


class MissingAPIKeyError(RuntimeError):
    """Raised when no API key is configured."""


def get_llm_provider() -> LLMProvider:
    """Factory: currently always OpenAI provider."""
    from src.llm.openai_provider import OpenAIProvider

    return OpenAIProvider()
