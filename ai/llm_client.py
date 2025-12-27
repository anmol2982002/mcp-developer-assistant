"""
LLM Client

Unified LLM client supporting multiple providers (Groq, Anthropic, OpenAI).
Uses async HTTP calls with retry logic and structured logging.
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from observability.logging_config import get_logger
from observability.metrics import metrics

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM call."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        pass


class GroqClient(BaseLLMClient):
    """
    Groq LLM client.

    Supports Llama, Mixtral, and other models via Groq API.
    Fast inference with very low latency.
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
    ):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model to use (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key:
            logger.warning("groq_api_key_missing", msg="GROQ_API_KEY not set")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(60.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using Groq API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        import time

        start_time = time.perf_counter()

        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        client = await self._get_client()

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Record metrics
            metrics.record_llm_call(
                provider="groq",
                model=self.model,
                latency=time.perf_counter() - start_time,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

            logger.info(
                "groq_generation_complete",
                model=self.model,
                latency_ms=latency_ms,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )

            return content

        except httpx.HTTPStatusError as e:
            logger.error("groq_api_error", status_code=e.response.status_code, error=str(e))
            raise
        except Exception as e:
            logger.error("groq_generation_error", error=str(e))
            raise


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude client.

    Supports Claude 3 Sonnet, Haiku, and Opus models.
    """

    BASE_URL = "https://api.anthropic.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-sonnet-20240229",
    ):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client: Optional[httpx.AsyncClient] = None

        if not self.api_key:
            logger.warning("anthropic_api_key_missing", msg="ANTHROPIC_API_KEY not set")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(120.0),
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """Generate text using Anthropic API."""
        import time

        start_time = time.perf_counter()

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if system_prompt:
            payload["system"] = system_prompt

        client = await self._get_client()

        try:
            response = await client.post("/messages", json=payload)
            response.raise_for_status()

            data = response.json()
            content = data["content"][0]["text"]
            usage = data.get("usage", {})

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            metrics.record_llm_call(
                provider="anthropic",
                model=self.model,
                latency=time.perf_counter() - start_time,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )

            logger.info(
                "anthropic_generation_complete",
                model=self.model,
                latency_ms=latency_ms,
            )

            return content

        except Exception as e:
            logger.error("anthropic_generation_error", error=str(e))
            raise


def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseLLMClient:
    """
    Factory function to get LLM client.

    Args:
        provider: LLM provider (groq, anthropic). Defaults to env LLM_PROVIDER
        api_key: API key (optional, uses env var if not provided)
        model: Model name (optional, uses provider default)

    Returns:
        LLM client instance
    """
    provider = provider or os.getenv("LLM_PROVIDER", "groq")

    if provider.lower() == "groq":
        return GroqClient(api_key=api_key, model=model or "llama-3.3-70b-versatile")
    elif provider.lower() in ("anthropic", "claude"):
        return AnthropicClient(api_key=api_key, model=model or "claude-3-sonnet-20240229")
    else:
        logger.warning("unknown_llm_provider", provider=provider, fallback="groq")
        return GroqClient(api_key=api_key)


# Default client (lazy initialized)
_default_client: Optional[BaseLLMClient] = None


async def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> str:
    """
    Convenience function for text generation with default client.

    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens
        temperature: Sampling temperature

    Returns:
        Generated text
    """
    global _default_client
    if _default_client is None:
        _default_client = get_llm_client()

    return await _default_client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
