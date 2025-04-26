"""Provider-agnostic abstraction layer (sync)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, ClassVar

from .retry_utils import rate_limit_retry

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from anthropic import Anthropic
    from openai import OpenAI
    from vertexai import VertexAI

    from .data_class import GenerateParameters, Messages
# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AbstractLLMClient(ABC):
    """Uniform synchronous *generate* interface for every provider."""

    #: Registry key (e.g. "openai", "anthropic")
    provider: ClassVar[str]

    # ---- lifecycle --------------------------------------------------------

    def __init__(self, model: str) -> None:
        self.model = model
        self.client = self._create_client()  # provider SDK instance

    @abstractmethod
    def _create_client(self) -> OpenAI | Anthropic | VertexAI:
        """Instantiate the underlying SDK client."""

    @abstractmethod
    def generate(self, messages: Messages, params: GenerateParameters) -> str:
        """Provider-specific call - may raise ProviderError family."""

    # ---- public API --------------------------------------------------------

    @rate_limit_retry(max_attempts=3, wait_seconds=1)
    def call_with_retry(
        self,
        messages: Messages,
        params: GenerateParameters,
    ) -> str:
        """Generate text with automatic rate-limit retry."""
        return self.generate(messages=messages, params=params)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------


_REGISTRY: MutableMapping[str, type[AbstractLLMClient]] = {}


def register_provider(cls: type[AbstractLLMClient]) -> type[AbstractLLMClient]:
    key = getattr(cls, 'provider', None)
    if not key:
        raise ValueError("Provider client must define 'provider' class attr")
    if key in _REGISTRY:
        raise ValueError(f"Provider '{key}' already registered by {_REGISTRY[key].__name__}")
    _REGISTRY[key] = cls
    return cls


def resolve_client(model_uri: str) -> AbstractLLMClient:
    """Return a concrete client instance inferred from *model_uri*.

    Examples
    --------
    >>> client = resolve_client("openai:gpt-4o")
    >>> text = client.generate_with_retry([
    ...     Message(role="user", content="Hello")
    ... ])

    """
    if ':' not in model_uri:
        raise ValueError("model_uri must be in 'provider:model' format (e.g. 'openai:gpt-4o')")

    provider_key, model_name = model_uri.split(':', 1)

    try:
        cls = _REGISTRY[provider_key]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"Unknown provider '{provider_key}'. Registered: {list(_REGISTRY)}") from exc

    return cls(model=model_name)
