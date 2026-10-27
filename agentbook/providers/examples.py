"""Environment configuration shared by the runnable chapter examples.

Explicit arguments win over LLM_PROVIDER / LLM_MODEL. Credentials always belong
to the selected provider; a DeepSeek key is never sent to Moonshot or OpenAI.
"""

from __future__ import annotations

import os
from dataclasses import replace

from .models import Backend
from .registry import canonical_provider, lookup
from .resolution import resolve_backend


def example_provider(provider: str | None = None, *, default: str = "kimi") -> str:
    """Select a provider without requiring credentials (also usable by CLIs)."""
    selected = (provider or "").strip()
    if not selected or selected == "auto":
        selected = os.getenv("LLM_PROVIDER", "").strip() or default
    return canonical_provider(selected)


def example_model(
    provider: str, model: str | None = None, *, default_model: str | None = None
) -> str:
    """Resolve explicit model, shared override, provider override, then default."""
    spec = lookup(provider)
    legacy_model_var = {"kimi": "MOONSHOT_MODEL", "doubao": "ARK_MODEL"}.get(spec.name, "")
    return (
        (model or "").strip()
        or os.getenv("LLM_MODEL", "").strip()
        or os.getenv("MODEL_NAME", "").strip()
        or os.getenv(f"{spec.name.upper()}_MODEL", "").strip()
        or os.getenv(legacy_model_var, "").strip()
        or default_model
        or spec.default_model
    )


def resolve_example_backend(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    default: str = "kimi",
    default_model: str | None = None,
) -> Backend:
    """Resolve an example's provider, model and matching credentials together."""
    selected = example_provider(provider, default=default)
    backend = resolve_backend(
        selected, example_model(selected, model, default_model=default_model), api_key
    )
    # A custom endpoint is for the selected provider, never for a fallback's key.
    if base_url and not backend.using_openrouter:
        backend = replace(backend, base_url=base_url.strip())
    return backend
