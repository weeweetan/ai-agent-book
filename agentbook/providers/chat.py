"""Small Chat Completions adapter for the OpenAI-compatible chapter examples.

This does not replace the SDK or change endpoints. Sync, async and streaming
calls retain the SDK's return types. DeepSeek defaults to non-thinking mode in
these examples because many teaching loops store only content and tool calls.
Callers that preserve reasoning_content can explicitly enable thinking.
"""

from __future__ import annotations

import re
from typing import Any


def chat_request(**kwargs: Any) -> dict[str, Any]:
    """Copy a request and apply model-specific parameter compatibility."""
    model = str(kwargs.get("model", "")).lower().split("/")[-1]
    # SDK response objects contain fields that are not valid assistant inputs.
    # Several teaching loops append message.model_dump() directly to history.
    # Copy rather than mutate history, and retain DeepSeek's reasoning state.
    if "messages" in kwargs:
        messages = []
        for message in kwargs["messages"]:
            if hasattr(message, "model_dump"):
                message = message.model_dump(exclude_none=True)
            if isinstance(message, dict) and message.get("role") == "assistant":
                message = {
                    key: value
                    for key, value in message.items()
                    if key not in {"annotations", "parsed"}
                    and (value is not None or key == "content")
                }
                if model.startswith(("gpt-", "chatgpt-")) or re.match(r"^o[1-9](?:-|$)", model):
                    message.pop("reasoning_content", None)
                    message.pop("reasoning", None)
            messages.append(message)
        kwargs["messages"] = messages
    if re.match(r"^(gpt-[5-9]|o[1-9](?:-|$))", model):
        if "max_tokens" in kwargs:
            limit = kwargs.pop("max_tokens")
            kwargs.setdefault("max_completion_tokens", limit)
        # Omit sampling controls instead of sending unsupported fixed values.
        for field in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            kwargs.pop(field, None)
    if model.startswith("deepseek"):
        body = dict(kwargs.get("extra_body") or {})
        # Legacy reasoner models have no non-thinking mode. Their callers must
        # retain reasoning_content, as with an explicitly enabled thinking mode.
        if model != "deepseek-reasoner":
            body.setdefault("thinking", {"type": "disabled"})
        if body:
            kwargs["extra_body"] = body
    return kwargs


def chat_completion(client: Any, **kwargs: Any) -> Any:
    """Call a sync or async SDK client with a compatible request."""
    return client.chat.completions.create(**chat_request(**kwargs))
