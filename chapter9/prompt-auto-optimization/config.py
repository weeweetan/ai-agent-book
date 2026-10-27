"""
统一的 LLM 客户端配置。

默认使用 OpenAI（读取 OPENAI_API_KEY，模型 gpt-5.6-luna）。
也支持通过环境变量 LLM_PROVIDER 切换到 Moonshot / 火山方舟(ARK)，
它们都兼容 OpenAI 的 Chat Completions + 工具调用接口。

    export LLM_PROVIDER=openai   # 默认
    export LLM_PROVIDER=moonshot # 用 MOONSHOT_API_KEY
    export LLM_PROVIDER=ark      # 用 ARK_API_KEY，并需设置 ARK_MODEL

统一的 OpenRouter 兜底（fallback）：
    若所选 provider 自己的 Key 缺失，但设置了 OPENROUTER_API_KEY，则自动改走
    OpenRouter（https://openrouter.ai/api/v1），并把模型名映射到 OpenRouter 命名：
        gpt-*     -> openai/gpt-*
        claude-*  -> anthropic/claude-opus-4.8
        含 "/"    -> 原样透传
        其它      -> openai/gpt-5.6-luna
"""

from agentbook.providers import chat_completion, chat_request

from agentbook.providers import PROVIDERS, example_model, example_provider, resolve_example_backend

import os
import time
from typing import Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# 各提供商的默认配置：base_url / 环境变量名 / 默认模型
_PROVIDERS = {
    "openai": {
        "base_url": None,  # 使用 SDK 默认
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-5.6-luna",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "MOONSHOT_API_KEY",
        "default_model": "kimi-k3",
    },
    "ark": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "key_env": "ARK_API_KEY",
        # ARK 需要用推理接入点(endpoint id) 作为 model，请通过 ARK_MODEL 指定
        "default_model": os.getenv("ARK_MODEL", "doubao-seed-1-6-250615"),
    },
    "openrouter": {
        "base_url": OPENROUTER_BASE_URL,
        "key_env": "OPENROUTER_API_KEY",
        "default_model": "openai/gpt-4o-mini",
    },
}

API_TURNS = []


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json", exclude_none=True))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "openai").lower().strip()


def _to_openrouter_model(model: str) -> str:
    """把常见模型名映射到 OpenRouter 命名空间。"""
    if not model:
        return "openai/gpt-5.6-luna"
    if "/" in model:
        return model
    if model.startswith("gpt-"):
        return "openai/" + model
    if model.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    return "openai/gpt-5.6-luna"


def _is_reasoning_model(model: str) -> bool:
    """gpt-5.x / o1·o3·o4 / kimi-k3 / *reasoner 等推理模型：不接受 temperature=0，
    直连 gpt-5.x 还需组织实名且工具调用受限，故优先走 OpenRouter。"""
    m = (model or "").lower()
    return (m.startswith(("gpt-5", "o1", "o3", "o4"))
            or m.startswith("kimi-k3")
            or "reasoner" in m or "thinking" in m)


def _use_openrouter(cfg: dict) -> bool:
    """走 OpenRouter 的两种情形：
    1) provider 自己的 Key 缺失、但有 OPENROUTER_API_KEY（统一兜底）；
    2) 目标是 gpt-5.x 且有 OPENROUTER_API_KEY —— 直连 gpt-5.x 需组织实名、
       且 /chat/completions 工具调用受限，故即便有 OPENAI_API_KEY 也优先 OpenRouter。"""
    if not os.getenv("OPENROUTER_API_KEY"):
        return False
    if not os.getenv(cfg["key_env"]):
        return True
    model = os.getenv("LLM_MODEL") or cfg["default_model"]
    return (model or "").lower().startswith("gpt-5")


def get_model() -> str:
    return get_backend_metadata()["model"]


def get_client() -> OpenAI:
    backend = resolve_example_backend(default="openai")
    return OpenAI(api_key=backend.api_key, base_url=backend.base_url)


def record_completion(client: OpenAI, *, kind: str, **request: Any):
    """Execute and retain a credential-free raw request/response receipt."""
    started = time.time()
    request = chat_request(**request)
    response = chat_completion(client, **request)
    API_TURNS.append({
        "kind": kind,
        "provider": get_provider(),
        "endpoint": get_backend_metadata()["endpoint"],
        "request": _jsonable(request),
        "response": response.model_dump(mode="json", exclude_none=True),
        "elapsed_seconds": round(time.time() - started, 6),
    })
    return response


def reset_api_turns() -> None:
    API_TURNS.clear()


def get_api_turns() -> list[dict]:
    return list(API_TURNS)


def get_backend_metadata() -> dict[str, Any]:
    provider = example_provider(default="openai")
    spec = PROVIDERS[provider]
    model, base_url, routed = example_model(provider), spec.resolved_base_url(), provider
    if spec.api_key() or os.getenv("OPENROUTER_API_KEY"):
        backend = resolve_example_backend(provider)
        model, base_url = backend.model, backend.base_url
        routed = "openrouter" if backend.using_openrouter else backend.provider
    keys = PROVIDERS[routed].key_vars
    source = next((key for key in keys if os.getenv(key)), keys[0])
    return {"configured_provider": provider, "routed_provider": routed,
            "model": model, "endpoint": base_url.rstrip("/") + "/chat/completions",
            "credential_source_env": source, "credential_value_recorded": False}


def usage_summary() -> dict[str, Any]:
    prompt = completion = total = 0
    native_cost = 0.0
    native_cost_count = 0
    for turn in API_TURNS:
        usage = turn.get("response", {}).get("usage") or {}
        prompt += int(usage.get("prompt_tokens") or 0)
        completion += int(usage.get("completion_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
        if usage.get("cost") is not None:
            native_cost += float(usage["cost"])
            native_cost_count += 1
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
        "provider_reported_cost_usd": round(native_cost, 9) if native_cost_count else None,
        "provider_reported_cost_observations": native_cost_count,
        "cost_qualification": (
            "provider-native usage.cost summed across calls"
            if native_cost_count else "provider did not expose monetary cost; no price was guessed"
        ),
    }


# 全部 LLM 调用统一使用低温度，保证结果可复现；
# 但推理模型（gpt-5.x / o 系列 / kimi-k3 等）只接受默认 temperature=1，
# 故按当前解析出的模型自动选择默认温度（可用 LLM_TEMPERATURE 显式覆盖）。
def _default_temperature() -> str:
    provider = get_provider()
    cfg = _PROVIDERS.get(provider, _PROVIDERS["openai"])
    model = os.getenv("LLM_MODEL") or cfg["default_model"]
    return "1" if _is_reasoning_model(model) else "0"


def get_temperature() -> float:
    """在调用时按当前解析出的模型选择温度，使 CLI/env 的 --model/--provider
    覆盖生效。原来的模块级 TEMPERATURE 常量在 import 时就被固定，而 demo.py 在
    import 之后才设置 LLM_MODEL/LLM_PROVIDER，导致温度停留在默认模型的值
    （例如把非推理模型误用 temperature=1，破坏了本文件追求的可复现性）。"""
    return float(os.getenv("LLM_TEMPERATURE", _default_temperature()))
