"""Offline routing and real SDK wire-format tests for the chapter adapters."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
from openai import AsyncOpenAI, OpenAI
import pytest

from agentbook.providers import (
    PROVIDERS,
    chat_completion,
    chat_request,
    example_model,
    resolve_example_backend,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.endswith(("_API_KEY", "_BASE_URL", "_MODEL")) or name in {
            "LLM_PROVIDER",
            "MODEL_NAME",
            "MODEL",
            "PROVIDER",
        }:
            monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "provider,model,host,key_var",
    [
        ("kimi", "kimi-k3", "api.moonshot.cn", "MOONSHOT_API_KEY"),
        ("deepseek", "deepseek-flash", "api.deepseek.com", "DEEPSEEK_API_KEY"),
        ("openai", "gpt-4o", "api.openai.com", "OPENAI_API_KEY"),
    ],
)
def test_environment_selects_matching_model_endpoint_and_credential(
    monkeypatch, provider, model, host, key_var
):
    for var in ("MOONSHOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, var + "-secret")
    monkeypatch.setenv("LLM_PROVIDER", provider)
    backend = resolve_example_backend()
    assert backend.provider == provider
    assert backend.model == model
    assert httpx.URL(backend.base_url).host == host
    assert backend.api_key == key_var + "-secret"
    assert not backend.using_openrouter


def test_explicit_provider_and_model_win_and_custom_urls_are_honored(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "kimi")
    monkeypatch.setenv("LLM_MODEL", "shared-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://proxy.test/v1")
    backend = resolve_example_backend("deepseek", "explicit-model", "explicit-key")
    assert (backend.provider, backend.model, backend.api_key, backend.base_url) == (
        "deepseek",
        "explicit-model",
        "explicit-key",
        "https://proxy.test/v1",
    )


def test_explicit_openai_does_not_reroute_when_openrouter_key_exists(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    backend = resolve_example_backend("openai", "gpt-5")
    assert backend.api_key == "openai-secret"
    assert backend.base_url == "https://api.openai.com/v1"


def test_provider_model_and_blank_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", " ")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
    assert example_model("deepseek") == "deepseek-v4-pro"
    assert example_model("deepseek", "chosen") == "chosen"


def test_missing_selected_key_never_reuses_other_vendor_key(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "kimi-secret")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        resolve_example_backend("deepseek")


def test_fallback_credential_is_not_forwarded_to_primary_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    backend = resolve_example_backend("kimi", base_url="https://primary.test/v1")
    assert backend.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o3", "o4-mini", "openai/gpt-5"])
def test_reasoning_parameters_do_not_send_unsupported_sampling_or_token_fields(model):
    result = chat_request(model=model, max_tokens=4096, temperature=1, top_p=0.9)
    assert result == {"model": model, "max_completion_tokens": 4096}


def test_deepseek_default_and_explicit_thinking_preserve_original_request():
    body = {"thinking": {"type": "enabled"}}
    messages = [{"role": "assistant", "content": "", "reasoning_content": "context"}]
    result = chat_request(model="deepseek-flash", extra_body=body, messages=messages)
    assert result["extra_body"] == body
    assert result["messages"] == messages
    assert chat_request(model="deepseek-flash")["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "extra_body" not in chat_request(model="kimi-k3")


def test_response_only_fields_are_not_sent_back_as_assistant_inputs():
    message = {
        "role": "assistant",
        "content": None,
        "annotations": [],
        "parsed": None,
        "reasoning_content": "reasoning",
        "refusal": None,
    }
    openai_message = chat_request(model="gpt-4o", messages=[message])["messages"][0]
    assert openai_message == {"role": "assistant", "content": None}
    deepseek_message = chat_request(model="deepseek-flash", messages=[message])["messages"][0]
    assert deepseek_message["reasoning_content"] == "reasoning"
    assert "annotations" not in deepseek_message
    assert message["annotations"] == []  # original transcript remains intact


@pytest.mark.parametrize(
    "provider,model", [("openai", "gpt-5"), ("deepseek", "deepseek-flash"), ("kimi", "kimi-k3")]
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sdk_tool_round_trip_and_streaming(provider, model, asynchronous):
    """Check actual serialized SDK HTTP requests, including the second tool turn."""
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        assert request.headers["authorization"] == "Bearer fixture-key"
        assert request.url.host == httpx.URL(PROVIDERS[provider].base_url).host
        if provider == "openai":
            assert "temperature" not in payload and "max_tokens" not in payload
            assert payload["max_completion_tokens"] == 4096
        if provider == "deepseek":
            assert payload["thinking"] == {"type": "disabled"}
        if payload.get("stream"):
            chunk = {
                "id": "stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": "done"}, "finish_reason": None}],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            )
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        }
        return httpx.Response(
            200,
            json={
                "id": "test-1",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
            },
        )

    request = dict(
        model=model,
        max_tokens=4096,
        temperature=0.3,
        messages=[{"role": "user", "content": "look it up"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    )

    def append_tool(response):
        request["messages"].append(response.choices[0].message.model_dump())
        request["messages"].append({"role": "tool", "tool_call_id": "call-1", "content": "found"})

    async def run_async():
        async with AsyncOpenAI(
            api_key="fixture-key",
            base_url=PROVIDERS[provider].base_url,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        ) as client:
            append_tool(await chat_completion(client, **request))
            stream = await chat_completion(client, **request, stream=True)
            async with stream:
                return "".join([part.choices[0].delta.content or "" async for part in stream])

    if asynchronous:
        assert asyncio.run(run_async()) == "done"
    else:
        with OpenAI(
            api_key="fixture-key",
            base_url=PROVIDERS[provider].base_url,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        ) as client:
            append_tool(chat_completion(client, **request))
            with chat_completion(client, **request, stream=True) as stream:
                assert "".join(part.choices[0].delta.content or "" for part in stream) == "done"
    assert len(seen) == 2
    assert seen[1]["messages"][-1]["tool_call_id"] == "call-1"


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize(
    "directory,expression",
    [
        ("chapter1/learning-from-experience", "__import__('llm_agent').LLMAgent()"),
        ("chapter2/system-hint", "__import__('agent').SystemHintAgent()"),
        ("chapter2/kv-cache", "__import__('agent').KVCacheAgent(api_key=None)"),
    ],
)
def test_real_agent_constructors_follow_environment(monkeypatch, provider, directory, expression):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(provider.upper() + "_API_KEY", "fixture-key")
    code = f"agent = {expression}; print(agent.model); print(agent.client.base_url)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / directory,
        text=True,
        capture_output=True,
        check=True,
    )
    assert PROVIDERS[provider].default_model in result.stdout
    assert PROVIDERS[provider].base_url in result.stdout


@pytest.mark.parametrize("provider", ["deepseek", "openai"])
@pytest.mark.parametrize(
    "directory,expression",
    [
        ("chapter2/context-compression", "Config.resolve_llm()"),
        (
            "chapter3/agentic-rag",
            "LLMConfig(provider=os.environ['LLM_PROVIDER']).get_client_config()",
        ),
        (
            "chapter3/contextual-retrieval",
            "LLMConfig(provider=os.environ['LLM_PROVIDER']).get_client_config()",
        ),
        (
            "chapter3/agentic-rag-for-user-memory",
            "LLMConfig(provider=os.environ['LLM_PROVIDER']).get_client_config()",
        ),
        (
            "chapter3/contextual-retrieval-for-user-memory",
            "LLMConfig(provider=os.environ['LLM_PROVIDER']).get_client_config()",
        ),
        ("chapter3/mem0", "vars(KimiConfig())"),
        ("chapter4/execution-tools", "Config.get_llm_config()"),
    ],
)
def test_chapter_configuration_uses_selected_endpoint(monkeypatch, provider, directory, expression):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv(provider.upper() + "_API_KEY", "fixture-key")
    monkeypatch.setenv(provider.upper() + "_BASE_URL", "https://proxy.test/v1")
    code = "import os; from config import *; print(" + expression + ")"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / directory,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "https://proxy.test/v1" in result.stdout
    assert PROVIDERS[provider].default_model in result.stdout


def test_interactive_cli_does_not_send_fallback_key_to_primary_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-fixture-key")
    code = """
import builtins, sys
import main
original = main.SystemHintAgent
def capture(*args, **kwargs):
    agent = original(*args, **kwargs)
    assert str(agent.client.base_url) == "https://openrouter.ai/api/v1/"
    assert agent.client.api_key == "router-fixture-key"
    return agent
main.SystemHintAgent = capture
builtins.input = lambda *args: "quit"
sys.argv = ["main.py", "--provider", "deepseek", "--mode", "interactive"]
main.main()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / "chapter2/system-hint",
        text=True,
        capture_output=True,
        check=True,
    )
