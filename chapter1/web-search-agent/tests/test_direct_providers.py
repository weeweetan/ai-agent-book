"""Official provider searches work with a single key and real SDK wire formats."""

import copy
import json
from unittest.mock import Mock

import httpx
from openai import OpenAI
import pytest
import requests

from agent import WebSearchAgent, _render_citations, is_failure_answer
import main

MODEL_KEY = "selected-provider-key"
SOURCE_URL = "https://example.com/source"


def mock_sdk(monkeypatch, handler):
    """Exercise SDK serialization and parsing, without any network or real key."""
    calls = []

    def transport(request):
        calls.append(request)
        return handler(request)

    def constructor(**kwargs):
        return OpenAI(
            **kwargs,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(transport)),
        )

    monkeypatch.setattr("agent.OpenAI", constructor)
    return calls


def chat_reply(*, tool_call=False):
    message = {"role": "assistant", "content": "Answer based on search"}
    if tool_call:
        message.update(
            content=None,
            tool_calls=[
                {
                    "id": "call-search",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query":"example"}',
                    },
                }
            ],
        )
    return {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_call else "stop",
                "message": message,
            }
        ],
    }


def responses_reply():
    text = "Answer based on search.citeturn0search0"
    return {
        "id": "resp-1",
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o",
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "tools": [{"type": "web_search"}],
        "output": [
            {
                "type": "web_search_call",
                "id": "ws-1",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "example",
                    "sources": [{"type": "url", "url": SOURCE_URL}],
                },
            },
            {
                "id": "msg-1",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "start_index": text.index(""),
                                "end_index": len(text),
                                "url": SOURCE_URL,
                                "title": "Source",
                            }
                        ],
                    }
                ],
            },
        ],
    }


@pytest.mark.parametrize(
    "base_url", ["https://api.deepseek.com", "https://api.deepseek.com/v1/"]
)
def test_deepseek_search_round_trip_without_serper(monkeypatch, base_url):
    monkeypatch.setenv("DEEPSEEK_API_KEY", MODEL_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    replies = iter([chat_reply(tool_call=True), chat_reply()])
    calls = mock_sdk(
        monkeypatch, lambda request: httpx.Response(200, json=next(replies))
    )
    payload = {
        "results": [
            {
                "title": "Source",
                "url": SOURCE_URL,
                "summary": "Evidence",
                "content": "Page content",
            }
        ]
    }
    response = Mock(status_code=200)
    response.json.return_value = payload
    post = Mock(return_value=response)
    monkeypatch.setattr("agent.requests.post", post)
    agent = WebSearchAgent(provider="deepseek", base_url=base_url)

    assert agent.model == "deepseek-flash"
    assert agent.search_and_answer("example") == "Answer based on search"
    post.assert_called_once_with(
        "https://api.deepseek.com/v1/tools/search",
        headers={
            "Authorization": f"Bearer {MODEL_KEY}",
            "Content-Type": "application/json",
        },
        json={"query": "example", "count": 5, "need_content": True},
        timeout=agent._request_timeout,
    )
    assert len(calls) == 2
    assert all(
        request.headers["Authorization"] == f"Bearer {MODEL_KEY}" for request in calls
    )
    followup = json.loads(calls[1].content)
    assert json.loads(followup["messages"][-1]["content"]) == payload
    assert followup["messages"][-1]["tool_call_id"] == "call-search"
    assert followup["tools"][0]["function"]["name"] == "web_search"
    assert [step["type"] for step in agent.get_trace()] == [
        "action",
        "observation",
        "answer",
    ]
    assert agent.api_turns[1]["kind"] == "deepseek_search"
    assert MODEL_KEY not in json.dumps(agent.get_api_turns())
    agent.client.close()


def test_deepseek_cli_explicit_key_and_custom_endpoint(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "unused-env-key")
    monkeypatch.setenv("SERPER_API_KEY", "unused-serper-key")
    calls = mock_sdk(
        monkeypatch, lambda request: httpx.Response(200, json=chat_reply())
    )
    post = Mock()
    post.return_value.status_code = 200
    post.return_value.json.return_value = {"results": []}
    monkeypatch.setattr("agent.requests.post", post)
    agent = WebSearchAgent(
        provider="deepseek", api_key=MODEL_KEY, base_url="https://proxy.example/api/v1/"
    )
    assert json.loads(agent._execute_formula("web_search", '{"query":"example"}')) == {
        "results": []
    }
    assert post.call_args.args == ("https://proxy.example/api/v1/tools/search",)
    assert post.call_args.kwargs["headers"]["Authorization"] == f"Bearer {MODEL_KEY}"
    assert calls == []
    agent.client.close()


@pytest.mark.parametrize("status", [401, 402, 404, 429, 500])
def test_deepseek_search_failure_is_recorded_and_not_reported_as_answer(
    monkeypatch, status
):
    calls = mock_sdk(
        monkeypatch,
        lambda request: httpx.Response(200, json=chat_reply(tool_call=True)),
    )
    response = Mock(status_code=status)
    response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    monkeypatch.setattr("agent.requests.post", Mock(return_value=response))
    agent = WebSearchAgent(provider="deepseek", api_key=MODEL_KEY)

    answer = agent.search_and_answer("example")

    assert is_failure_answer(answer)
    assert f"HTTP {status}" in answer
    assert len(calls) == 1
    assert all(step["type"] != "answer" for step in agent.trace)
    receipt = agent.api_turns[-1]
    assert receipt["kind"] == "deepseek_search"
    assert receipt["http_status"] == status
    assert receipt["error"]["class"] == "HTTPError"
    assert MODEL_KEY not in json.dumps(agent.get_api_turns())
    agent.client.close()


@pytest.mark.parametrize("query", ["", "   ", "x" * 129, None, 42])
def test_deepseek_rejects_invalid_search_arguments_before_request(monkeypatch, query):
    mock_sdk(monkeypatch, lambda request: pytest.fail("No model request expected"))
    post = Mock()
    monkeypatch.setattr("agent.requests.post", post)
    agent = WebSearchAgent(provider="deepseek", api_key=MODEL_KEY)
    with pytest.raises(ValueError, match="1–128"):
        agent._execute_formula("web_search", json.dumps({"query": query}))
    post.assert_not_called()
    agent.client.close()


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5", "gpt-6-astra"])
def test_openai_hosted_search_with_citations_and_no_serper(monkeypatch, model):
    monkeypatch.setenv("OPENAI_API_KEY", MODEL_KEY)
    calls = mock_sdk(
        monkeypatch, lambda request: httpx.Response(200, json=responses_reply())
    )
    agent = WebSearchAgent(provider="openai", model=model)

    answer = agent.search_and_answer("example", max_iterations=3)

    assert answer == f"Answer based on search.[Source]({SOURCE_URL})"
    assert len(calls) == 1
    request = calls[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["Authorization"] == f"Bearer {MODEL_KEY}"
    body = json.loads(request.content)
    assert body["tools"] == [{"type": "web_search"}]
    assert body["tool_choice"] == "required"
    assert body["max_tool_calls"] == 3
    assert body["max_output_tokens"] <= 16384
    assert body["include"] == ["web_search_call.action.sources"]
    assert body["store"] is False
    assert body["model"] == model
    assert body["input"] == "example"
    assert "max_tokens" not in body and "max_completion_tokens" not in body
    assert ("temperature" in body) == (model == "gpt-4o")
    assert [step["type"] for step in agent.trace] == ["action", "observation", "answer"]
    assert agent.trace[0]["args"]["query"] == "example"
    assert SOURCE_URL in agent.trace[1]["content"]
    assert agent.conversation_history[-1] == {"role": "assistant", "content": answer}
    assert agent.api_turns[0]["kind"] == "responses"
    assert MODEL_KEY not in json.dumps(agent.get_api_turns())
    agent.client.close()


@pytest.mark.parametrize("search_status", [None, "failed", "in_progress"])
def test_openai_does_not_claim_success_without_completed_search(
    monkeypatch, search_status
):
    payload = responses_reply()
    if search_status is None:
        payload["output"].pop(0)
    else:
        payload["output"][0]["status"] = search_status
    mock_sdk(monkeypatch, lambda request: httpx.Response(200, json=payload))
    agent = WebSearchAgent(provider="openai", api_key=MODEL_KEY)
    assert is_failure_answer(agent.search_and_answer("example"))
    assert all(step["type"] != "answer" for step in agent.trace)
    agent.client.close()


@pytest.mark.parametrize("status", [401, 404, 429, 500])
def test_openai_http_error_is_recorded(monkeypatch, status):
    mock_sdk(
        monkeypatch,
        lambda request: httpx.Response(
            status, json={"error": {"message": "Unavailable"}}
        ),
    )
    agent = WebSearchAgent(provider="openai", api_key=MODEL_KEY)
    assert is_failure_answer(agent.search_and_answer("example"))
    assert "error" in agent.api_turns[0]
    assert MODEL_KEY not in json.dumps(agent.get_api_turns())
    agent.client.close()


def test_openai_token_limit_preserves_partial_answer_and_marks_truncation(monkeypatch):
    payload = responses_reply()
    payload.update(
        status="incomplete", incomplete_details={"reason": "max_output_tokens"}
    )
    mock_sdk(monkeypatch, lambda request: httpx.Response(200, json=payload))
    agent = WebSearchAgent(provider="openai", api_key=MODEL_KEY)
    answer = agent.search_and_answer("example")
    assert "Answer based on search" in answer and "截断" in answer
    assert agent.trace[-1]["content"] == answer
    assert agent.conversation_history[-1]["content"] == answer
    agent.client.close()


def test_openai_questions_reset_search_evidence_and_zero_budget_makes_no_call(
    monkeypatch,
):
    calls = mock_sdk(
        monkeypatch, lambda request: httpx.Response(200, json=responses_reply())
    )
    agent = WebSearchAgent(provider="openai", api_key=MODEL_KEY)
    agent.search_and_answer("first")
    agent.search_and_answer("second")
    assert len(calls) == 2
    assert len(agent.api_turns) == 1
    assert len(agent.trace) == 3
    assert agent.conversation_history[1]["content"] == "second"
    assert is_failure_answer(agent.search_and_answer("third", max_iterations=0))
    assert len(calls) == 2
    assert agent.api_turns == [] and agent.trace == []
    agent.client.close()


def test_citation_ranges_preserve_prose_and_handle_multiple_links():
    part = {
        "text": "First. Second.",
        "annotations": [
            {
                "type": "url_citation",
                "start_index": 0,
                "end_index": 6,
                "url": SOURCE_URL,
                "title": "A",
            },
            {
                "type": "url_citation",
                "start_index": 7,
                "end_index": 14,
                "url": SOURCE_URL,
                "title": "B",
            },
        ],
    }
    original = copy.deepcopy(part)
    assert (
        _render_citations(part) == f"First. [A]({SOURCE_URL}) Second. [B]({SOURCE_URL})"
    )
    assert part == original


def test_cli_passes_provider_without_injecting_kimi_defaults(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr("agent.WebSearchAgent", constructor)
    monkeypatch.setattr(main, "run_single_question", Mock())
    main.main(["query", "--provider", "openai"])
    assert constructor.call_args.kwargs["provider"] == "openai"
    assert constructor.call_args.kwargs["model"] is None
    assert constructor.call_args.kwargs["base_url"] is None
    assert constructor.call_args.kwargs["api_key"] is None


def test_user_deepseek_cli_initializes_without_serper(monkeypatch):
    mock_sdk(
        monkeypatch, lambda request: pytest.fail("Initialization needs no network")
    )
    interactive = Mock()
    monkeypatch.setattr(main, "run_interactive_mode", interactive)
    main.main(
        [
            "--provider",
            "deepseek",
            "--base-url",
            "https://api.deepseek.com",
            "--api-key",
            MODEL_KEY,
        ]
    )
    instance = interactive.call_args.args[0]
    assert instance.provider == "deepseek"
    assert instance.model == "deepseek-flash"
    assert instance.base_url == "https://api.deepseek.com"
    instance.client.close()
