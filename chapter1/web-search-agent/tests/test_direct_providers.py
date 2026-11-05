"""Alternate LLM providers execute real search through an independent tool."""

import json
from unittest.mock import Mock

import pytest

from agent import WebSearchAgent
import main


@pytest.mark.parametrize(
    "provider,model,url",
    [
        ("deepseek", "deepseek-flash", "https://api.deepseek.com"),
        ("openai", "gpt-4o", "https://api.openai.com/v1"),
    ],
)
def test_direct_search_round_trip(
    monkeypatch, make_choice, make_tool_call, provider, model, url
):
    monkeypatch.setenv(provider.upper() + "_API_KEY", "model-key")
    monkeypatch.setenv("SERPER_API_KEY", "search-key")
    client = Mock()
    client.chat.completions.create.side_effect = [
        Mock(
            choices=[
                make_choice(finish_reason="tool_calls", tool_calls=[make_tool_call()])
            ]
        ),
        Mock(choices=[make_choice(content="Answer based on search")]),
    ]
    constructor = Mock(return_value=client)
    monkeypatch.setattr("agent.OpenAI", constructor)
    response = Mock()
    response.json.return_value = {
        "organic": [
            {"title": "Source", "link": "https://example.com", "snippet": "Evidence"}
        ]
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("agent.requests.post", post)
    agent = WebSearchAgent(provider=provider)
    assert agent.model == model
    assert constructor.call_args.kwargs["base_url"] == url
    assert agent.search_and_answer("example") == "Answer based on search"
    assert post.call_count == 1
    assert post.call_args.args == ("https://google.serper.dev/search",)
    assert post.call_args.kwargs["headers"]["X-API-KEY"] == "search-key"
    assert "model-key" not in json.dumps(post.call_args.kwargs)
    assert any(turn["kind"] == "serper_search" for turn in agent.api_turns)
    assert client.chat.completions.create.call_count == 2
    second = client.chat.completions.create.call_args.kwargs
    assert second["tools"][0]["function"]["name"] == "web_search"


def test_no_search_key_fails_before_constructing_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "model-key")
    constructor = Mock()
    monkeypatch.setattr("agent.OpenAI", constructor)
    with pytest.raises(ValueError, match="SERPER_API_KEY"):
        WebSearchAgent(provider="deepseek")
    constructor.assert_not_called()


def test_cli_passes_provider_without_injecting_kimi_defaults(monkeypatch):
    constructor = Mock()
    monkeypatch.setattr("agent.WebSearchAgent", constructor)
    monkeypatch.setattr(main, "run_single_question", Mock())
    main.main(["query", "--provider", "openai"])
    assert constructor.call_args.kwargs["provider"] == "openai"
    assert constructor.call_args.kwargs["model"] is None
    assert constructor.call_args.kwargs["base_url"] is None
    assert constructor.call_args.kwargs["api_key"] is None
