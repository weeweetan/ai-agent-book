"""Provider selection must not discard an explicit --model override."""

import os

import pytest

import demo


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in list(os.environ):
        if key.endswith(("_API_KEY", "_MODEL")) or key in {"MODEL", "MODEL_NAME", "LLM_PROVIDER"}:
            monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("provider", ["openai", "deepseek"])
def test_explicit_model_is_used_with_matching_provider(monkeypatch, provider):
    monkeypatch.setenv(provider.upper() + "_API_KEY", "fixture-key")
    client, model = demo.build_client_and_model(provider, model_override="custom-model")
    assert model == "custom-model"
    assert provider in str(client.base_url)
    client.close()


def test_auto_obeys_shared_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture-key")
    client, model = demo.build_client_and_model()
    assert model == "deepseek-v4-pro"
    assert "api.deepseek.com" in str(client.base_url)
    client.close()
