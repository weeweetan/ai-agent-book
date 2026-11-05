"""Run provider-enabled chapter modules as a reader would, from their directory."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Pick the configuration boundary where a project has one. Retrieval engines,
# external servers and media generation are tested by their own chapter suites.
EXAMPLES = [
    ("chapter1/context", "config", ()),
    ("chapter1/learning-from-experience", "llm_agent", ()),
    ("chapter1/web-search-agent", "main", ()),
    ("chapter2/agent-skills-ppt", "demo", ("pptx",)),
    ("chapter2/context-compression", "config", ()),
    ("chapter2/kv-cache", "agent", ()),
    ("chapter2/prompt-injection", "agent", ()),
    ("chapter2/system-hint", "config", ()),
    ("chapter3/agentic-rag", "config", ()),
    ("chapter3/agentic-rag-for-user-memory", "config", ()),
    ("chapter3/contextual-retrieval", "config", ()),
    ("chapter3/contextual-retrieval-for-user-memory", "config", ()),
    ("chapter3/mem0", "config", ()),
    ("chapter3/memobase", "config", ()),
    ("chapter3/user-memory", "config", ()),
    ("chapter4/execution-tools", "config", ()),
    ("chapter5/agent-creator", "creator", ()),
    ("chapter5/code-for-logic", "demo", ("constraint",)),
    ("chapter5/code-for-math", "demo", ()),
    ("chapter5/conversational-ui", "agent", ()),
    ("chapter5/dynamic-form", "demo", ("bs4",)),
    ("chapter5/erp-agent", "agent", ()),
    ("chapter5/paper-to-ppt", "agents", ("PIL",)),
    ("chapter5/paper-to-video", "demo", ("PIL",)),
    ("chapter6/agent-with-event-trigger", "agent", ("mcp",)),
    ("chapter6/async-agent", "runtime", ()),
    ("chapter9/prompt-auto-optimization", "config", ()),
    ("chapter9/self-evolving-tools", "agent", ()),
    ("chapter10/autonomous-phone-registration", "decision", ()),
    ("chapter10/parallel-web-research", "llm", ()),
    ("chapter10/voice-werewolf", "werewolf.agent", ()),
]

STARTUP = """
import importlib, importlib.util, json, os, socket, sys
from pathlib import Path
if os.environ.get('EXPECT_UNINSTALLED_AGENTBOOK') == '1':
    assert importlib.util.find_spec('agentbook') is None, 'An installed package would hide checkout import failures'
def deny_network(*args, **kwargs):
    raise AssertionError('Startup checks must not access the network')
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
importlib.import_module(sys.argv[1])
import agentbook
from agentbook.providers import resolve_example_backend
assert Path(agentbook.__file__).resolve() == Path(sys.argv[2]).resolve()
backend = resolve_example_backend()
print('STARTUP:' + json.dumps({
    'provider': backend.provider, 'model': backend.model,
    'base_url': backend.base_url, 'key': backend.api_key,
}))
"""


@pytest.mark.parametrize("provider", ["kimi", "deepseek", "openai"])
@pytest.mark.parametrize(
    "directory,module,optional", EXAMPLES, ids=[x[0] for x in EXAMPLES]
)
def test_example_starts_from_its_own_directory(provider, directory, module, optional):
    for dependency in optional:
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(
                f"This example requires {dependency}; checkout-startup CI installs it"
            )
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.endswith(("_API_KEY", "_BASE_URL", "_MODEL"))
        and name
        not in {"PYTHONPATH", "LLM_PROVIDER", "MODEL", "MODEL_NAME", "PROVIDER"}
    }
    key_name = (
        "MOONSHOT_API_KEY" if provider == "kimi" else provider.upper() + "_API_KEY"
    )
    env.update(
        {
            "LLM_PROVIDER": provider,
            key_name: "startup-fixture-key",
            "SERPER_API_KEY": "startup-search-key",
            "PYTHON_DOTENV_DISABLED": "1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            "-X",
            "utf8",
            "-c",
            STARTUP,
            module,
            str(ROOT / "agentbook/__init__.py"),
        ],
        cwd=ROOT / directory,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = next(
        line.removeprefix("STARTUP:")
        for line in result.stdout.splitlines()
        if line.startswith("STARTUP:")
    )
    backend = json.loads(payload)
    assert backend["provider"] == provider
    assert backend["key"] == "startup-fixture-key"
    assert backend["base_url"].startswith(
        {
            "kimi": "https://api.moonshot.cn/v1",
            "deepseek": "https://api.deepseek.com",
            "openai": "https://api.openai.com/v1",
        }[provider]
    )
