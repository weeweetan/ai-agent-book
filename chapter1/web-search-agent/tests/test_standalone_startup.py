"""Exercise the documented entry points in fresh Python processes.

The chapter CI installs only this example's requirements.txt, so these checks
also cover readers who have not installed the shared agentbook package.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1]


def run_python(*args, cwd, no_site=False):
    env = os.environ.copy()
    # Ignore inherited repository paths and do not read developer credentials.
    env.pop("PYTHONPATH", None)
    env["PYTHON_DOTENV_DISABLED"] = "1"
    options = ["-E", "-X", "utf8"]
    if no_site:
        options.append("-S")
    return subprocess.run(
        [sys.executable, *options, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )


@pytest.mark.parametrize(
    "module", ["main", "agent", "examples", "quickstart", "run_experiment_1_2"]
)
def test_example_modules_import_in_a_fresh_process(module):
    result = run_python("-c", f"import {module}", cwd=EXAMPLE_DIR)

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("outside_example", [False, True])
@pytest.mark.parametrize("no_site", [False, True])
def test_offline_cli_runs_from_checkout_without_pythonpath(
    tmp_path, outside_example, no_site
):
    output = tmp_path / "demo.json"
    cwd = tmp_path if outside_example else EXAMPLE_DIR
    result = run_python(
        str(EXAMPLE_DIR / "main.py"),
        "--provider",
        "offline-demo",
        "--output",
        str(output),
        cwd=cwd,
        no_site=no_site,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == {"question", "trace", "answer"}
    assert payload["question"] == "Moonshot AI 的 Context Caching 是什么技术？"
    assert payload["trace"][-1]["type"] == "answer"
    assert payload["answer"] == payload["trace"][-1]["content"]
    assert "离线演示模式" in result.stdout


def test_help_works_without_third_party_dependencies():
    result = run_python("main.py", "--help", cwd=EXAMPLE_DIR, no_site=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "offline-demo" in result.stdout


def test_missing_online_dependencies_show_install_command():
    result = run_python(
        "main.py", "--provider", "openai", cwd=EXAMPLE_DIR, no_site=True
    )

    assert result.returncode == 2
    assert "pip install -r" in result.stderr
    assert "requirements.txt" in result.stderr
    assert "Traceback" not in result.stderr


def test_offline_mode_ignores_invalid_online_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    monkeypatch.setenv("SEARCH_TIMEOUT", "not-a-number")
    result = run_python(
        "main.py",
        "--provider",
        "offline-demo",
        "--output",
        str(tmp_path / "demo.json"),
        cwd=EXAMPLE_DIR,
        no_site=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
