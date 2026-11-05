"""
Configuration for Memobase Agent with Kimi K3 Model
"""

# Locate the shared package when this example is run from a source checkout.
import sys as _sys
from pathlib import Path as _CheckoutPath

_sys.path.insert(0, str(_CheckoutPath(__file__).resolve().parents[2]))

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _openrouter_model_id(model) -> str:
    """Map a provider-native model name to an OpenRouter model id, used by the
    universal OpenRouter fallback. An explicit OPENROUTER_MODEL env var wins."""
    override = os.getenv("OPENROUTER_MODEL")
    if override:
        return override
    m = (model or "").strip()
    if not m:
        return "openai/gpt-5.6-luna"
    if "/" in m:
        return m
    ml = m.lower()
    if ml.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai/" + m
    if ml.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    if ml.startswith("kimi"):
        # kimi-k3 is not on OpenRouter; moonshotai/kimi-k2.6 is the closest hosted id.
        return "moonshotai/kimi-k2.6"
    return "openai/gpt-5.6-luna"


# Legacy exported names retained for existing agent imports.
from agentbook.providers import PROVIDERS, example_model, example_provider, resolve_example_backend

LLM_PROVIDER = example_provider()
_spec = PROVIDERS[LLM_PROVIDER]
KIMI_API_KEY = _spec.api_key()
KIMI_BASE_URL = _spec.resolved_base_url()
KIMI_MODEL = example_model(LLM_PROVIDER)
if KIMI_API_KEY or os.getenv("OPENROUTER_API_KEY"):
    _backend = resolve_example_backend(LLM_PROVIDER)
    KIMI_API_KEY, KIMI_BASE_URL, KIMI_MODEL = _backend.api_key, _backend.base_url, _backend.model

# Model Parameters
MODEL_TEMPERATURE = 0.7
MODEL_MAX_TOKENS = 4096
MODEL_TOP_P = 0.95

# Context Window Configuration
CONTEXT_WINDOW_SIZE = 128000  # Experiment context budget (K3 itself supports up to 1M tokens)
MAX_MEMORY_ENTRIES = 100
MEMORY_COMPRESSION_THRESHOLD = 50  # Compress when memory exceeds this

# Memobase Configuration
MEMOBASE_CONFIG = {
    "memory_types": [
        "episodic",      # Task-specific memories
        "semantic",      # General knowledge
        "procedural",    # Learned procedures and patterns
        "working"        # Short-term working memory
    ],
    "retention_policy": "adaptive",  # adaptive, fixed, or decay
    "compression_strategy": "hierarchical",  # hierarchical, summary, or selective
    "storage_backend": "local",  # local, redis, or postgresql
}

# LOCOMO Benchmark Configuration
LOCOMO_CONFIG = {
    "benchmark_path": Path("benchmarks/locomo"),
    "evaluation_metrics": [
        "task_completion",
        "reasoning_accuracy",
        "memory_utilization",
        "context_efficiency",
        "adaptation_score"
    ],
    "task_categories": [
        "multi_turn_reasoning",
        "long_context_qa",
        "task_planning",
        "knowledge_integration",
        "tool_usage"
    ],
    "max_turns": 20,
    "timeout_seconds": 300
}

# Memory Database Configuration
MEMORY_DB_PATH = Path("memory_store")
MEMORY_DB_PATH.mkdir(exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = Path("logs") / "memobase_agent.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Agent Configuration
AGENT_CONFIG = {
    "name": "MemobaseAgent",
    "version": "1.0.0",
    "capabilities": [
        "long_term_memory",
        "context_compression",
        "adaptive_learning",
        "tool_calling",
        "multi_turn_reasoning"
    ],
    "max_retries": 3,
    "retry_delay": 1.0,
}

# Tool Configuration
ENABLE_WEB_SEARCH = True
ENABLE_CODE_EXECUTION = True
ENABLE_FILE_OPERATIONS = True
ENABLE_DATABASE_ACCESS = True

# Performance Optimization
BATCH_SIZE = 10
CACHE_ENABLED = True
CACHE_TTL = 3600  # 1 hour
PARALLEL_PROCESSING = True
MAX_WORKERS = 4

# Experimental Features
ENABLE_MEMORY_CONSOLIDATION = True  # Consolidate memories during idle time
ENABLE_PREDICTIVE_CACHING = True    # Pre-fetch likely needed memories
ENABLE_ADAPTIVE_COMPRESSION = True  # Adjust compression based on usage patterns
