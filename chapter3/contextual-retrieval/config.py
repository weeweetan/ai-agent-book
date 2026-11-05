"""Configuration for Agentic RAG System"""

# Locate the shared package when this example is run from a source checkout.
import sys as _sys
from pathlib import Path as _CheckoutPath

_sys.path.insert(0, str(_CheckoutPath(__file__).resolve().parents[2]))

from agentbook.providers import PROVIDERS, canonical_provider, resolve_example_backend

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


def _openrouter_model_id(model: Optional[str]) -> str:
    """Map a provider-native model name to an OpenRouter model id, used by the
    universal OpenRouter fallback. An explicit OPENROUTER_MODEL env var wins."""
    override = os.getenv("OPENROUTER_MODEL")
    if override:
        return override
    m = (model or "").strip()
    if not m:
        return "openai/gpt-5.6-luna"
    if "/" in m:
        return m  # already an OpenRouter-style id (e.g. openai/gpt-5.6-luna)
    ml = m.lower()
    if ml.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai/" + m
    if ml.startswith("claude-"):
        return "anthropic/claude-opus-4.8"
    if ml.startswith("kimi"):
        # kimi-k3 is not on OpenRouter; moonshotai/kimi-k2.6 is the closest hosted id.
        return "moonshotai/kimi-k2.6"
    # Provider-native ids (kimi-*/doubao-*/qwen/deepseek-*) not hosted on
    # OpenRouter under the same name -> a widely-available OpenAI chat model.
    return "openai/gpt-5.6-luna"


class Provider(str, Enum):
    """Supported LLM providers"""
    DASHSCOPE = "dashscope"  # Alibaba Cloud Model Studio / Bailian (Qwen)
    SILICONFLOW = "siliconflow"
    DOUBAO = "doubao"
    KIMI = "kimi"
    MOONSHOT = "moonshot"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GROQ = "groq"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"


class KnowledgeBaseType(str, Enum):
    """Knowledge base backend types"""
    LOCAL = "local"  # Local retrieval pipeline
    DIFY = "dify"    # Dify knowledge base API
    RAPTOR = "raptor"  # RAPTOR tree-based index
    GRAPHRAG = "graphrag"  # GraphRAG graph-based index


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "kimi"  # Default provider
    model: Optional[str] = None  # Will use provider defaults if not specified
    api_key: Optional[str] = None  # Will read from env if not provided
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = True
    
    # Provider-specific defaults
    PROVIDER_DEFAULTS = {
        "dashscope": {
            "model": "qwen3.7-plus",
            "base_url": os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        },
        "siliconflow": {
            "model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "base_url": "https://api.siliconflow.cn/v1"
        },
        "doubao": {
            "model": "doubao-seed-1-6-thinking-250715",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3"
        },
        "kimi": {
            "model": "kimi-k3",
            "base_url": "https://api.moonshot.cn/v1"
        },
        "moonshot": {
            "model": "kimi-k3",
            "base_url": "https://api.moonshot.cn/v1"
        },
        "openrouter": {
            "model": "openai/gpt-5.6-luna",
            "base_url": "https://openrouter.ai/api/v1"
        },
        "openai": {
            "model": "gpt-5.6-luna",
            "base_url": "https://api.openai.com/v1"
        },
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1"
        },
        "together": {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "base_url": "https://api.together.xyz"
        },
        "deepseek": {
            "model": "deepseek-flash",
            "base_url": "https://api.deepseek.com/v1"
        }
    }
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key from environment"""
        env_mappings = {
            "dashscope": "DASHSCOPE_API_KEY",
            "qwen": "DASHSCOPE_API_KEY",
            "bailian": "DASHSCOPE_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY",
            "doubao": "ARK_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        return os.getenv(env_mappings.get(provider.lower(), ""))
    
    def get_client_config(self) -> tuple[Dict[str, Any], str]:
        provider = canonical_provider(self.provider)
        if provider in PROVIDERS:
            backend = resolve_example_backend(provider, self.model, self.api_key)
            return {"api_key": backend.api_key, "base_url": backend.base_url}, backend.model
        # Retain the experiment's Groq / Together support.
        defaults = self.PROVIDER_DEFAULTS.get(provider)
        if defaults is None:
            raise ValueError(f"Unsupported provider: {provider}")
        key = self.api_key or os.getenv(f"{provider.upper()}_API_KEY")
        if not key:
            raise ValueError(f"Set {provider.upper()}_API_KEY")
        return {"api_key": key, "base_url": defaults["base_url"]}, self.model or defaults["model"]


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base configuration"""
    type: KnowledgeBaseType = KnowledgeBaseType.LOCAL
    
    # Local retrieval pipeline config
    local_base_url: str = "http://localhost:4242"
    local_top_k: int = 3
    
    # Dify config
    dify_api_key: Optional[str] = field(default_factory=lambda: os.getenv("DIFY_API_KEY"))
    dify_base_url: str = "https://api.dify.ai/v1"
    dify_dataset_id: Optional[str] = None
    dify_top_k: int = 10
    
    # RAPTOR tree-based index config
    raptor_base_url: str = "http://localhost:4242"
    raptor_top_k: int = 10
    raptor_search_levels: bool = True  # Search across multiple tree levels
    
    # GraphRAG graph-based index config
    graphrag_base_url: str = "http://localhost:4242"
    graphrag_top_k: int = 10
    graphrag_search_type: str = "hybrid"  # entity, community, or hybrid
    
    # Document storage
    document_store_path: str = "document_store.json"
    
    
@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    chunk_size: int = 2048  # Characters per chunk
    max_chunk_size: int = 1024  # Max size when respecting paragraph boundaries
    chunk_overlap: int = 200  # Overlap between chunks
    respect_paragraph_boundary: bool = True
    min_chunk_size: int = 100  # Minimum chunk size


@dataclass 
class AgentConfig:
    """Agent configuration"""
    max_iterations: int = 10  # Max reasoning iterations
    enable_reasoning_trace: bool = True
    enable_citations: bool = True
    strict_knowledge_base: bool = True  # Only answer from knowledge base
    conversation_history_limit: int = 20  # Max conversation turns to keep
    verbose: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    dataset_path: str = "evaluation/legal_qa_dataset.json"
    results_path: str = "evaluation/results"
    metrics: list = field(default_factory=lambda: ["accuracy", "relevance", "citation_quality"])


@dataclass
class Config:
    """Main configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        config = cls()
        
        # Override from env
        if provider := os.getenv("LLM_PROVIDER"):
            config.llm.provider = provider
        if model := os.getenv("LLM_MODEL"):
            config.llm.model = model
        if kb_type := os.getenv("KB_TYPE"):
            config.knowledge_base.type = KnowledgeBaseType(kb_type.lower())
        
        return config
