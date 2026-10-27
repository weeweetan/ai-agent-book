"""
Configuration module for System-Hint Enhanced Agent
"""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from agentbook.providers import PROVIDERS, canonical_provider

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the System-Hint Agent"""
    
    # API Configuration
    api_key: Optional[str] = None
    provider: str = "kimi"
    model: Optional[str] = None
    
    # System Hint Features
    enable_timestamps: bool = True
    enable_tool_counter: bool = True
    enable_todo_list: bool = True
    enable_detailed_errors: bool = True
    enable_system_state: bool = True
    
    # Formatting Options
    timestamp_format: str = "%Y-%m-%d %H:%M:%S"
    simulate_time_delay: bool = False
    
    # Execution Options
    max_iterations: int = 20
    verbose: bool = False
    timeout: int = 30  # seconds for command execution
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables"""
        provider = canonical_provider(os.getenv("LLM_PROVIDER", "kimi"))
        return cls(
            # Provider credentials and aliases come from the shared registry.
            api_key=PROVIDERS.get(provider, PROVIDERS["kimi"]).api_key(),
            provider=provider,
            model=os.getenv("LLM_MODEL") or os.getenv("MODEL_NAME"),
            enable_timestamps=os.getenv("ENABLE_TIMESTAMPS", "true").lower() == "true",
            enable_tool_counter=os.getenv("ENABLE_TOOL_COUNTER", "true").lower() == "true",
            enable_todo_list=os.getenv("ENABLE_TODO_LIST", "true").lower() == "true",
            enable_detailed_errors=os.getenv("ENABLE_DETAILED_ERRORS", "true").lower() == "true",
            enable_system_state=os.getenv("ENABLE_SYSTEM_STATE", "true").lower() == "true",
            timestamp_format=os.getenv("TIMESTAMP_FORMAT", "%Y-%m-%d %H:%M:%S"),
            simulate_time_delay=os.getenv("SIMULATE_TIME_DELAY", "false").lower() == "true",
            max_iterations=int(os.getenv("MAX_ITERATIONS", "20")),
            verbose=os.getenv("VERBOSE", "false").lower() == "true",
            timeout=int(os.getenv("COMMAND_TIMEOUT", "30"))
        )
    
    def validate(self) -> bool:
        """Validate the configuration"""
        if not self.api_key and not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("API key is required. Set the selected provider's key or OPENROUTER_API_KEY fallback.")

        self.provider = canonical_provider(self.provider)
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        
        return True


# Default configuration presets
PRESETS = {
    "full": AgentConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True
    ),
    "minimal": AgentConfig(
        enable_timestamps=False,
        enable_tool_counter=False,
        enable_todo_list=False,
        enable_detailed_errors=False,
        enable_system_state=False
    ),
    "debug": AgentConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True,
        verbose=True
    ),
    "demo": AgentConfig(
        enable_timestamps=True,
        enable_tool_counter=True,
        enable_todo_list=True,
        enable_detailed_errors=True,
        enable_system_state=True,
        simulate_time_delay=True
    )
}


def get_config(preset: Optional[str] = None) -> AgentConfig:
    """
    Get configuration from environment or preset
    
    Args:
        preset: Optional preset name ('full', 'minimal', 'debug', 'demo')
        
    Returns:
        AgentConfig instance
    """
    if preset and preset in PRESETS:
        config = PRESETS[preset]
        # Override with environment API key if available
        environment = AgentConfig.from_env()
        config.provider, config.api_key, config.model = environment.provider, environment.api_key, environment.model
        return config
    
    return AgentConfig.from_env()
