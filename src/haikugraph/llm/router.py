"""LLM router for dispatching to appropriate models based on role.

This module routes LLM calls to the correct model (planner or narrator)
using configuration from environment variables.

Supported providers:
- ollama: Local models via Ollama (default)
- anthropic: Claude models via Anthropic API
- openai: GPT models via OpenAI API

Environment variables:
- HG_LLM_PROVIDER: Provider to use (ollama, anthropic, openai)
- HG_ANTHROPIC_API_KEY: Anthropic API key (for Claude)
- HG_OPENAI_API_KEY: OpenAI API key (for GPT)
- HG_PLANNER_MODEL: Model for planning tasks
- HG_NARRATOR_MODEL: Model for narration tasks
"""

import importlib.util
import os
from typing import Any

from haikugraph.llm.ollama_client import ollama_chat


# Default models per provider
DEFAULT_MODELS = {
    "ollama": {
        "planner": "qwen2.5:14b-instruct",  # Upgraded from 7b
        "narrator": "llama3.1:8b",
        "intent": "qwen2.5:14b-instruct",
    },
    "anthropic": {
        "planner": "claude-3-5-sonnet-20241022",
        "narrator": "claude-3-5-haiku-20241022",
        "intent": "claude-3-5-haiku-20241022",
    },
    "openai": {
        "planner": "gpt-4o",
        "narrator": "gpt-4o-mini",
        "intent": "gpt-4o-mini",
    },
}

# Legacy defaults for backward compatibility
DEFAULT_PLANNER_MODEL = "qwen2.5:14b-instruct"
DEFAULT_NARRATOR_MODEL = "llama3.1:8b"


def _call_anthropic(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
) -> str:
    """Call Anthropic API (Claude models)."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: pip install anthropic"
        )
    
    api_key = os.environ.get("HG_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. "
            "Set HG_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY environment variable."
        )
    
    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    
    # Extract system message if present
    system_content = None
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            api_messages.append(msg)
    
    # Call API
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens or 4096,
        temperature=temperature,
        system=system_content or "You are a helpful data assistant.",
        messages=api_messages,
    )
    
    return response.content[0].text


def _call_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
) -> str:
    """Call OpenAI API (GPT models)."""
    try:
        import importlib
        openai_module = importlib.import_module("openai")
    except ImportError:
        raise ImportError(
            "openai package not installed. "
            "Install with: pip install openai"
        )
    
    api_key = os.environ.get("HG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. "
            "Set HG_OPENAI_API_KEY or OPENAI_API_KEY environment variable."
        )
    
    client = openai_module.OpenAI(api_key=api_key, timeout=timeout)
    
    # Call API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens or 4096,
    )
    
    return response.choices[0].message.content


def call_llm(
    messages: list[dict[str, str]],
    *,
    role: str = "planner",
    max_tokens: int | None = None,
    timeout: int = 60,
    provider: str | None = None,
    model: str | None = None,
    temperature_override: float | None = None,
) -> str:
    """Route LLM call to appropriate model based on role and provider.

    Args:
        messages: List of message dicts with 'role' and 'content'
        role: One of 'planner', 'intent', or 'narrator' to select the appropriate model
        max_tokens: Maximum tokens in response (optional)
        timeout: Request timeout in seconds

    Returns:
        Response text content

    Raises:
        ValueError: If role is invalid or LLM call fails
    """
    resolved_provider = (provider or os.environ.get("HG_LLM_PROVIDER", "ollama")).lower()
    
    # Validate role
    if role not in ["planner", "intent", "narrator"]:
        raise ValueError(
            f"Invalid role: {role}. Must be 'planner', 'intent', or 'narrator'"
        )
    
    # Get model for this role and provider
    default_model = DEFAULT_MODELS.get(resolved_provider, {}).get(role, DEFAULT_PLANNER_MODEL)
    
    if role == "planner":
        role_model = os.environ.get("HG_PLANNER_MODEL", default_model)
        temperature = float(os.environ.get("HG_PLANNER_TEMPERATURE", "0"))
    elif role == "intent":
        role_model = os.environ.get("HG_INTENT_MODEL", default_model)
        temperature = float(os.environ.get("HG_INTENT_TEMPERATURE", "0"))
    elif role == "narrator":
        role_model = os.environ.get("HG_NARRATOR_MODEL", default_model)
        temperature = float(os.environ.get("HG_NARRATOR_TEMPERATURE", "0.3"))
        timeout = max(timeout, 90)
    
    if model is not None:
        role_model = model
    if temperature_override is not None:
        temperature = temperature_override
    
    # Route to appropriate provider
    if resolved_provider == "anthropic":
        return _call_anthropic(messages, role_model, temperature, max_tokens, timeout)
    elif resolved_provider == "openai":
        return _call_openai(messages, role_model, temperature, max_tokens, timeout)
    elif resolved_provider == "ollama":
        return ollama_chat(
            messages,
            model=role_model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: {resolved_provider}. "
            "Supported: ollama, anthropic, openai"
        )


def _has_module(module_name: str) -> bool:
    """Return True when a module is installed in the current environment."""
    return importlib.util.find_spec(module_name) is not None


def get_available_providers() -> list[str]:
    """Get list of available LLM providers based on installed packages and API keys."""
    available = ["ollama"]  # Always available
    
    if _has_module("anthropic") and (
        os.environ.get("HG_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    ):
        available.append("anthropic")
    
    if _has_module("openai") and (
        os.environ.get("HG_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    ):
        available.append("openai")
    
    return available


def get_current_config() -> dict[str, Any]:
    """Get current LLM configuration."""
    provider = os.environ.get("HG_LLM_PROVIDER", "ollama").lower()
    return {
        "provider": provider,
        "planner_model": os.environ.get(
            "HG_PLANNER_MODEL",
            DEFAULT_MODELS.get(provider, {}).get("planner", DEFAULT_PLANNER_MODEL),
        ),
        "narrator_model": os.environ.get(
            "HG_NARRATOR_MODEL",
            DEFAULT_MODELS.get(provider, {}).get("narrator", DEFAULT_NARRATOR_MODEL),
        ),
        "available_providers": get_available_providers(),
    }
