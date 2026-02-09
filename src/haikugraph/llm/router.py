"""LLM router for dispatching to appropriate models based on role.

This module routes LLM calls to the correct model (planner or narrator)
using configuration from environment variables.
"""

import os

from haikugraph.llm.ollama_client import ollama_chat


DEFAULT_PLANNER_MODEL = "qwen2.5:7b-instruct"
DEFAULT_NARRATOR_MODEL = "llama3.1:8b"


def call_llm(
    messages: list[dict[str, str]],
    *,
    role: str = "planner",
    max_tokens: int | None = None,
    timeout: int = 30,
) -> str:
    """Route LLM call to appropriate model based on role.

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
    provider = os.environ.get("HG_LLM_PROVIDER", "ollama")
    
    if provider != "ollama":
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            "A6 requires HG_LLM_PROVIDER=ollama"
        )
    
    if role == "planner":
        model = os.environ.get("HG_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
        temperature = float(os.environ.get("HG_PLANNER_TEMPERATURE", "0"))
    elif role == "intent":
        # Intent classification uses same model as planner with deterministic temp
        model = os.environ.get("HG_PLANNER_MODEL", DEFAULT_PLANNER_MODEL)
        temperature = float(os.environ.get("HG_INTENT_TEMPERATURE", "0"))
    elif role == "narrator":
        model = os.environ.get("HG_NARRATOR_MODEL", DEFAULT_NARRATOR_MODEL)
        temperature = float(os.environ.get("HG_NARRATOR_TEMPERATURE", "0.4"))
    else:
        raise ValueError(
            f"Invalid role: {role}. Must be 'planner', 'intent', or 'narrator'"
        )
    
    return ollama_chat(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
