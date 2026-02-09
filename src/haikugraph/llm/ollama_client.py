"""Ollama client wrapper for local LLM inference.

This module provides a thin wrapper around the Ollama API for HaikuGraph,
supporting retry logic and proper error handling for MacBook-friendly local models.
"""

import json
import os
import time
from typing import Any

import requests


def ollama_chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 30,
) -> str:
    """Call Ollama API with messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Ollama model name (e.g. qwen2.5:7b-instruct)
        temperature: Temperature for sampling (default: 0 for deterministic)
        max_tokens: Maximum tokens in response (optional, Ollama handles internally)
        timeout: Request timeout in seconds

    Returns:
        Response text content

    Raises:
        ValueError: If Ollama API call fails after retries
        ConnectionError: If cannot connect to Ollama service
    """
    base_url = os.environ.get("HG_OLLAMA_BASE_URL", "http://localhost:11434")
    max_retries = int(os.environ.get("HG_MAX_RETRIES", "2"))
    
    endpoint = f"{base_url}/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    
    # Add max_tokens if specified (Ollama calls it num_predict)
    if max_tokens is not None:
        payload["options"]["num_predict"] = max_tokens
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            
            result = response.json()
            if "message" not in result or "content" not in result["message"]:
                raise ValueError(f"Unexpected Ollama response format: {result}")
            
            return result["message"]["content"]
        
        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries:
                # Exponential backoff: 0.5s, 1s, 2s
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {base_url}. "
                    "Ensure Ollama is running (ollama serve or Ollama app)."
                ) from e
        
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                # Retry on timeout
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                raise ValueError(
                    f"Ollama request timed out after {timeout}s (model: {model})"
                ) from e
        
        except requests.exceptions.HTTPError as e:
            last_error = e
            # 5xx errors are transient, retry
            if 500 <= response.status_code < 600 and attempt < max_retries:
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                raise ValueError(
                    f"Ollama API error ({response.status_code}): {response.text}"
                ) from e
        
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 0.5 * (2 ** attempt)
                time.sleep(wait_time)
                continue
            else:
                raise ValueError(f"Unexpected error calling Ollama: {e}") from e
    
    # Should not reach here, but just in case
    raise ValueError(f"Failed after {max_retries} retries. Last error: {last_error}")
