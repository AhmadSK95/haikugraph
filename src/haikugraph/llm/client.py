"""LLM client wrapper for OpenAI API calls."""

import json
import os
from typing import Any


def call_openai(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> str:
    """
    Call OpenAI API with messages.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name (defaults to gpt-4o-mini)
        temperature: Temperature for sampling (default: 0 for deterministic)
        max_tokens: Maximum tokens in response

    Returns:
        Response text content

    Raises:
        ValueError: If OPENAI_API_KEY not set or API call fails
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. Please set it to your OpenAI API key."
        )

    try:
        import openai
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        ) from None

    if model is None:
        model = "gpt-4o-mini"

    client = openai.OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        raise ValueError(f"OpenAI API call failed: {e}") from e


def parse_json_response(response: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response, handling common formatting issues.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dict

    Raises:
        json.JSONDecodeError: If response is not valid JSON
    """
    # Strip markdown code blocks if present
    text = response.strip()
    if text.startswith("```"):
        # Remove markdown fences
        lines = text.split("\n")
        # Find start and end of code block
        start_idx = 0
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if i == 0 and line.startswith("```"):
                start_idx = i + 1
            elif i > 0 and line.strip().startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    return json.loads(text)
