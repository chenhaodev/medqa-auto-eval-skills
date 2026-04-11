"""
Model abstraction layer for LLM-as-judge.
Supports claude-haiku-4-5 via Anthropic SDK and minimax-m2.5 via HTTP.
"""

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any

# Supported model identifiers
CLAUDE_HAIKU = "claude-haiku-4-5"
CLAUDE_HAIKU_FULL = "claude-haiku-4-5-20251001"
MINIMAX_M2 = "minimax-m2.5"

DEFAULT_MODEL = CLAUDE_HAIKU


@dataclass(frozen=True)
class ModelResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


def call_model(prompt: str, model: str = DEFAULT_MODEL, system: str = "") -> ModelResponse:
    """
    Call the specified LLM with the given prompt.
    Returns a ModelResponse with content and token usage.
    """
    if model in (CLAUDE_HAIKU, CLAUDE_HAIKU_FULL, "claude-haiku-4-5-20251001"):
        return _call_claude(prompt, model=CLAUDE_HAIKU_FULL, system=system)
    elif model == MINIMAX_M2:
        return _call_minimax(prompt, system=system)
    else:
        # Try treating as a Claude model ID
        return _call_claude(prompt, model=model, system=system)


def _call_claude(prompt: str, model: str = CLAUDE_HAIKU_FULL, system: str = "") -> ModelResponse:
    """Call Claude via Anthropic SDK."""
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set")

    client = anthropic.Anthropic(api_key=api_key)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    message = client.messages.create(**kwargs)

    return ModelResponse(
        content=message.content[0].text,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        model=model,
    )


def _call_minimax(prompt: str, system: str = "") -> ModelResponse:
    """Call MiniMax M2.5 via HTTP API."""
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise EnvironmentError("MINIMAX_API_KEY environment variable not set")

    api_url = os.environ.get("MINIMAX_API_URL", "https://api.minimax.chat/v1/text/chatcompletion_v2")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": "MiniMax-M1",
        "messages": messages,
        "max_tokens": 2048,
    }).encode("utf-8")

    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"MiniMax API error {e.code}: {body}") from e

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})

    return ModelResponse(
        content=content,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=MINIMAX_M2,
    )


def available_models() -> list[str]:
    """Return list of supported model identifiers."""
    return [CLAUDE_HAIKU, MINIMAX_M2]
