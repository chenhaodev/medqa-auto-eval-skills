"""
Model abstraction layer for LLM-as-judge.
Supports:
  - claude-haiku-4-5        via Anthropic SDK   (ANTHROPIC_API_KEY)
  - minimax-m2.5            via SiliconFlow      (MINIMAX_API_KEY, MINIMAX_BASE_URL, MINIMAX_MODEL)
  - deepseek-chat           via DeepSeek API     (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)

Loads .env from the project root automatically if present.
"""

import os
import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ── Dotenv loader (no external deps) ─────────────────────────────────────────

def _load_dotenv(overwrite: bool = False) -> None:
    """Load KEY=VALUE pairs from .env in the project root (two levels up from this file).

    Args:
        overwrite: If True, overwrite existing environment variables (useful when .env
                   has been updated since the process started).
    """
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (overwrite or key not in os.environ):
                os.environ[key] = value


_load_dotenv(overwrite=True)


# ── Model identifiers ─────────────────────────────────────────────────────────

CLAUDE_HAIKU = "claude-haiku-4-5"
CLAUDE_HAIKU_FULL = "claude-haiku-4-5-20251001"
MINIMAX_M2 = "minimax-m2.5"
DEEPSEEK_CHAT = "deepseek-chat"
DEEPSEEK_REASONER = "deepseek-reasoner"

DEFAULT_MODEL = CLAUDE_HAIKU

_CLAUDE_PREFIXES = ("claude-",)
_MINIMAX_IDS = {MINIMAX_M2, "minimax", "minimax-m2"}
_DEEPSEEK_IDS = {DEEPSEEK_CHAT, DEEPSEEK_REASONER, "deepseek"}


@dataclass(frozen=True)
class ModelResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


def call_model(prompt: str, model: str = DEFAULT_MODEL, system: str = "") -> ModelResponse:
    """
    Call the specified LLM with the given prompt.

    Args:
        prompt: The user-facing prompt text
        model:  Model identifier (claude-haiku-4-5 | minimax-m2.5 | deepseek-chat | deepseek-reasoner)
        system: Optional system prompt

    Returns:
        ModelResponse with content and token usage
    """
    model_lower = model.lower()

    if any(model_lower.startswith(p) for p in _CLAUDE_PREFIXES):
        canonical = CLAUDE_HAIKU_FULL if model_lower in (CLAUDE_HAIKU, CLAUDE_HAIKU_FULL) else model
        return _call_claude(prompt, model=canonical, system=system)

    if model_lower in _MINIMAX_IDS:
        return _call_openai_compatible(
            prompt=prompt,
            system=system,
            api_key=os.environ.get("MINIMAX_API_KEY", ""),
            base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.siliconflow.cn/v1"),
            model_name=os.environ.get("MINIMAX_MODEL", "siliconflow-cn/Pro/MiniMaxAI/MiniMax-M2.5"),
            model_id=MINIMAX_M2,
        )

    if model_lower in _DEEPSEEK_IDS:
        model_name = os.environ.get(
            "DEEPSEEK_MODEL",
            DEEPSEEK_REASONER if model_lower == DEEPSEEK_REASONER else DEEPSEEK_CHAT,
        )
        return _call_openai_compatible(
            prompt=prompt,
            system=system,
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model_name=model_name,
            model_id=model_lower,
        )

    # Fallback: treat as Claude model ID
    return _call_claude(prompt, model=model, system=system)


# ── Provider implementations ─────────────────────────────────────────────────

def _call_claude(prompt: str, model: str = CLAUDE_HAIKU_FULL, system: str = "") -> ModelResponse:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Run: pip install anthropic") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

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


def _call_openai_compatible(
    prompt: str,
    system: str,
    api_key: str,
    base_url: str,
    model_name: str,
    model_id: str,
) -> ModelResponse:
    """Generic OpenAI-compatible chat completions caller (no openai package required)."""
    if not api_key:
        raise EnvironmentError(f"API key not set for model '{model_id}'")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.0,
    }).encode("utf-8")

    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        raise RuntimeError(f"{model_id} API error {e.code}: {body}") from e

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return ModelResponse(
        content=content,
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        model=model_id,
    )


def available_models() -> list[str]:
    return [CLAUDE_HAIKU, MINIMAX_M2, DEEPSEEK_CHAT, DEEPSEEK_REASONER]
