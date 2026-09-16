"""Optional Claude API layer.

The app is fully functional without an API key — every feature has a local
fallback. When a key is present, summaries and answers are written by Claude
instead of being assembled from the document's own sentences.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "claude-sonnet-5"


def _read_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if value:
        return value.strip()
    try:  # Streamlit Cloud stores keys in st.secrets
        import streamlit as st

        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def get_api_key() -> str:
    return _read_secret("ANTHROPIC_API_KEY")


def get_model() -> str:
    return _read_secret("ANTHROPIC_MODEL") or DEFAULT_MODEL


def ai_available() -> bool:
    if not get_api_key():
        return False
    try:
        import anthropic  # noqa: F401

        return True
    except ImportError:
        return False


def complete(prompt: str, system: str = "", max_tokens: int = 1200) -> str:
    """Send one prompt to Claude. Raises on failure so callers can fall back."""
    import anthropic

    client = anthropic.Anthropic(api_key=get_api_key())
    response = client.messages.create(
        model=get_model(),
        max_tokens=max_tokens,
        system=system or "You are a careful document analyst.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "\n".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    ).strip()
