"""Central configuration loaded from environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from project root (or current working directory)
_load_path = Path(__file__).resolve().parent.parent / ".env"
if _load_path.exists():
    load_dotenv(_load_path)
else:
    load_dotenv()

# --- LLM settings ---
LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
LLM_BASE_URL: str | None = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
BELIEF_MODEL: str = os.getenv("BELIEF_MODEL", "gpt-4o")


def get_openai_client(**overrides) -> OpenAI:
    """Return an OpenAI client configured from env vars.

    Any keyword arguments (e.g. api_key, base_url) override env values.
    """
    kwargs: dict = {"api_key": LLM_API_KEY}
    if LLM_BASE_URL:
        kwargs["base_url"] = LLM_BASE_URL
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return OpenAI(**kwargs)
