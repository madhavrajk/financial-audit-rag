"""Runtime configuration, loaded from environment variables (and an optional .env file).

Everything tunable lives here so the same code runs locally, in Docker, and in evals
with different settings and no code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where to create an API key for each provider preset in .env.example.
KEY_PAGES = {
    "generativelanguage.googleapis.com": (
        "a free Google Gemini key",
        "https://aistudio.google.com/apikey",
    ),
    "openrouter.ai": ("an OpenRouter key", "https://openrouter.ai/keys"),
    "api.openai.com": ("an OpenAI key", "https://platform.openai.com/api-keys"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # --- Model provider (any OpenAI-compatible API) ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None  # optional for local servers (Ollama, vLLM, ...)
    llm_model: str = ""  # required: the model ID as your provider names it
    # Used once if the grounding repair fails; empty string disables escalation.
    escalation_model: str = ""
    # Cheap model for follow-up rewriting; defaults to llm_model when empty.
    rewrite_model: str = ""
    # Embeddings may come from a different provider; base URL and key default to the LLM's.
    embed_base_url: str = ""
    embed_api_key: SecretStr | None = None
    # Empty disables dense search (BM25 only). The prebuilt index uses Qwen3-Embedding-8B.
    embed_model: str = ""
    embed_dims: int = 1024
    judge_model: str = ""  # evals only; defaults to llm_model (prefer a different family)
    # Extra/override prices, USD per 1M tokens: {"model": [input, cached_input, output]}.
    model_prices: dict[str, tuple[float, float, float]] = Field(default_factory=dict)

    # --- Agent behaviour ---
    max_steps: int = Field(default=8, ge=1, le=20)
    # Hard cost cap: cumulative prompt tokens per request before a forced final answer
    # (a normal question uses ~10-20k; runaway searching hit 300k before this guard).
    max_request_tokens: int = 120_000
    max_output_tokens: int = 4096
    temperature: float = 0.0
    llm_timeout_s: float = 90.0
    request_timeout_s: float = 180.0
    # Upper bound for the adaptive (AIMD) limiter; it backs off on 429s automatically.
    max_concurrent_llm_calls: int = 32
    reasoning_effort: str = "low"  # sent only to models that accept it

    # --- Data ---
    db_path: Path = PROJECT_ROOT / "data" / "financials.db"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    filings_dir: Path = PROJECT_ROOT / "data" / "filings"  # companies added with `ingest add`
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    trace_dir: Path = PROJECT_ROOT / "logs"
    # Only for downloading filings by ticker: SEC requires a contact, "Name email@example.com".
    sec_user_agent: str = ""

    # --- Caching ---
    answer_cache: str = "memory"  # "memory" | "none"
    answer_cache_ttl_s: int = 24 * 3600
    answer_cache_max_entries: int = 1000

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000
    rate_limit_per_minute: int = 30
    max_question_chars: int = 4000

    @property
    def api_key(self) -> str | None:
        return self.llm_api_key.get_secret_value() if self.llm_api_key else None

    @property
    def embed_url(self) -> str:
        return self.embed_base_url or self.llm_base_url

    @property
    def embed_key(self) -> str | None:
        return self.embed_api_key.get_secret_value() if self.embed_api_key else self.api_key

    @staticmethod
    def is_local(url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host in {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}

    def provider_problems(self) -> list[str]:
        """Missing provider settings, phrased as fixes (served by /readyz)."""
        if not self.llm_model:  # nothing configured yet; a missing key would only add noise
            what, page = KEY_PAGES["generativelanguage.googleapis.com"]
            return [
                "No model configured. Copy .env.example to .env (it is set up for Google "
                f"Gemini's free tier), create {what} at {page}, paste it after LLM_API_KEY= "
                "in .env, and restart."
            ]
        if not self.api_key and not self.is_local(self.llm_base_url):
            known = KEY_PAGES.get(urlparse(self.llm_base_url).hostname or "")
            fix = (
                f"Create {known[0]} at {known[1]}, paste it after LLM_API_KEY= in .env"
                if known
                else f"It is required by {self.llm_base_url}; paste your provider's key "
                "after LLM_API_KEY= in .env"
            )
            return [f"LLM_API_KEY is not set. {fix}, and restart."]
        return []


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
