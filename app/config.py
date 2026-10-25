"""Configuration module for the Autonomous Data Analyst Agent.

Loads settings from environment variables (.env file).
Centralizes model initialization for NVIDIA NIM (Muse Glimmer).
Never exposes or logs API keys.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    """LLM provider configuration."""
    provider: str
    api_key: str
    model: str
    base_url: str
    available: bool  # True if a real, valid API key is configured

    @property
    def is_demo_mode(self) -> bool:
        return not self.available

    def __repr__(self) -> str:
        masked_key = f"***{self.api_key[-4:]}" if len(self.api_key) >= 4 else ("***" if self.api_key else "")
        return (
            f"LLMConfig(provider={self.provider!r}, model={self.model!r}, "
            f"base_url={self.base_url!r}, available={self.available!r}, "
            f"api_key={masked_key!r})"
        )


@dataclass(frozen=True)
class ExecutionConfig:
    """Code execution configuration."""
    max_correction_attempts: int
    code_execution_timeout: int
    max_code_size_kb: int
    max_output_size_kb: int
    backend: str = "auto"  # "auto", "docker", "subprocess"


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    llm: LLMConfig
    execution: ExecutionConfig


def load_config(env_file: str | None = None) -> AppConfig:
    """Load and validate configuration from environment variables.
    
    Prioritizes NVIDIA_API_KEY for NVIDIA NIM (Muse Glimmer),
    with backwards-compatible fallback to LLM_API_KEY.
    """
    if env_file is not None:
        load_dotenv(env_file, override=True)
    else:
        load_dotenv()

    # Prioritize NVIDIA_API_KEY, fallback to LLM_API_KEY
    api_key = (os.getenv("NVIDIA_API_KEY") or os.getenv("LLM_API_KEY") or "").strip()
    placeholder_keys = {"", "your_api_key_here", "your_existing_key", "your_key_here"}
    has_key = bool(api_key) and (api_key not in placeholder_keys)

    llm = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "nvidia"),
        api_key=api_key if has_key else "",
        model=os.getenv("LLM_MODEL", "meta/muse-glimmer-30b"),
        base_url=os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        available=has_key,
    )

    execution = ExecutionConfig(
        max_correction_attempts=int(os.getenv("MAX_CORRECTION_ATTEMPTS", "3")),
        code_execution_timeout=int(os.getenv("CODE_EXECUTION_TIMEOUT", "120")),
        max_code_size_kb=int(os.getenv("MAX_CODE_SIZE_KB", "100")),
        max_output_size_kb=int(os.getenv("MAX_OUTPUT_SIZE_KB", "500")),
        backend=os.getenv("EXECUTION_BACKEND", "auto").lower(),
    )

    return AppConfig(llm=llm, execution=execution)


def get_llm(config: AppConfig, timeout: int | None = None, max_retries: int = 2):
    """Centralized LLM factory for NVIDIA Muse Glimmer (NVIDIA NIM).
    
    Creates a LangChain ChatOpenAI client configured for the NVIDIA NIM endpoint.
    Returns None if no API key is available or if client creation fails.
    Never logs or exposes API keys.
    """
    if not config.llm.available:
        return None

    effective_timeout = timeout if timeout is not None else int(os.getenv("LLM_TIMEOUT", "120"))

    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.llm.model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            temperature=0.2,
            max_tokens=4096,
            timeout=effective_timeout,
            max_retries=max_retries,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize LLM ({config.llm.model}): {type(e).__name__}: {e}")
        return None
