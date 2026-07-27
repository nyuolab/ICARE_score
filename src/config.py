# config.py
# Configuration file for RRGEval project
# Environment variables are loaded from .env file by the shell scripts before running Python.
# See .env.example for the full list of configurable variables.

import os
from typing import Optional


class Config:
    """Configuration class for RRGEval project."""

    # Base data path - root directory containing RRG_models/, RRG_evaluation/, cxr_report_datasets/
    BASE_DATA_PATH: str = os.getenv("RRGEVAL_BASE_DATA_PATH", "")

    # API Configuration
    API_KEY: str = os.getenv("RRGEVAL_API_KEY", "your_api_key_here")
    API_URL: str = os.getenv("RRGEVAL_API_URL", "http://your_api_url_here")

    # Model Configuration
    MODEL_NAME: str = os.getenv("RRGEVAL_MODEL_NAME", "llama-3-3-70b-chat")

    # Default parameters
    DEFAULT_MAX_TOKENS: int = int(os.getenv("RRGEVAL_MAX_TOKENS", "10000"))
    DEFAULT_TIMEOUT: int = int(os.getenv("RRGEVAL_TIMEOUT", "30"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("RRGEVAL_TEMPERATURE", "0"))
    DEFAULT_TOP_P: float = float(os.getenv("RRGEVAL_TOP_P", "1"))
    DEFAULT_N: int = int(os.getenv("RRGEVAL_N", "1"))
    DEFAULT_SEED: int = int(os.getenv("RRGEVAL_SEED", "123"))

    # Generation specific parameters
    GENERATION_MAX_TOKENS: int = int(os.getenv("RRGEVAL_GENERATION_MAX_TOKENS", "130000"))
    GENERATION_TIMEOUT: int = int(os.getenv("RRGEVAL_GENERATION_TIMEOUT", "600"))

    # Filtering specific parameters
    FILTERING_MAX_TOKENS: int = int(os.getenv("RRGEVAL_FILTERING_MAX_TOKENS", "10"))

    # API auth header type: "bearer" (standard, for vLLM/Ollama/OpenAI) or "apikey" / "api-key" (legacy/private)
    API_AUTH_HEADER_TYPE: str = os.getenv("RRGEVAL_API_AUTH_HEADER_TYPE", "bearer")

    # Document type selects an internal prompt pack:
    #   radiology -> prompts/radiology_specific (default)
    #   generic   -> prompts/generic
    DOCUMENT_TYPE: str = os.getenv("DOCUMENT_TYPE", "radiology")

    # Maintainer override for prompt pack directory (optional; prefer DOCUMENT_TYPE)
    PROMPT_DIR: str = os.getenv("ICARE_PROMPT_DIR", "")

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is set."""
        valid = True
        if not cls.BASE_DATA_PATH:
            print("Warning: BASE_DATA_PATH not set. Please set RRGEVAL_BASE_DATA_PATH in .env file.")
            valid = False
        if cls.API_KEY == "your_api_key_here":
            print("Warning: API_KEY not set. Please set RRGEVAL_API_KEY in .env file.")
            valid = False
        if cls.API_URL == "http://your_api_url_here":
            print("Warning: API_URL not set. Please set RRGEVAL_API_URL in .env file.")
            valid = False
        return valid 