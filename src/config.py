# config.py
# Configuration file for RRGEval project
# This file should be added to .gitignore to prevent sensitive information from being committed

import os
from typing import Optional

class Config:
    """Configuration class for RRGEval project."""
    
    # API Configuration
    API_KEY: str = os.getenv("RRGEVAL_API_KEY", "your_api_key_here")
    API_URL: str = os.getenv("RRGEVAL_API_URL", "http://your_api_url_here")
    
    # Model Configuration
    MODEL_NAME: str = os.getenv("RRGEVAL_MODEL_NAME", "llama3-3-70b-chat")
    
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
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is set."""
        if cls.API_KEY == "your_api_key_here":
            print("Warning: API_KEY not set. Please set RRGEVAL_API_KEY environment variable.")
            return False
        if cls.API_URL == "http://your_api_url_here":
            print("Warning: API_URL not set. Please set RRGEVAL_API_URL environment variable.")
            return False
        return True

# Create a .env.example file for users to copy
def create_env_example():
    """Create a .env.example file with all required environment variables."""
    env_example_content = """# RRGEval Environment Variables
# Copy this file to .env and fill in your actual values

# API Configuration
RRGEVAL_API_KEY=your_actual_api_key_here
RRGEVAL_API_URL=http://your_actual_api_url_here

# Model Configuration
RRGEVAL_MODEL_NAME=llama3-3-70b-chat

# Default parameters
RRGEVAL_MAX_TOKENS=10000
RRGEVAL_TIMEOUT=30
RRGEVAL_TEMPERATURE=0
RRGEVAL_TOP_P=1
RRGEVAL_N=1
RRGEVAL_SEED=123

# Generation specific parameters
RRGEVAL_GENERATION_MAX_TOKENS=130000
RRGEVAL_GENERATION_TIMEOUT=600

# Filtering specific parameters
RRGEVAL_FILTERING_MAX_TOKENS=10
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example_content)
    print("Created .env.example file. Please copy it to .env and fill in your actual values.")

if __name__ == "__main__":
    create_env_example() 