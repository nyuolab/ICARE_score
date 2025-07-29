# src/utils.py
import requests
import json
import os
from typing import Dict, Any, Optional, List
from config import Config

def make_llama_request(
    prompt: str,
    url: str = Config.API_URL,
    api_key: str = Config.API_KEY,
    max_tokens: int = Config.DEFAULT_MAX_TOKENS,
    temperature: float = Config.DEFAULT_TEMPERATURE,
    timeout: int = Config.DEFAULT_TIMEOUT,
    model: str = Config.MODEL_NAME,
    seed: int = Config.DEFAULT_SEED,
    top_p: float = Config.DEFAULT_TOP_P,
    n: int = Config.DEFAULT_N,
    stream: bool = False,
    stop: Optional[List[str]] = None,
    frequency_penalty: float = 0.0
) -> Optional[Dict[str, Any]]:
    """Make a request to the LLAMA API."""
    
    headers = {
        "apiKey": api_key,
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "stream": stream,
        "stop": stop,
        "seed": seed,
        "frequency_penalty": frequency_penalty
    }


    try:
        response = requests.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        predicted_answer = response.json()["choices"][0]["message"]["content"].strip()
        return predicted_answer
    except Exception as e:
        print(f"Error in API call: {e}")
        return None

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)
