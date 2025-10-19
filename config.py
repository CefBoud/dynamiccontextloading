"""
Example config for the loader project, mimicking the main project's config.py.

Loads .env file and sets up LLM config for LiteLLM.
"""

import os
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()  # Load .env file

llm_config: Dict[str, Optional[str]] = {
    key: value
    for key, value in {
        "model": os.environ.get("MODEL"),
        "api_base": os.environ.get("API_BASE"),
        "api_key": os.environ.get("API_KEY"),
    }.items()
    if value is not None
}
