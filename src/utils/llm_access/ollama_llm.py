import json
import re
from datetime import datetime
from typing import Any, Dict

import requests

from src.paths import get_paths
from src.utils.util_files_functions import save_text_to_file


def testing_connection(LLM_CONFIG):
    # Test Ollama connection
    print(f"\n🔌 Testing connection to Ollama ({LLM_CONFIG['base_url']})...")
    try:
        response = requests.get(f"{LLM_CONFIG['base_url']}/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama connection successful")
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Ollama: {e}")


def fix_json_almost(json_text: str) -> str:
    """
    Fixes JSON that:
      - Has single-line comments // ...
      - Is missing commas between key-value pairs
      - Can be a list of objects
    Returns a Python list of dicts.
    """
    # Clean response
    response_text = json_text.strip()
    if "{" in response_text:
        response_text = response_text[response_text.find("{") :]
    if "}" in response_text:
        response_text = response_text[: response_text.rfind("}") + 1]

    # 1️⃣ Remove single-line comments
    no_comments = re.sub(r"//.*", "", response_text)

    # 2️⃣ Add missing commas between quoted key-value pairs
    # Pattern: "value""key":  ->  "value", "key":
    fixed_commas = re.sub(r'(".*?")\s*(".*?")\s*:', r"\1, \2:", no_comments)

    return fixed_commas


def prompt_builder(prompt: str, llm_config: dict) -> Dict[str, Any]:
    """
    Call LLM and return parsed JSON.
    Attempts to fix broken JSON once before failing.

    Args:
        prompt: The extraction prompt
        llm_config: LLM configuration

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If JSON is invalid even after fix attempt
        requests.RequestException: If API call fails
    """
    url = f"{llm_config['base_url']}/api/generate"

    payload = {
        "model": llm_config["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": llm_config["temperature"],
            "num_predict": llm_config["max_tokens"],
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    save_text_to_file(data=prompt, output_file=get_paths().DATA_TEMP_PATH / f"prompts/{timestamp}_prompt.txt", log=False)

    if len(prompt) > llm_config.get("max_prompt_length", 15000):
        print(f"⚠️  Warning: Prompt length ({len(prompt)}) exceeds max limit ({llm_config.get('max_prompt_length', 15000)})")

    try:
        response = requests.post(url, json=payload, timeout=llm_config["timeout"])
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")

        # Try to parse JSON
        try:
            clean_text = fix_json_almost(response_text)
            save_text_to_file(data=clean_text, output_file=get_paths().DATA_TEMP_PATH / f"prompts/{timestamp}_response.txt", log=False)
            parsed = json.loads(clean_text)
            return parsed

        except json.JSONDecodeError as e:
            print(f"⚠️  JSON error: {e},")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            paths = get_paths()
            output_file = paths.DATA_TEMP_PATH / f"{paths.FILE_ENTITIES_RAW.stem}_{timestamp}.txt"
            save_text_to_file(data=response_text, output_file=output_file, log=False)
            raise e

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        raise
