import json
from typing import Any, Dict

import requests


def testing_connection(LLM_CONFIG):
    # Test Ollama connection
    print(f"\n🔌 Testing connection to Ollama ({LLM_CONFIG['base_url']})...")
    try:
        response = requests.get(f"{LLM_CONFIG['base_url']}/api/tags", timeout=5)
        response.raise_for_status()
        print("✅ Ollama connection successful")
    except Exception as e:
        raise ConnectionError(f"Cannot connect to Ollama: {e}")


def fix_broken_json(broken_json: str, error_message: str, llm_config: dict) -> str:
    """
    Ask LLM to fix broken JSON.

    Args:
        broken_json: The malformed JSON string
        error_message: The error message from json.loads()
        llm_config: LLM configuration

    Returns:
        Fixed JSON string (not parsed)
    """
    fix_prompt = f"""The following JSON has a syntax error. Fix it and return ONLY the corrected JSON.

ERROR: {error_message}

BROKEN JSON:
{broken_json}

Return ONLY the fixed JSON. No explanation. Start with {{ and end with }}.
"""

    url = f"{llm_config['base_url']}/api/generate"

    payload = {
        "model": llm_config["model"],
        "prompt": fix_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Deterministic for fixing
            "num_predict": llm_config["max_tokens"],
        },
    }

    response = requests.post(url, json=payload, timeout=llm_config["timeout"])
    response.raise_for_status()

    result = response.json()
    response_text = result.get("response", "")

    # Clean response
    response_text = response_text.strip()
    if "{" in response_text:
        response_text = response_text[response_text.find("{") :]
    if "}" in response_text:
        response_text = response_text[: response_text.rfind("}") + 1]

    return response_text


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

    if len(prompt) > llm_config.get("max_prompt_length", 15000):
        print(f"⚠️  Warning: Prompt length ({len(prompt)}) exceeds max limit ({llm_config.get('max_prompt_length', 15000)})")

    try:
        response = requests.post(url, json=payload, timeout=llm_config["timeout"])
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")

        # Clean response
        response_text = response_text.strip()
        if "{" in response_text:
            response_text = response_text[response_text.find("{") :]
        if "}" in response_text:
            response_text = response_text[: response_text.rfind("}") + 1]

        # Try to parse JSON
        try:
            parsed = json.loads(response_text)
            return parsed

        except json.JSONDecodeError as e:
            # First parse failed - try to fix
            print(f"⚠️  JSON error: {e}, attempting fix...")

            try:
                fixed_text = fix_broken_json(response_text, str(e), llm_config)
                parsed = json.loads(fixed_text)
                print("✓ JSON fixed successfully")
                return parsed

            except Exception as fix_error:
                # Fix failed - re-raise original error
                print(f"✗ Fix failed: {fix_error}")
                print(f"Original response (first 500 chars): {response_text[:500]}")
                raise e

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        raise
