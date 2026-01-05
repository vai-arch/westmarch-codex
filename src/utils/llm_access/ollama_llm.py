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
