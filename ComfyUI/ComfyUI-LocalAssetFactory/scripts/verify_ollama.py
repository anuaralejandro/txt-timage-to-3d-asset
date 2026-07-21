#!/usr/bin/env python3
"""
verify_ollama.py — Verify the local Ollama server is running and configured.

Run from any directory:
    python scripts/verify_ollama.py
"""

import os
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not installed. Run: pip install requests")
    sys.exit(1)


def main():
    print("=" * 60)
    print("  LocalAssetFactory — Ollama Verification")
    print("=" * 60)
    print()

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip().rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "mistral").strip()

    print(f"  Ollama URL:   {base_url}")
    print(f"  Target model: {model}")
    print()

    # 1. Connectivity
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=10)
        r.raise_for_status()
        print("  ✓  Ollama server is reachable")
    except requests.ConnectionError:
        print(f"  ✗  Cannot connect to Ollama at {base_url}")
        print("     → Make sure Ollama is running: ollama serve")
        return 1
    except Exception as exc:
        print(f"  ✗  Ollama connection error: {exc}")
        return 1

    # 2. List models
    try:
        data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        print(f"  ✓  Available models ({len(models)}):")
        for m in models:
            marker = " ←" if model in m else ""
            print(f"       - {m}{marker}")
    except Exception as exc:
        print(f"  ⚠  Could not parse model list: {exc}")
        models = []

    # 3. Check target model
    if models:
        found = any(model in m for m in models)
        if found:
            print(f"\n  ✓  Target model '{model}' is available")
        else:
            print(f"\n  ⚠  Target model '{model}' NOT found in available models")
            print(f"     → Pull it with: ollama pull {model}")
    else:
        print(f"\n  ⚠  No models found. Pull one with: ollama pull {model}")

    # 4. Quick generation test
    print("\n  Testing generation...")
    try:
        r = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "Say 'hello' in one word.",
                "stream": False,
                "options": {"num_predict": 10},
            },
            timeout=60,
        )
        r.raise_for_status()
        response = r.json().get("response", "").strip()
        print(f"  ✓  Generation test passed. Response: {response[:50]}")
    except Exception as exc:
        print(f"  ✗  Generation test failed: {exc}")
        return 1

    print("\n  ✅  Ollama verification complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
