#!/usr/bin/env python
r"""
Standalone verification script for Google AI Studio API-key authentication.

Usage (from project root):
    export GEMINI_API_KEY=your_google_ai_studio_api_key
    python test_gemini_auth.py

On Windows PowerShell:
    $env:GEMINI_API_KEY = "your_google_ai_studio_api_key"
    python test_gemini_auth.py

This script:
  - Initialises a Gemini Developer API client with GEMINI_API_KEY
  - Makes one minimal generate_content request
  - Prints only success, model name, and response text
  - Never prints the API key
"""

import os
import sys


def main():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    if not api_key:
        print("ERROR: Required environment variable GEMINI_API_KEY is not set.")
        sys.exit(1)

    print(f"Model      : {model}")
    print("API key set: yes")
    print()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        print("Client initialised successfully. Sending test prompt...")
        response = client.models.generate_content(
            model=model,
            contents="Say 'Hello from the Gemini Developer API!' in one sentence.",
        )

        text = (response.text or "").strip()
        print()
        print(f"✅ SUCCESS")
        print(f"   Model used : {model}")
        print(f"   Response   : {text}")

    except Exception as exc:
        print()
        print(f"❌ FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
