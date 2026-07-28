#!/usr/bin/env python
r"""
Standalone verification script for Vertex AI service-account authentication.

Usage (from project root):
    export GOOGLE_APPLICATION_CREDENTIALS=credentials/google-service-account.json
    export GOOGLE_CLOUD_PROJECT=geminikeyaccess
    export GOOGLE_CLOUD_LOCATION=global
    python test_vertex_auth.py

On Windows PowerShell:
    $env:GOOGLE_APPLICATION_CREDENTIALS = "credentials/google-service-account.json"
    $env:GOOGLE_CLOUD_PROJECT = "geminikeyaccess"
    $env:GOOGLE_CLOUD_LOCATION = "global"
    python test_vertex_auth.py

This script:
  - Loads the project's centralized Vertex AI client
  - Uses ADC through GOOGLE_APPLICATION_CREDENTIALS
  - Makes one minimal generate_content request
  - Prints only success, model name, and response text
  - Never prints credential contents or tokens
"""

import os
import sys


def main():
    try:
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            "AI_Evaluator_Backend.settings",
        )
        import django
        django.setup()

        from django.conf import settings
        from AI_Evaluator_Backend.llm import get_llm

        if not settings.GOOGLE_CLOUD_PROJECT:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured.")
        if not settings.GOOGLE_APPLICATION_CREDENTIALS:
            raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not configured.")

        print(f"Project         : {settings.GOOGLE_CLOUD_PROJECT}")
        print(f"Location        : {settings.GOOGLE_CLOUD_LOCATION}")
        print(f"Model           : {settings.GEMINI_MODEL}")
        print("Credential file : configured")
        print()

        client = get_llm()

        print("Client initialised successfully. Sending test prompt...")
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents="Say 'Hello from Vertex AI!' in one sentence.",
        )

        text = (response.text or "").strip()
        print()
        print(f"✅ SUCCESS")
        print(f"   Model used : {settings.GEMINI_MODEL}")
        print(f"   Response   : {text}")

    except Exception as exc:
        print()
        print(f"❌ FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
