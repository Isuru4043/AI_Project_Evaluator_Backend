#!/usr/bin/env python
r"""
Standalone verification script for Vertex AI ADC authentication.

Usage (from project root):
    export GOOGLE_APPLICATION_CREDENTIALS=secrets/google-service-account.json
    export GOOGLE_CLOUD_PROJECT=geminikeyaccess
    export GOOGLE_CLOUD_LOCATION=global
    python test_vertex_auth.py

On Windows PowerShell:
    $env:GOOGLE_APPLICATION_CREDENTIALS = (Resolve-Path ".\secrets\google-service-account.json").Path
    $env:GOOGLE_CLOUD_PROJECT = "geminikeyaccess"
    $env:GOOGLE_CLOUD_LOCATION = "global"
    python test_vertex_auth.py

This script:
  - Initialises a genai.Client with vertexai=True, project, and location
  - Relies on ADC via GOOGLE_APPLICATION_CREDENTIALS
  - Makes one minimal generate_content request
  - Prints only success, model name, and response text
  - Never prints credentials or token information
"""

import os
import sys


def main():
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "geminikeyaccess")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "global")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    print(f"Project  : {project}")
    print(f"Location : {location}")
    print(f"Model    : {model}")
    print(f"Creds set: {'yes' if creds_path else 'no (using metadata / ADC chain)'}")
    print()

    try:
        from google import genai

        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
        )

        print("Client initialised successfully. Sending test prompt...")
        response = client.models.generate_content(
            model=model,
            contents="Say 'Hello from Vertex AI!' in one sentence.",
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
