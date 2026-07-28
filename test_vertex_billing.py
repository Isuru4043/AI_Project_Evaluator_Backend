#!/usr/bin/env python
r"""
Vertex AI billing verification script.

Generates synthetic input (~30 000 tokens), sends 3 sequential requests via
the centralized Vertex AI client, and reports token usage with an estimated cost.

Prerequisites:
    pip install google-genai google-auth

Usage (PowerShell):
    $env:GOOGLE_APPLICATION_CREDENTIALS = "credentials/google-service-account.json"
    $env:GOOGLE_CLOUD_PROJECT    = "geminikeyaccess"
    $env:GOOGLE_CLOUD_LOCATION   = "global"
    $env:GEMINI_MODEL            = "gemini-3.1-flash-lite"
    $env:RUN_VERTEX_BILLING_TEST = "YES"
    python test_vertex_billing.py

Usage (bash / zsh):
    export GOOGLE_APPLICATION_CREDENTIALS=credentials/google-service-account.json
    export GOOGLE_CLOUD_PROJECT=geminikeyaccess
    export GOOGLE_CLOUD_LOCATION=global
    export GEMINI_MODEL=gemini-3.1-flash-lite
    export RUN_VERTEX_BILLING_TEST=YES
    python test_vertex_billing.py

Never prints credential contents, private keys, access tokens, or full paths.
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent / ".env")


# ── helpers ──────────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _require_env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        print(f"ERROR: Required environment variable {name} is not set.")
        sys.exit(1)
    return value


# ── synthetic input builder ──────────────────────────────────────────────────

_PASSAGE = (
    "Software engineering is the systematic application of engineering "
    "approaches to the development of software. It encompasses the entire "
    "software development lifecycle including requirements analysis, system "
    "design, implementation, testing, deployment, and maintenance. Modern "
    "software engineering practices emphasise iterative development, "
    "continuous integration and continuous delivery, automated testing, "
    "code review, and infrastructure as code. Cloud-native architectures "
    "leverage containerisation, microservices, service meshes, and "
    "declarative APIs to build resilient, scalable, and observable systems. "
    "Observability is achieved through structured logging, distributed "
    "tracing, and metrics collection using open standards such as "
    "OpenTelemetry. Security is integrated throughout the lifecycle via "
    "threat modelling, static analysis, dependency scanning, and runtime "
    "protection. Data engineering pipelines ingest, transform, and serve "
    "data using batch and stream processing frameworks. Machine learning "
    "operations extend DevOps practices to model training, validation, "
    "deployment, and monitoring. Effective technical documentation covers "
    "architecture decision records, API specifications, runbooks, and "
    "postmortem analyses. Performance engineering involves load testing, "
    "profiling, capacity planning, and autoscaling configuration. "
    "Reliability engineering defines service level objectives, error "
    "budgets, and incident response procedures to maintain system "
    "availability and user trust.\n\n"
)


def build_synthetic_input(target_chars: int = 120_000) -> str:
    """
    Repeat a neutral technical passage until we reach approximately
    target_chars characters (~25 000–40 000 tokens at ~4 chars/token).
    """
    repeats = (target_chars // len(_PASSAGE)) + 1
    text = _PASSAGE * repeats
    return text[:target_chars]


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── confirmation guard ───────────────────────────────────────────────
    guard = _env("RUN_VERTEX_BILLING_TEST", "")
    if guard != "YES":
        print("=" * 70)
        print("VERTEX AI BILLING TEST — NOT EXECUTED")
        print("=" * 70)
        print()
        print("This script makes BILLABLE Vertex AI requests.")
        print("To confirm you want to proceed, set:")
        print()
        print("    RUN_VERTEX_BILLING_TEST=YES")
        print()
        print("The estimated cost for a default run is well below US$1.")
        sys.exit(0)

    # ── validate environment ─────────────────────────────────────────────
    credentials_path = _require_env("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = _require_env("GOOGLE_CLOUD_PROJECT")
    location = _env("GOOGLE_CLOUD_LOCATION", "global")
    model = _env("GEMINI_MODEL", "gemini-3.1-flash-lite")

    input_price = float(_env("VERTEX_INPUT_PRICE_PER_MILLION", "1.50"))
    output_price = float(_env("VERTEX_OUTPUT_PRICE_PER_MILLION", "9.00"))
    num_requests = 3

    print("=" * 70)
    print("VERTEX AI BILLING VERIFICATION")
    print("=" * 70)
    print(f"  Project        : {project_id}")
    print(f"  Location       : {location}")
    print(f"  Model          : {model}")
    print(f"  Credentials    : {'configured' if credentials_path else 'not set'}")
    print(f"  Requests       : {num_requests}")
    print(f"  Input $/1M     : ${input_price:.2f}")
    print(f"  Output $/1M    : ${output_price:.2f}")
    print()

    # ── build synthetic input ────────────────────────────────────────────
    synthetic_text = build_synthetic_input()
    prompt = (
        "You are a senior technical writer. Read the following reference "
        "material and produce a structured summary with these sections: "
        "1) Overview (3 sentences), 2) Key Practices (bullet list of 8 items "
        "with one-sentence explanations), 3) Architecture Patterns (bullet "
        "list of 6 items), 4) Reliability and Security considerations "
        "(4 paragraphs), 5) Conclusion (2 sentences). Be detailed and use "
        "professional technical language.\n\n"
        "REFERENCE MATERIAL:\n"
        f"{synthetic_text}"
    )
    print(f"  Synthetic input: {len(synthetic_text):,} chars")
    print()

    # ── initialise client ────────────────────────────────────────────────
    try:
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            "AI_Evaluator_Backend.settings",
        )
        import django
        django.setup()

        from AI_Evaluator_Backend.llm import get_llm
        from google.genai import types
    except ImportError:
        print("ERROR: google-genai is not installed. Run: pip install google-genai")
        sys.exit(1)

    try:
        client = get_llm()
        print("  Client         : initialised ✓")
    except Exception as exc:
        print(f"  Client         : FAILED — {type(exc).__name__}: {exc}")
        sys.exit(1)

    # ── count tokens ─────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("TOKEN COUNT (pre-flight)")
    print("─" * 70)
    try:
        count_result = client.models.count_tokens(
            model=model,
            contents=prompt,
        )
        estimated_input_tokens = count_result.total_tokens
        print(f"  Estimated input tokens : {estimated_input_tokens:,}")
    except Exception as exc:
        print(f"  count_tokens failed    : {type(exc).__name__}: {exc}")
        estimated_input_tokens = None

    # ── generate content ─────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("REQUESTS")
    print("─" * 70)

    cumulative_input = 0
    cumulative_output = 0
    cumulative_total = 0

    gen_config = types.GenerateContentConfig(
        max_output_tokens=1500,
    )

    for i in range(1, num_requests + 1):
        if i > 1:
            print(f"\n  ⏳ Waiting 5 seconds before request {i}...")
            time.sleep(5)

        print(f"\n  ── Request {i}/{num_requests} ──")

        try:
            t0 = time.time()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=gen_config,
            )
            latency_ms = int((time.time() - t0) * 1000)

            usage = response.usage_metadata
            prompt_tokens = usage.prompt_token_count or 0
            candidates_tokens = usage.candidates_token_count or 0
            total_tokens = usage.total_token_count or 0
            cached_tokens = getattr(usage, "cached_content_token_count", None)

            cumulative_input += prompt_tokens
            cumulative_output += candidates_tokens
            cumulative_total += total_tokens

            text = (response.text or "").strip()
            preview = text[:200] + ("…" if len(text) > 200 else "")

            print(f"  Status               : ✅ success ({latency_ms:,} ms)")
            print(f"  Model                : {model}")
            print(f"  prompt_token_count   : {prompt_tokens:,}")
            print(f"  candidates_token_count: {candidates_tokens:,}")
            print(f"  total_token_count    : {total_tokens:,}")
            if cached_tokens is not None:
                print(f"  cached_content_tokens: {cached_tokens:,}")
            print(f"  Response preview     : {preview}")

        except Exception as exc:
            err_text = str(exc).lower()
            err_type = type(exc).__name__

            if "401" in err_text or "403" in err_text:
                category = "Authentication / IAM / API not enabled"
            elif "404" in err_text:
                category = "Model or location not found"
            elif "429" in err_text or "quota" in err_text or "resource_exhausted" in err_text:
                category = "Quota / rate limit / capacity"
            else:
                category = "Unexpected error"

            safe_msg = str(exc)[:500]
            print(f"  Status               : ❌ FAILED")
            print(f"  Category             : {category}")
            print(f"  Error                : {err_type}: {safe_msg}")
            print()
            print("  Stopping — will not retry failed requests.")
            break

    # ── summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("CUMULATIVE TOKEN USAGE")
    print("=" * 70)
    print(f"  Input  tokens : {cumulative_input:,}")
    print(f"  Output tokens : {cumulative_output:,}")
    print(f"  Total  tokens : {cumulative_total:,}")

    # ── cost estimate ────────────────────────────────────────────────────
    input_cost = cumulative_input / 1_000_000 * input_price
    output_cost = cumulative_output / 1_000_000 * output_price
    estimated_cost = input_cost + output_cost

    print()
    print("=" * 70)
    print("ESTIMATED COST (Standard PayGo — NOT an official invoice)")
    print("=" * 70)
    print(f"  Input  : {cumulative_input:,} tokens × ${input_price:.2f}/1M = ${input_cost:.6f}")
    print(f"  Output : {cumulative_output:,} tokens × ${output_price:.2f}/1M = ${output_cost:.6f}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  Total estimated cost : ${estimated_cost:.6f}")
    print()
    print("  ⚠  This is an ESTIMATE based on list prices you provided.")
    print("     Actual charges depend on your billing agreement, committed-")
    print("     use discounts, and the pricing tier of your project.")
    print("     Check the Google Cloud Billing console for official costs.")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
