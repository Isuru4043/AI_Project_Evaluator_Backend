"""Run the isolated multi-provider VivaSense model benchmark.

This command is intentionally dry-run unless ``--live`` is supplied.
"""

import os
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from viva_evaluator.services.model_benchmark.registry import (
    BenchmarkConfigurationError,
    configured_key,
    default_registry_path,
    load_cases,
    load_model_registry,
)
from viva_evaluator.services.model_benchmark.runner import BenchmarkRunner, BudgetLedger


class Command(BaseCommand):
    help = "Validate or run the isolated multi-provider VivaSense benchmark."

    def add_arguments(self, parser):
        parser.add_argument("--registry", default=str(default_registry_path()))
        parser.add_argument("--dataset")
        parser.add_argument("--models", default="")
        parser.add_argument("--output", default="benchmarks/results/model_outputs.jsonl")
        parser.add_argument("--max-cases", type=int, default=0)
        parser.add_argument("--timeout", type=float, default=60.0)
        parser.add_argument("--max-retries", type=int, default=2)
        parser.add_argument(
            "--paid-cap-usd",
            type=float,
            default=float(os.getenv("BENCHMARK_PAID_SPEND_CAP_USD", "25")),
        )
        parser.add_argument("--list-models", action="store_true")
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--allow-paid", action="store_true")

    def handle(self, *args, **options):
        try:
            models = load_model_registry(options["registry"])
        except BenchmarkConfigurationError as exc:
            raise CommandError(str(exc)) from exc

        requested = {item.strip() for item in options["models"].split(",") if item.strip()}
        if requested:
            known = {item.id for item in models}
            unknown = requested - known
            if unknown:
                raise CommandError(f"Unknown model IDs: {', '.join(sorted(unknown))}")
            models = [item for item in models if item.id in requested]
        else:
            models = [item for item in models if item.enabled]

        if options["list_models"]:
            self.stdout.write("ID | provider | model | key | billing | configured")
            for spec in models:
                self.stdout.write(
                    f"{spec.id} | {spec.provider} | {spec.model} | "
                    f"{spec.api_key_env} | {spec.billing_mode} | "
                    f"{'yes' if configured_key(spec) else 'NO'}"
                )
            return

        dataset = options.get("dataset")
        if not dataset:
            raise CommandError("--dataset is required unless --list-models is used.")
        try:
            cases = load_cases(dataset)
        except (OSError, BenchmarkConfigurationError) as exc:
            raise CommandError(str(exc)) from exc
        if options["max_cases"] > 0:
            cases = cases[: options["max_cases"]]

        compatible_pairs = [
            (spec, case)
            for case in cases
            for spec in models
            if set(case.required_capabilities).issubset(spec.capabilities)
        ]
        excluded_pairs = len(models) * len(cases) - len(compatible_pairs)
        self.stdout.write(
            f"Validated {len(models)} model(s) and {len(cases)} case(s). "
            f"Planned compatible calls: {len(compatible_pairs)}."
        )
        if excluded_pairs:
            self.stdout.write(self.style.WARNING(
                f"Skipped {excluded_pairs} incompatible model/case pair(s) based on capabilities."
            ))
        missing = [spec.api_key_env for spec in models if not configured_key(spec)]
        if missing:
            self.stdout.write(self.style.WARNING(
                "Missing key variables: " + ", ".join(sorted(set(missing)))
            ))
        if not options["live"]:
            self.stdout.write(self.style.SUCCESS(
                "Dry run complete. No API requests were sent. Add --live to execute."
            ))
            return

        output = Path(options["output"])
        runner = BenchmarkRunner(
            output_path=output,
            budget=BudgetLedger(paid_cap_usd=max(0.0, options["paid_cap_usd"])),
            timeout_seconds=max(1.0, options["timeout"]),
            max_retries=max(0, options["max_retries"]),
        )
        try:
            results = runner.run(
                models=models,
                cases=cases,
                allow_paid=options["allow_paid"],
            )
        except RuntimeError as exc:
            raise CommandError(str(exc)) from exc

        successes = sum(item.status == "success" for item in results)
        failures = len(results) - successes
        self.stdout.write(self.style.SUCCESS(
            f"Finished: {successes} succeeded, {failures} failed. Results: {output.resolve()}"
        ))
