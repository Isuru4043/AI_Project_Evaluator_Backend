from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from viva_evaluator.services.model_benchmark.code_dataset import (
    CodeDatasetError,
    build_code_dataset,
)


class Command(BaseCommand):
    help = "Build the frozen VivaSense code-understanding benchmark dataset."

    def add_arguments(self, parser):
        parser.add_argument(
            "--spec",
            default="benchmarks/datasets/code_understanding/spec.json",
        )
        parser.add_argument(
            "--output-root",
            default="benchmarks/datasets/code_understanding",
        )
        parser.add_argument("--backend-root", default=".")
        parser.add_argument("--frontend-root", required=True)

    def handle(self, *args, **options):
        try:
            manifest = build_code_dataset(
                spec_path=options["spec"],
                output_root=options["output_root"],
                repository_roots={
                    "backend": Path(options["backend_root"]),
                    "frontend": Path(options["frontend_root"]),
                },
            )
        except (OSError, ValueError, CodeDatasetError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            f"Built {manifest['case_count']} frozen code-understanding cases."
        ))
