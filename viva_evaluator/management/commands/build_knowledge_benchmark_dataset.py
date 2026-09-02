from django.core.management.base import BaseCommand, CommandError

from viva_evaluator.services.model_benchmark.knowledge_dataset import (
    build_knowledge_dataset,
)


class Command(BaseCommand):
    help = "Build the frozen VivaSense knowledge-preparation benchmark dataset."

    def add_arguments(self, parser):
        parser.add_argument(
            "--spec",
            default="benchmarks/datasets/knowledge_preparation/spec.json",
        )
        parser.add_argument(
            "--output",
            default="benchmarks/datasets/knowledge_preparation",
        )

    def handle(self, *args, **options):
        try:
            manifest = build_knowledge_dataset(options["spec"], options["output"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(
            f"Built {manifest['pilot_case_count']} knowledge-preparation cases."
        ))
