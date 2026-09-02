import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from viva_evaluator.services.model_benchmark.scoring import score_result_record


class Command(BaseCommand):
    help = "Apply deterministic benchmark scorers to saved model outputs."

    def add_arguments(self, parser):
        parser.add_argument("--results", required=True)
        parser.add_argument("--dataset-root", default="benchmarks/datasets")
        parser.add_argument("--output", required=True)

    def handle(self, *args, **options):
        results_path = Path(options["results"])
        output_path = Path(options["output"])
        dataset_root = Path(options["dataset_root"])
        if not results_path.is_file():
            raise CommandError(f"Results file not found: {results_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scored = []
        with results_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    scored.append(score_result_record(result, dataset_root=dataset_root))
                except (json.JSONDecodeError, OSError, KeyError, TypeError) as exc:
                    raise CommandError(
                        f"Unable to score {results_path}:{line_number}: {exc}"
                    ) from exc

        with output_path.open("w", encoding="utf-8") as handle:
            for item in scored:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        count = sum(
            str(item.get("score_status", "")).startswith("scored")
            for item in scored
        )
        self.stdout.write(self.style.SUCCESS(
            f"Scored {count} of {len(scored)} records. Output: {output_path.resolve()}"
        ))
