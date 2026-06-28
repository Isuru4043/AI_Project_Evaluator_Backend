"""
C2 calibration report — how well do AI scores agree with examiner scores?

Usage:
    python manage.py eval_calibration
    python manage.py eval_calibration --project <project_uuid>
    python manage.py eval_calibration --out calibration.json
"""

import json

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Compare AI recommended scores against examiner final scores (C2).'

    def add_arguments(self, parser):
        parser.add_argument('--project', type=str, default=None,
                            help='Restrict to one project UUID.')
        parser.add_argument('--out', type=str, default=None,
                            help='Optional path to write the JSON result.')

    def handle(self, *args, **options):
        from viva_evaluator.services.evaluation import (
            calibration_from_db, format_calibration_report,
        )

        result = calibration_from_db(project_id=options.get('project'))
        self.stdout.write(format_calibration_report(result))

        out = options.get('out')
        if out:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            self.stdout.write(self.style.SUCCESS(f'\nWrote JSON to {out}'))
