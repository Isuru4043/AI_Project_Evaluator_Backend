"""
C1 question-quality evaluation — generates questions across a submission's
rubric (optionally under several ablation conditions) and prints aggregate
quality metrics.

Usage:
    python manage.py eval_questions --submission <uuid>
    python manage.py eval_questions --submission <uuid> --conditions full,no_anchoring,no_critic
    python manage.py eval_questions --submission <uuid> --per-criterion 2 --out q_metrics.json

NOTE: this calls the LLM for each question, so it consumes API quota.
"""

import json

from django.core.management.base import BaseCommand, CommandError


# condition name → ablation flag dict
CONDITION_FLAGS = {
    'full':         {},
    'no_anchoring': {'disable_anchoring': True},
    'no_critic':    {'disable_critic': True},
    'no_kg':        {'disable_kg': True},
    'no_tier1':     {'disable_tier1_validation': True},
    'no_section':   {'disable_section_aware': True},
}


class Command(BaseCommand):
    help = 'Generate questions across a rubric and report quality metrics (C1).'

    def add_arguments(self, parser):
        parser.add_argument('--submission', type=str, required=True,
                            help='ProjectSubmission UUID to evaluate against.')
        parser.add_argument('--conditions', type=str, default='full',
                            help='Comma list: ' + ','.join(CONDITION_FLAGS.keys()))
        parser.add_argument('--per-criterion', type=int, default=1,
                            help='Questions to generate per criterion (default 1).')
        parser.add_argument('--out', type=str, default=None,
                            help='Optional path to write the JSON results.')

    def handle(self, *args, **options):
        from core.models import ProjectSubmission
        from viva_evaluator.services.pipeline.turn_pipeline import load_rubric
        from viva_evaluator.services.ablation import run_single_ablation, AblationFlags
        from viva_evaluator.services.evaluation import (
            compute_question_metrics, format_metrics_table,
        )

        sub_id = options['submission']
        try:
            submission = ProjectSubmission.objects.get(id=sub_id)
        except ProjectSubmission.DoesNotExist:
            raise CommandError(f'No ProjectSubmission with id={sub_id}')

        conditions = [c.strip() for c in options['conditions'].split(',') if c.strip()]
        for c in conditions:
            if c not in CONDITION_FLAGS:
                raise CommandError(f'Unknown condition "{c}". '
                                   f'Valid: {", ".join(CONDITION_FLAGS)}')

        rubric = load_rubric(submission.project)
        if not rubric:
            raise CommandError('Project has no rubric criteria configured.')

        per_criterion = max(1, int(options['per_criterion']))
        self.stdout.write(
            f'Evaluating submission={sub_id} | criteria={len(rubric)} | '
            f'per_criterion={per_criterion} | conditions={conditions}\n'
        )

        raw_by_condition = {}
        metrics_by_condition = {}

        for cond in conditions:
            flags = AblationFlags.from_dict(CONDITION_FLAGS[cond])
            results = []
            for crit in rubric:
                for _ in range(per_criterion):
                    try:
                        run = run_single_ablation(
                            submission=submission,
                            criterion_name=crit['name'],
                            criterion_description=crit['description'],
                            flags=flags,
                            last_answer='',
                            difficulty='medium',
                        )
                        results.append(run)
                        self.stdout.write(
                            f"  [{cond}] {crit['name'][:28]:28s} "
                            f"t1={run['tier1_passed']} crit={run.get('critic_passed')} "
                            f":: {run['question_text'][:60]}"
                        )
                    except Exception as exc:
                        self.stderr.write(f"  [{cond}] {crit['name']}: ERROR {exc}")
            raw_by_condition[cond] = results
            metrics_by_condition[cond] = compute_question_metrics(results)

        self.stdout.write('\n' + format_metrics_table(metrics_by_condition))

        out = options.get('out')
        if out:
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(
                    {'metrics': metrics_by_condition, 'raw': raw_by_condition},
                    f, indent=2, default=str,
                )
            self.stdout.write(self.style.SUCCESS(f'\nWrote JSON to {out}'))
