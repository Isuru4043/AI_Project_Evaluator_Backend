"""Report validation health from persisted viva-question audit records."""

import json

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = (
        "Report persisted question-validation, degradation, fallback, and "
        "source-attribution rates plus LLM turn latency/cost telemetry without "
        "making LLM calls."
    )

    def add_arguments(self, parser):
        parser.add_argument('--session', type=str)
        parser.add_argument('--project', type=str)
        parser.add_argument('--out', type=str)
        parser.add_argument(
            '--baseline',
            type=str,
            help='Prior JSON report; prints selected current-minus-baseline deltas.',
        )
        parser.add_argument(
            '--enforce',
            action='store_true',
            help='Exit non-zero when a configured performance gate fails.',
        )
        parser.add_argument(
            '--min-turns',
            type=int,
            default=settings.VIVA_PERF_MIN_TURNS,
        )
        parser.add_argument(
            '--max-p95-latency-ms',
            type=float,
            default=settings.VIVA_PERF_MAX_P95_TURN_LATENCY_MS,
        )
        parser.add_argument(
            '--max-mean-calls-per-turn',
            type=float,
            default=settings.VIVA_PERF_MAX_MEAN_CALLS_PER_TURN,
        )
        parser.add_argument(
            '--max-degraded-rate',
            type=float,
            default=settings.VIVA_PERF_MAX_DEGRADED_RATE,
        )
        parser.add_argument(
            '--max-fallback-rate',
            type=float,
            default=settings.VIVA_PERF_MAX_FALLBACK_RATE,
        )
        parser.add_argument(
            '--min-tier1-pass-rate',
            type=float,
            default=settings.VIVA_PERF_MIN_TIER1_PASS_RATE,
        )
        parser.add_argument(
            '--max-mean-cost-per-turn-usd',
            type=float,
            default=settings.VIVA_PERF_MAX_MEAN_COST_PER_TURN_USD,
        )
        parser.add_argument(
            '--include-not-applicable',
            action='store_true',
            help='Include examiner and legacy questions without validation audits.',
        )

    def handle(self, *args, **options):
        from viva_evaluator.models import (
            VivaAnswerExtension,
            VivaQuestionExtension,
        )
        from viva_evaluator.services.evaluation import (
            compute_llm_telemetry_metrics,
            compute_question_metrics,
            compute_tts_metrics,
            format_llm_telemetry_report,
            format_metrics_table,
            persisted_audit_to_result,
        )

        queryset = VivaQuestionExtension.objects.select_related(
            'question__session'
        ).order_by('question__generated_at')
        if options.get('session'):
            queryset = queryset.filter(question__session_id=options['session'])
        if options.get('project'):
            queryset = queryset.filter(
                question__session__project_id=options['project']
            )
        if not options.get('include_not_applicable'):
            queryset = queryset.exclude(validation_status='not_applicable')

        results = [persisted_audit_to_result(item) for item in queryset]
        metrics = compute_question_metrics(results)
        self.stdout.write(format_metrics_table({'persisted': metrics}))
        self.stdout.write(
            '\nvalidation_status_distribution=' + json.dumps(
                metrics.get('validation_status_distribution', {}),
                sort_keys=True,
            )
        )

        telemetry_summaries = []
        for item in queryset:
            audit = item.generation_audit or {}
            if isinstance(audit, dict):
                telemetry_summaries.append(
                    audit.get('llm_telemetry') or {}
                )
        answer_queryset = VivaAnswerExtension.objects.select_related(
            'answer__question__session'
        ).order_by('answer__answered_at')
        if options.get('session'):
            answer_queryset = answer_queryset.filter(
                answer__question__session_id=options['session']
            )
        if options.get('project'):
            answer_queryset = answer_queryset.filter(
                answer__question__session__project_id=options['project']
            )
        for item in answer_queryset:
            analysis = item.detailed_ai_analysis or {}
            if isinstance(analysis, dict):
                telemetry_summaries.append(
                    analysis.get('llm_telemetry') or {}
                )
        telemetry_metrics = compute_llm_telemetry_metrics(
            telemetry_summaries
        )
        tts_metrics = compute_tts_metrics(
            results,
            price_per_1000_characters_usd=(
                settings.ELEVENLABS_PRICE_PER_1000_CHARACTERS_USD
            ),
        )
        self.stdout.write('\n' + format_llm_telemetry_report(telemetry_metrics))
        self.stdout.write(
            '\noperation_call_distribution=' + json.dumps(
                telemetry_metrics.get('operation_call_distribution', {}),
                sort_keys=True,
            )
        )
        self.stdout.write(
            '\noperation_metrics=' + json.dumps(
                telemetry_metrics.get('operation_metrics', {}),
                sort_keys=True,
            )
        )
        self.stdout.write(
            '\ntts_metrics=' + json.dumps(tts_metrics, sort_keys=True)
        )

        gates = _evaluate_gates(metrics, telemetry_metrics, options)
        self.stdout.write(
            '\nperformance_gates=' + json.dumps(gates, sort_keys=True)
        )

        baseline_delta = {}
        if options.get('baseline'):
            baseline_delta = _baseline_delta(
                options['baseline'],
                metrics,
                telemetry_metrics,
                tts_metrics,
            )
            self.stdout.write(
                '\nbaseline_delta=' + json.dumps(
                    baseline_delta,
                    sort_keys=True,
                )
            )

        if options.get('out'):
            with open(options['out'], 'w', encoding='utf-8') as output_file:
                json.dump(
                    {
                        'metrics': metrics,
                        'llm_telemetry_metrics': telemetry_metrics,
                        'tts_metrics': tts_metrics,
                        'performance_gates': gates,
                        'baseline_delta': baseline_delta,
                        'questions': results,
                    },
                    output_file,
                    indent=2,
                    default=str,
                )
            self.stdout.write(self.style.SUCCESS(f"Wrote {options['out']}"))

        if options.get('enforce') and gates['overall_status'] != 'passed':
            failed = ', '.join(
                name
                for name, result in gates['checks'].items()
                if result['status'] != 'passed'
            )
            raise CommandError(f'Performance gates failed: {failed}')


def _evaluate_gates(question_metrics, telemetry_metrics, options):
    checks = {}

    def maximum(name, actual, threshold, *, enabled=True):
        if not enabled:
            return
        checks[name] = {
            'actual': actual,
            'threshold': threshold,
            'comparison': '<=',
            'status': 'passed' if actual <= threshold else 'failed',
        }

    def minimum(name, actual, threshold):
        checks[name] = {
            'actual': actual,
            'threshold': threshold,
            'comparison': '>=',
            'status': 'passed' if actual >= threshold else 'failed',
        }

    minimum(
        'sample_size',
        int(telemetry_metrics.get('turn_count', 0)),
        max(0, int(options['min_turns'])),
    )
    maximum(
        'p95_turn_latency_ms',
        float(telemetry_metrics.get('p95_turn_latency_ms', 0)),
        float(options['max_p95_latency_ms']),
    )
    maximum(
        'mean_calls_per_turn',
        float(telemetry_metrics.get('mean_calls_per_turn', 0)),
        float(options['max_mean_calls_per_turn']),
    )
    maximum(
        'degraded_validation_rate',
        float(question_metrics.get('degraded_validation_rate', 0)),
        float(options['max_degraded_rate']),
    )
    maximum(
        'fallback_rate',
        float(question_metrics.get('fallback_rate', 0)),
        float(options['max_fallback_rate']),
    )
    minimum(
        'tier1_pass_rate',
        float(question_metrics.get('tier1_pass_rate', 0)),
        float(options['min_tier1_pass_rate']),
    )
    cost_threshold = float(options['max_mean_cost_per_turn_usd'])
    maximum(
        'mean_estimated_cost_per_turn_usd',
        float(telemetry_metrics.get('mean_estimated_cost_per_turn_usd', 0)),
        cost_threshold,
        enabled=cost_threshold > 0,
    )
    return {
        'overall_status': (
            'passed'
            if checks and all(item['status'] == 'passed' for item in checks.values())
            else 'failed'
        ),
        'checks': checks,
    }


def _baseline_delta(path, question_metrics, llm_metrics, tts_metrics):
    try:
        with open(path, 'r', encoding='utf-8') as baseline_file:
            baseline = json.load(baseline_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise CommandError(f'Could not read baseline report: {exc}') from exc

    comparisons = {
        'tier1_pass_rate': (
            question_metrics,
            baseline.get('metrics') or {},
        ),
        'fallback_rate': (
            question_metrics,
            baseline.get('metrics') or {},
        ),
        'p95_turn_latency_ms': (
            llm_metrics,
            baseline.get('llm_telemetry_metrics') or {},
        ),
        'mean_calls_per_turn': (
            llm_metrics,
            baseline.get('llm_telemetry_metrics') or {},
        ),
        'mean_estimated_cost_per_turn_usd': (
            llm_metrics,
            baseline.get('llm_telemetry_metrics') or {},
        ),
        'tts_ready_rate': (
            {'tts_ready_rate': tts_metrics.get('ready_rate', 0)},
            {
                'tts_ready_rate': (
                    baseline.get('tts_metrics') or {}
                ).get('ready_rate', 0),
            },
        ),
    }
    return {
        name: round(
            float(current.get(name, 0)) - float(previous.get(name, 0)),
            8,
        )
        for name, (current, previous) in comparisons.items()
    }
