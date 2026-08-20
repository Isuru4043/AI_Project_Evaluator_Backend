"""
C1 — Question-quality metrics.

Aggregates the diagnostics that `generate_anchored_question` (and the ablation
runner) already emit per question, into dissertation-ready rates.

A "result" dict is expected to look like the Questioner output:
    {
        'question_text':  str,
        'tier1_passed':   bool,
        'tier1_failures': [str, ...],
        'critic_ran':     bool,
        'critic_passed':  bool | None,
        'critic_scores':  {'specificity': float, 'bloom_alignment': float,
                           'hallucination': bool},
        'attempts':       int,
        'blooms_level':   str,
        'latency_ms':     int,            # optional (ablation runner adds it)
    }
"""

from collections import Counter
from typing import Any, Dict, List

# Spoken-length window (mirrors tier1_validator MIN_WORDS / MAX_WORDS).
SPOKEN_MIN_WORDS = 12
SPOKEN_MAX_WORDS = 60


def _word_count(text: str) -> int:
    return len((text or '').split())


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(values: List[float], percentile: float) -> float:
    """Nearest-rank percentile without a NumPy dependency."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    rank = max(1, int((percentile * len(ordered)) + 0.999999))
    return ordered[min(rank - 1, len(ordered) - 1)]


def persisted_audit_to_result(extension) -> Dict:
    """Flatten one persisted extension into the standard metrics input shape."""
    audit = dict(extension.generation_audit or {})
    tier1 = audit.get('tier1') or {}
    critic = audit.get('critic') or {}
    validation = audit.get('validation') or {}
    tts = audit.get('tts') or {}
    question = extension.question
    return {
        'question_id': str(question.id),
        'question_text': question.question_text,
        'blooms_level': audit.get('target_bloom') or question.blooms_level,
        'candidate_hash': audit.get('candidate_hash', ''),
        'tier1_passed': tier1.get('passed', False),
        'tier1_failures': tier1.get('failures', []),
        'critic_ran': critic.get('ran', False),
        'critic_passed': critic.get('passed'),
        'critic_scores': critic.get('scores', {}),
        'attempts': audit.get('attempts', 0),
        'source_reference_ids': audit.get('source_reference_ids', []),
        'validation_status': (
            extension.validation_status
            or validation.get('status')
            or 'not_applicable'
        ),
        'validation_degraded': extension.validation_degraded,
        'fallback_used': extension.fallback_used,
        'degradation_reason': validation.get('degradation_reason', ''),
        'tts': dict(tts) if isinstance(tts, dict) else {},
    }


def compute_question_metrics(results: List[Dict]) -> Dict:
    """
    Aggregate a batch of question-generation results into quality rates.

    Returns a dict of metrics. All rates are in [0, 1]. Empty input yields
    a zeroed report with n=0.
    """
    n = len(results)
    if n == 0:
        return {'n': 0}

    tier1_passes = 0
    anchored = 0
    doc_location_violations = 0
    compound = 0
    too_long = 0
    too_short = 0

    critic_ran = 0
    critic_passes = 0
    hallucinations = 0
    spec_scores: List[float] = []
    bloom_scores: List[float] = []

    spoken_ok = 0
    attempts_list: List[int] = []
    latency_list: List[float] = []
    bloom_counter: Counter = Counter()
    validation_status_counter: Counter = Counter()
    degraded_validations = 0
    fallback_questions = 0
    critic_unavailable = 0
    source_attributed = 0

    for r in results:
        failures = r.get('tier1_failures', []) or []
        failure_str = ' '.join(failures)

        if r.get('tier1_passed'):
            tier1_passes += 1
        if 'missing_anchor' not in failure_str:
            anchored += 1
        if 'document_location_reference' in failure_str:
            doc_location_violations += 1
        if 'compound_question' in failure_str:
            compound += 1
        if 'too_long' in failure_str:
            too_long += 1
        if 'too_short' in failure_str:
            too_short += 1

        if r.get('critic_ran'):
            critic_ran += 1
            if r.get('critic_passed'):
                critic_passes += 1
        cs = r.get('critic_scores') or {}
        if 'specificity' in cs:
            spec_scores.append(float(cs.get('specificity', 0.0)))
        if 'bloom_alignment' in cs:
            bloom_scores.append(float(cs.get('bloom_alignment', 0.0)))
        if cs.get('hallucination'):
            hallucinations += 1

        wc = _word_count(r.get('question_text', ''))
        if SPOKEN_MIN_WORDS <= wc <= SPOKEN_MAX_WORDS:
            spoken_ok += 1

        attempts_list.append(int(r.get('attempts', 1)))
        if r.get('latency_ms') is not None:
            latency_list.append(float(r['latency_ms']))
        if r.get('blooms_level'):
            bloom_counter[r['blooms_level']] += 1
        validation_status = r.get('validation_status') or 'legacy_unknown'
        validation_status_counter[validation_status] += 1
        if r.get('validation_degraded'):
            degraded_validations += 1
        if r.get('fallback_used'):
            fallback_questions += 1
        if validation_status == 'critic_unavailable':
            critic_unavailable += 1
        if r.get('source_reference_ids'):
            source_attributed += 1

    return {
        'n':                      n,
        'tier1_pass_rate':        round(tier1_passes / n, 4),
        'anchoring_rate':         round(anchored / n, 4),
        'doc_location_violation_rate': round(doc_location_violations / n, 4),
        'compound_rate':          round(compound / n, 4),
        'too_long_rate':          round(too_long / n, 4),
        'too_short_rate':         round(too_short / n, 4),
        'spoken_length_ok_rate':  round(spoken_ok / n, 4),
        'critic_ran':             critic_ran,
        'critic_pass_rate':       round(critic_passes / critic_ran, 4) if critic_ran else None,
        'hallucination_rate':     round(hallucinations / n, 4),
        'mean_specificity':       round(_mean(spec_scores), 4) if spec_scores else None,
        'mean_bloom_alignment':   round(_mean(bloom_scores), 4) if bloom_scores else None,
        'mean_attempts':          round(_mean(attempts_list), 3),
        'mean_latency_ms':        round(_mean(latency_list), 1) if latency_list else None,
        'bloom_distribution':     dict(bloom_counter),
        'validation_status_distribution': dict(validation_status_counter),
        'fully_validated_rate':   round(
            validation_status_counter['fully_validated'] / n,
            4,
        ),
        'degraded_validation_rate': round(degraded_validations / n, 4),
        'fallback_rate':          round(fallback_questions / n, 4),
        'critic_unavailable_rate': round(critic_unavailable / n, 4),
        'source_attribution_rate': round(source_attributed / n, 4),
    }


def format_metrics_table(metrics_by_condition: Dict[str, Dict]) -> str:
    """
    Render one or more metric sets side by side as a text table.

    Args:
        metrics_by_condition: {condition_label: metrics_dict}
    """
    if not metrics_by_condition:
        return '(no metrics)'

    rows = [
        ('n',                      lambda m: m.get('n', 0)),
        ('anchoring_rate',         lambda m: m.get('anchoring_rate')),
        ('tier1_pass_rate',        lambda m: m.get('tier1_pass_rate')),
        ('critic_pass_rate',       lambda m: m.get('critic_pass_rate')),
        ('fully_validated_rate',   lambda m: m.get('fully_validated_rate')),
        ('degraded_validation_rate', lambda m: m.get('degraded_validation_rate')),
        ('fallback_rate',          lambda m: m.get('fallback_rate')),
        ('critic_unavailable_rate', lambda m: m.get('critic_unavailable_rate')),
        ('source_attribution_rate', lambda m: m.get('source_attribution_rate')),
        ('hallucination_rate',     lambda m: m.get('hallucination_rate')),
        ('mean_specificity',       lambda m: m.get('mean_specificity')),
        ('mean_bloom_alignment',   lambda m: m.get('mean_bloom_alignment')),
        ('spoken_length_ok_rate',  lambda m: m.get('spoken_length_ok_rate')),
        ('doc_location_violation_rate', lambda m: m.get('doc_location_violation_rate')),
        ('mean_attempts',          lambda m: m.get('mean_attempts')),
        ('mean_latency_ms',        lambda m: m.get('mean_latency_ms')),
    ]

    conditions = list(metrics_by_condition.keys())
    col_w = max(22, *(len(c) for c in conditions)) + 2
    label_w = 28

    def fmt(v):
        if v is None:
            return '—'
        if isinstance(v, float):
            return f'{v:.3f}'
        return str(v)

    lines = []
    header = 'metric'.ljust(label_w) + ''.join(c.ljust(col_w) for c in conditions)
    lines.append(header)
    lines.append('-' * len(header))
    for name, getter in rows:
        line = name.ljust(label_w)
        for c in conditions:
            line += fmt(getter(metrics_by_condition[c])).ljust(col_w)
        lines.append(line)
    return '\n'.join(lines)


def compute_llm_telemetry_metrics(summaries: List[Dict[str, Any]]) -> Dict:
    """Aggregate persisted per-turn LLM summaries, deduplicated by trace ID."""
    unique = {}
    anonymous = []
    for summary in summaries:
        if not isinstance(summary, dict) or not summary:
            continue
        trace_id = str(summary.get('trace_id') or '')
        if trace_id:
            unique.setdefault(trace_id, summary)
        else:
            anonymous.append(summary)
    traces = list(unique.values()) + anonymous
    if not traces:
        return {'turn_count': 0}

    duration_values = [float(item.get('duration_ms', 0)) for item in traces]
    call_values = [float(item.get('call_count', 0)) for item in traces]
    token_values = [float(item.get('total_tokens', 0)) for item in traces]
    operation_counts: Counter = Counter()
    operation_tokens: Counter = Counter()
    operation_latencies: Dict[str, List[float]] = {}
    operation_costs: Counter = Counter()
    model_routes: Counter = Counter()
    trace_kind_durations: Dict[str, List[float]] = {}
    total_calls = 0
    token_usage_calls = 0
    costed_calls = 0

    for trace in traces:
        trace_kind = str(trace.get('trace_kind') or 'unspecified')
        trace_kind_durations.setdefault(trace_kind, []).append(
            float(trace.get('duration_ms', 0))
        )
        total_calls += int(trace.get('call_count', 0))
        token_usage_calls += int(trace.get('token_usage_call_count', 0))
        costed_calls += int(trace.get('costed_call_count', 0))
        for call in trace.get('calls') or []:
            operation = str(call.get('operation') or 'unspecified')
            operation_counts[operation] += 1
            operation_latencies.setdefault(operation, []).append(
                float(call.get('latency_ms', 0))
            )
            model_routes[str(call.get('model_route') or 'unspecified')] += 1
            if call.get('total_tokens') is not None:
                operation_tokens[operation] += int(call['total_tokens'])
            if call.get('estimated_cost_usd') is not None:
                operation_costs[operation] += float(
                    call['estimated_cost_usd']
                )

    total_estimated_cost = sum(
        float(item.get('estimated_cost_usd', 0)) for item in traces
    )
    operation_metrics = {}
    for operation, count in operation_counts.items():
        latencies = operation_latencies.get(operation, [])
        operation_metrics[operation] = {
            'call_count': count,
            'total_tokens': int(operation_tokens.get(operation, 0)),
            'p50_latency_ms': round(_percentile(latencies, 0.50), 1),
            'p95_latency_ms': round(_percentile(latencies, 0.95), 1),
            'estimated_cost_usd': round(
                float(operation_costs.get(operation, 0)),
                8,
            ),
        }

    trace_kind_metrics = {
        trace_kind: {
            'turn_count': len(values),
            'p50_latency_ms': round(_percentile(values, 0.50), 1),
            'p95_latency_ms': round(_percentile(values, 0.95), 1),
        }
        for trace_kind, values in trace_kind_durations.items()
    }

    return {
        'turn_count': len(traces),
        'llm_call_count': total_calls,
        'mean_calls_per_turn': round(_mean(call_values), 3),
        'p50_calls_per_turn': round(_percentile(call_values, 0.50), 3),
        'p95_calls_per_turn': round(_percentile(call_values, 0.95), 3),
        'p50_turn_latency_ms': round(_percentile(duration_values, 0.50), 1),
        'p95_turn_latency_ms': round(_percentile(duration_values, 0.95), 1),
        'provider_latency_ms': sum(
            int(item.get('provider_latency_ms', 0)) for item in traces
        ),
        'retry_count': sum(int(item.get('retry_count', 0)) for item in traces),
        'fallback_call_count': sum(
            int(item.get('fallback_call_count', 0)) for item in traces
        ),
        'input_characters': sum(
            int(item.get('input_characters', 0)) for item in traces
        ),
        'prompt_original_characters': sum(
            int(item.get('prompt_original_characters', 0)) for item in traces
        ),
        'prompt_sent_characters': sum(
            int(item.get('prompt_sent_characters', 0)) for item in traces
        ),
        'prompt_truncated_call_count': sum(
            int(item.get('prompt_truncated_call_count', 0))
            for item in traces
        ),
        'prompt_truncation_rate': round(
            sum(
                int(item.get('prompt_truncated_call_count', 0))
                for item in traces
            ) / total_calls,
            4,
        ) if total_calls else 0.0,
        'output_characters': sum(
            int(item.get('output_characters', 0)) for item in traces
        ),
        'input_tokens': sum(int(item.get('input_tokens', 0)) for item in traces),
        'output_tokens': sum(int(item.get('output_tokens', 0)) for item in traces),
        'total_tokens': sum(int(item.get('total_tokens', 0)) for item in traces),
        'mean_tokens_per_turn': round(_mean(token_values), 1),
        'p50_tokens_per_turn': round(_percentile(token_values, 0.50), 1),
        'p95_tokens_per_turn': round(_percentile(token_values, 0.95), 1),
        'token_usage_coverage': round(
            token_usage_calls / total_calls,
            4,
        ) if total_calls else 1.0,
        'estimated_cost_usd': round(total_estimated_cost, 8),
        'mean_estimated_cost_per_turn_usd': round(
            total_estimated_cost / len(traces),
            8,
        ),
        'cost_estimate_coverage': round(
            costed_calls / total_calls,
            4,
        ) if total_calls else 1.0,
        'operation_call_distribution': dict(operation_counts),
        'operation_token_distribution': dict(operation_tokens),
        'operation_metrics': operation_metrics,
        'model_route_distribution': dict(model_routes),
        'trace_kind_metrics': trace_kind_metrics,
    }


def compute_tts_metrics(
    results: List[Dict],
    *,
    price_per_1000_characters_usd: float = 0.0,
) -> Dict:
    """Aggregate persisted speculative-TTS outcomes and character cost."""
    records = [
        result.get('tts') or {}
        for result in results
        if isinstance(result.get('tts'), dict)
        and (result.get('tts') or {}).get('enabled') is True
    ]
    if not records:
        return {'enabled_question_count': 0}

    statuses: Counter = Counter(
        str(record.get('status') or 'unknown') for record in records
    )
    known_cache = [
        record for record in records if record.get('cache_hit') is not None
    ]
    latencies = [
        float(record['generation_latency_ms'])
        for record in records
        if record.get('generation_latency_ms') is not None
    ]
    generated_characters = sum(
        int(record.get('characters', 0) or 0)
        for record in records
        if record.get('cache_hit') is not True
        and record.get('status') == 'ready'
    )
    estimated_cost = (
        generated_characters * max(0.0, price_per_1000_characters_usd)
    ) / 1000
    return {
        'enabled_question_count': len(records),
        'status_distribution': dict(statuses),
        'ready_rate': round(statuses['ready'] / len(records), 4),
        'failed_rate': round(statuses['failed'] / len(records), 4),
        'pending_rate': round(statuses['pending'] / len(records), 4),
        'cache_hit_rate': round(
            sum(record.get('cache_hit') is True for record in known_cache)
            / len(known_cache),
            4,
        ) if known_cache else None,
        'speculative_waste_count': sum(
            record.get('speculative_wasted') is True for record in records
        ),
        'generated_characters': generated_characters,
        'p50_generation_latency_ms': (
            round(_percentile(latencies, 0.50), 1) if latencies else None
        ),
        'p95_generation_latency_ms': (
            round(_percentile(latencies, 0.95), 1) if latencies else None
        ),
        'estimated_cost_usd': round(estimated_cost, 8),
        'price_per_1000_characters_usd': max(
            0.0,
            price_per_1000_characters_usd,
        ),
    }


def format_llm_telemetry_report(metrics: Dict) -> str:
    if not metrics or metrics.get('turn_count', 0) == 0:
        return 'LLM telemetry: no persisted traces'
    rows = [
        ('turn_count', metrics.get('turn_count')),
        ('llm_call_count', metrics.get('llm_call_count')),
        ('mean_calls_per_turn', metrics.get('mean_calls_per_turn')),
        ('p50_calls_per_turn', metrics.get('p50_calls_per_turn')),
        ('p95_calls_per_turn', metrics.get('p95_calls_per_turn')),
        ('p50_turn_latency_ms', metrics.get('p50_turn_latency_ms')),
        ('p95_turn_latency_ms', metrics.get('p95_turn_latency_ms')),
        ('retry_count', metrics.get('retry_count')),
        ('fallback_call_count', metrics.get('fallback_call_count')),
        ('input_characters', metrics.get('input_characters')),
        ('output_characters', metrics.get('output_characters')),
        ('total_tokens', metrics.get('total_tokens')),
        ('mean_tokens_per_turn', metrics.get('mean_tokens_per_turn')),
        ('p50_tokens_per_turn', metrics.get('p50_tokens_per_turn')),
        ('p95_tokens_per_turn', metrics.get('p95_tokens_per_turn')),
        ('token_usage_coverage', metrics.get('token_usage_coverage')),
        ('estimated_cost_usd', metrics.get('estimated_cost_usd')),
        ('mean_cost_per_turn_usd', metrics.get('mean_estimated_cost_per_turn_usd')),
        ('cost_estimate_coverage', metrics.get('cost_estimate_coverage')),
        ('prompt_truncation_rate', metrics.get('prompt_truncation_rate')),
    ]
    width = max(len(label) for label, _value in rows)
    lines = ['LLM telemetry']
    lines.extend(f'{label.ljust(width)}  {value}' for label, value in rows)
    return '\n'.join(lines)
