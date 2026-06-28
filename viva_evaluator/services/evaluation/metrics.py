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
from typing import Dict, List

# Spoken-length window (mirrors tier1_validator MIN_WORDS / MAX_WORDS).
SPOKEN_MIN_WORDS = 12
SPOKEN_MAX_WORDS = 60


def _word_count(text: str) -> int:
    return len((text or '').split())


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


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
