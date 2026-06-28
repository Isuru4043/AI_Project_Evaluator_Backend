"""
C2 — Score calibration.

Compares the AI's recommended scores against examiner-confirmed final scores
(both stored on core.FinalScore) to quantify how well the system agrees with
human experts.

Pure math over existing data — no LLM, runs offline.
"""

import math
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Core statistics (pure functions over (ai, examiner) pairs)
# =============================================================================

def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def _rankdata(values: List[float]) -> List[float]:
    """Average-rank transform (ties share the mean rank)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0   # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    return _pearson(_rankdata(xs), _rankdata(ys))


def compute_calibration(pairs: List[Tuple[float, float]]) -> Dict:
    """
    Compute calibration stats from (ai_score, examiner_score) pairs.

    Returns:
        {
            'n', 'pearson', 'spearman', 'mae', 'rmse',
            'within_1_rate', 'within_0_5_rate',
            'mean_ai', 'mean_examiner', 'mean_bias',
        }
    """
    pairs = [(float(a), float(e)) for a, e in pairs if a is not None and e is not None]
    n = len(pairs)
    if n == 0:
        return {'n': 0}

    ai = [a for a, _ in pairs]
    ex = [e for _, e in pairs]
    errors = [a - e for a, e in pairs]
    abs_err = [abs(d) for d in errors]

    return {
        'n':                n,
        'pearson':          _round(_pearson(ai, ex)),
        'spearman':         _round(_spearman(ai, ex)),
        'mae':              round(sum(abs_err) / n, 4),
        'rmse':             round(math.sqrt(sum(d * d for d in errors) / n), 4),
        'within_1_rate':    round(sum(1 for d in abs_err if d <= 1.0) / n, 4),
        'within_0_5_rate':  round(sum(1 for d in abs_err if d <= 0.5) / n, 4),
        'mean_ai':          round(sum(ai) / n, 4),
        'mean_examiner':    round(sum(ex) / n, 4),
        'mean_bias':        round(sum(errors) / n, 4),   # +ve = AI scores high
    }


def _round(v: Optional[float]) -> Optional[float]:
    return round(v, 4) if v is not None else None


# =============================================================================
# DB pull — overall + per-criterion calibration from core.FinalScore
# =============================================================================

def calibration_from_db(project_id: Optional[str] = None) -> Dict:
    """
    Pull (ai_recommended_score, examiner_final_score) pairs from FinalScore and
    compute overall + per-criterion calibration.

    Args:
        project_id: optional — restrict to one project's sessions.

    Returns:
        {
            'overall':       {calibration stats},
            'per_criterion': {criterion_name: {calibration stats}},
        }
    """
    from core.models import FinalScore

    qs = FinalScore.objects.filter(ai_recommended_score__isnull=False)
    if project_id:
        qs = qs.filter(session__project_id=project_id)

    qs = qs.select_related('criteria')

    overall_pairs: List[Tuple[float, float]] = []
    by_criterion: Dict[str, List[Tuple[float, float]]] = {}

    for fs in qs:
        ai = fs.ai_recommended_score
        ex = fs.examiner_final_score
        if ai is None or ex is None:
            continue
        pair = (float(ai), float(ex))
        overall_pairs.append(pair)
        cname = getattr(fs.criteria, 'criteria_name', str(fs.criteria_id))
        by_criterion.setdefault(cname, []).append(pair)

    return {
        'overall':       compute_calibration(overall_pairs),
        'per_criterion': {c: compute_calibration(p) for c, p in by_criterion.items()},
    }


# =============================================================================
# Reporting
# =============================================================================

def format_calibration_report(result: Dict) -> str:
    overall = result.get('overall', {})
    lines = ['=== Score Calibration (AI vs examiner) ===']
    if overall.get('n', 0) == 0:
        lines.append('No FinalScore rows with an AI recommended score were found.')
        return '\n'.join(lines)

    lines.append(
        f"overall  n={overall['n']}  "
        f"pearson={_fmt(overall['pearson'])}  spearman={_fmt(overall['spearman'])}  "
        f"MAE={_fmt(overall['mae'])}  RMSE={_fmt(overall['rmse'])}  "
        f"within±1={_fmt(overall['within_1_rate'])}  bias={_fmt(overall['mean_bias'])}"
    )
    lines.append('')
    lines.append('per-criterion:')
    per = result.get('per_criterion', {})
    for cname, stats in sorted(per.items(), key=lambda kv: (kv[1].get('mae') or 0), reverse=True):
        if stats.get('n', 0) == 0:
            continue
        lines.append(
            f"  {cname[:32]:32s} n={stats['n']:>3}  "
            f"r={_fmt(stats['pearson'])}  MAE={_fmt(stats['mae'])}  "
            f"within±1={_fmt(stats['within_1_rate'])}"
        )
    return '\n'.join(lines)


def _fmt(v) -> str:
    return f'{v:.3f}' if isinstance(v, float) else ('—' if v is None else str(v))
