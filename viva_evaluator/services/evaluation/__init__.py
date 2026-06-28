"""
Evaluation harness (C1/C2) — turns "we think it's good" into measured numbers.

C1  question-quality metrics  (metrics.py):
    Aggregate quality signals over a batch of generated questions —
    anchoring rate, Tier-1 pass rate, Critic pass rate, hallucination rate,
    specificity / Bloom-alignment, spoken-length compliance, attempts, latency.

C2  score calibration         (calibration.py):
    Compare the AI's recommended scores against examiner-confirmed final
    scores (stored on core.FinalScore) — correlation, MAE, RMSE, agreement.

Run via management commands:
    python manage.py eval_questions   --submission <uuid>
    python manage.py eval_calibration [--project <uuid>]
"""

from viva_evaluator.services.evaluation.metrics import (
    compute_question_metrics,
    format_metrics_table,
)
from viva_evaluator.services.evaluation.calibration import (
    compute_calibration,
    calibration_from_db,
    format_calibration_report,
)

__all__ = [
    'compute_question_metrics',
    'format_metrics_table',
    'compute_calibration',
    'calibration_from_db',
    'format_calibration_report',
]
