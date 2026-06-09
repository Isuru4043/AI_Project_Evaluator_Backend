"""
Speech confidence — derives a low/medium/high confidence flag from the
student's transcribed answer plus optional pause-timing metrics from the
frontend.

DESIGN (Web Speech API friendly):
    The browser does the actual speech recognition. Our backend doesn't
    receive audio; it receives the transcript text (and optionally a list
    of pause durations between speech segments).

    Two signals contribute:
      1. FILLER WORDS — "um", "uh", "like", "you know" — counted from the
         transcript text directly.
      2. PAUSE INTERVALS — long silences (>1500ms) between speech segments,
         supplied by the frontend in `speech_metrics.pause_intervals_ms`.

    Composite score:
        s = 0.3 * long_pauses + 0.1 * filler_count
        normalized to [0, 1] with a soft cap.

    Flag mapping:
        s > 0.6  → 'low'    (nervous, hesitant)
        0.3-0.6  → 'medium'
        s < 0.3  → 'high'   (confident, fluent)

CRITICAL: this flag is INFORMATIONAL ONLY. It does NOT enter the BKT update
or the rubric scoring. It is consumed by the Strategist purely to decide
HOW TO PHRASE the next question (e.g., reassure_and_clarify when low).

This is the asymmetric-multimodal handling pitch from the v3 spec.
"""

from viva_evaluator.services.confidence.speech_analyzer import (
    analyze_speech_confidence,
    SpeechConfidenceFlag,
    SpeechMetrics,
)

__all__ = [
    'analyze_speech_confidence',
    'SpeechConfidenceFlag',
    'SpeechMetrics',
]
