"""
Speech confidence analyzer — text-only (Web Speech API friendly).

PUBLIC API:
    analyze_speech_confidence(answer_text, speech_metrics=None) → dict

OUTPUT SHAPE:
    {
        'flag':            'low' | 'medium' | 'high',
        'composite_score': float,   # 0..1, higher = MORE hesitation
        'filler_count':    int,
        'filler_density':  float,   # fillers per 100 words
        'long_pause_count': int,
        'metrics_provided': bool,
    }

SCORING (matches v3 spec, calibrated for typed/spoken transcripts):
    long_pause_score   = min(long_pause_count * 0.15, 0.6)
    filler_score       = min(filler_density * 0.04, 0.4)   # density per 100 words
    composite          = long_pause_score + filler_score   ∈ [0, 1]

    flag = 'low'    if composite > 0.6
         = 'medium' if 0.3 ≤ composite ≤ 0.6
         = 'high'   if composite < 0.3
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

LONG_PAUSE_MS = 1500           # threshold for a "long" pause between segments

FILLER_WORDS = [
    r'\bum\b', r'\buh\b', r'\bhmm\b', r'\bhuh\b',
    r'\blike\b', r'\bbasically\b', r'\bactually\b',
    r'\byou know\b', r'\bsort of\b', r'\bkind of\b',
    r'\bso\.\.\.', r'\ber\b', r'\berm\b',
]
_FILLER_RE = re.compile('|'.join(FILLER_WORDS), re.IGNORECASE)

LOW_THRESHOLD = 0.6
HIGH_THRESHOLD = 0.3


# =============================================================================
# Types
# =============================================================================

class SpeechConfidenceFlag:
    LOW    = 'low'
    MEDIUM = 'medium'
    HIGH   = 'high'


@dataclass
class SpeechMetrics:
    """
    Frontend-supplied timing data. Sent alongside answer_text in the
    AnswerSubmitView body as `speech_metrics`. All fields optional.
    """
    duration_ms: Optional[int] = None
    pause_intervals_ms: Optional[List[int]] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'SpeechMetrics':
        if not data or not isinstance(data, dict):
            return cls()
        return cls(
            duration_ms=data.get('duration_ms'),
            pause_intervals_ms=data.get('pause_intervals_ms') or [],
            started_at=data.get('started_at'),
            ended_at=data.get('ended_at'),
        )


# =============================================================================
# Public API
# =============================================================================

def analyze_speech_confidence(
    answer_text: str,
    speech_metrics: Optional[Dict] = None,
) -> Dict:
    """
    Compute the confidence flag from the transcript + optional timing data.

    Args:
        answer_text:    The student's transcribed answer.
        speech_metrics: Optional dict from the frontend with pause intervals.

    Returns:
        Dict with flag, composite_score, filler_count, etc. (see module doc).
        Always returns a usable result — never raises.
    """
    text = (answer_text or '').strip()
    metrics = SpeechMetrics.from_dict(speech_metrics)

    # --- Filler words from text ----
    filler_count = len(_FILLER_RE.findall(text)) if text else 0
    word_count = max(1, len(text.split()))
    filler_density = (filler_count * 100.0) / word_count   # per 100 words

    # --- Long pauses from frontend metrics ----
    pauses = metrics.pause_intervals_ms or []
    long_pause_count = sum(1 for p in pauses if isinstance(p, (int, float)) and p >= LONG_PAUSE_MS)

    # --- Composite score ----
    long_pause_score = min(long_pause_count * 0.15, 0.6)
    filler_score = min(filler_density * 0.04, 0.4)
    composite = max(0.0, min(1.0, long_pause_score + filler_score))

    # --- Flag mapping ----
    if composite > LOW_THRESHOLD:
        flag = SpeechConfidenceFlag.LOW
    elif composite < HIGH_THRESHOLD:
        flag = SpeechConfidenceFlag.HIGH
    else:
        flag = SpeechConfidenceFlag.MEDIUM

    result = {
        'flag':              flag,
        'composite_score':   round(composite, 3),
        'filler_count':      filler_count,
        'filler_density':    round(filler_density, 2),
        'long_pause_count':  long_pause_count,
        'metrics_provided':  bool(metrics.pause_intervals_ms),
    }

    logger.info(
        'speech_confidence: flag=%s score=%.2f fillers=%d/%d (%.1f/100w) pauses>=%dms=%d',
        flag, composite, filler_count, word_count, filler_density,
        LONG_PAUSE_MS, long_pause_count,
    )
    return result
