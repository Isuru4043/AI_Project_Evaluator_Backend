"""Evidence fusion — pure logic, no Django, no database.

Every provider's output is normalized to `Evidence` before it gets here, so
this module is the ONLY place the "who spoke" decision is made, and it can be
unit-tested with scripted traces exactly as exam_cv's `decide_speaker` is.

The rule is deliberately boring: a weighted overlap vote with an explicit
abstention. Boring is a requirement, not a limitation — every decision has to
be explainable to an examiner disputing a grade, and "the model said so" is
not an explanation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Optional

# Fallback weights. Real deployments override via
# settings.ATTRIBUTION_SOURCE_WEIGHTS; see services/config.py.
DEFAULT_WEIGHTS: dict[str, float] = {
    'manual': 1.00,        # short-circuits the vote entirely
    'agora_stt': 0.90,     # per-UID stream, timestamped, carries the words
    'agora_volume': 0.85,  # per-UID stream, noisier
    'posthoc_cv': 0.80,    # full engine: real ArcFace + iris gaze
    'live_cv': 0.70,       # same logic, real-time constraints
    'submitter': 0.50,     # correlated with the speaker, often wrong in a room
}

# A winner must hold this much of the total weighted evidence...
MIN_SHARE = 0.60
# ...and lead the runner-up by this much. Both, or the window is uncertain.
MIN_MARGIN = 0.15
# Anyone at or above this share is recorded as a contributor to the answer.
CO_SPEAKER_SHARE = 0.20


@dataclass(frozen=True)
class Evidence:
    """One provider's claim that someone was speaking over a time span.

    student_id is None when a provider detected speech it could not attribute
    (an unrecognized face, an unmapped Agora UID). Such evidence contributes
    no weight to any candidate — it is carried only so the audit trail shows
    the window was contested rather than silent.
    """

    t_start: datetime
    t_end: datetime
    student_id: Optional[str]
    confidence: float
    source: str


@dataclass(frozen=True)
class Decision:
    student_id: Optional[str]        # None = uncertain, never a guess
    share: float
    margin: float
    outcome: str                     # attributed | uncertain | no_evidence | manual
    co_speakers: tuple[str, ...] = ()
    breakdown: dict = field(default_factory=dict)
    # Every candidate's fraction of the weighted evidence, summing to 1.0.
    # Marks for the answer are divided by these, so this is the field the
    # scoring path reads — `student_id` only names who leads it.
    shares: dict[str, float] = field(default_factory=dict)


def overlap_ms(ev: Evidence, w_start: datetime, w_end: datetime) -> float:
    """Milliseconds of `ev` that fall inside the window. Zero if disjoint."""
    start = max(ev.t_start, w_start)
    end = min(ev.t_end, w_end)
    if end <= start:
        return 0.0
    return (end - start).total_seconds() * 1000.0


def resolve(
    window_start: datetime,
    window_end: datetime,
    evidence: Iterable[Evidence],
    weights: Optional[dict[str, float]] = None,
    min_share: float = MIN_SHARE,
    min_margin: float = MIN_MARGIN,
    co_speaker_share: float = CO_SPEAKER_SHARE,
) -> Decision:
    """Fuse overlapping evidence into one decision for the answer window.

    A `manual` source short-circuits everything: the examiner (or the kiosk
    operator who picked a name from the dropdown) has stated who spoke, and no
    amount of model confidence outvotes a human.
    """
    weights = weights or DEFAULT_WEIGHTS
    evidence = list(evidence)

    # --- human override wins outright -----------------------------------
    for ev in evidence:
        if ev.source == 'manual' and ev.student_id is not None:
            if overlap_ms(ev, window_start, window_end) > 0:
                return Decision(
                    student_id=ev.student_id,
                    share=1.0,
                    margin=1.0,
                    outcome='manual',
                    breakdown={'manual': {ev.student_id: 1.0}},
                    # A human naming one speaker means the whole answer is
                    # theirs; they did not describe a split.
                    shares={ev.student_id: 1.0},
                )

    # --- weighted overlap vote ------------------------------------------
    score: dict[str, float] = defaultdict(float)
    breakdown: dict[str, dict[str, float]] = defaultdict(dict)
    unattributable_ms = 0.0

    for ev in evidence:
        ms = overlap_ms(ev, window_start, window_end)
        if ms <= 0:
            continue
        if ev.student_id is None:
            unattributable_ms += ms
            continue
        contribution = ms * max(0.0, min(1.0, ev.confidence)) * weights.get(ev.source, 0.5)
        if contribution <= 0:
            continue
        score[ev.student_id] += contribution
        breakdown[ev.source][ev.student_id] = round(
            breakdown[ev.source].get(ev.student_id, 0.0) + contribution, 2
        )

    total = sum(score.values())
    if not total:
        return Decision(
            student_id=None,
            share=0.0,
            margin=0.0,
            outcome='no_evidence',
            breakdown={'unattributable_ms': round(unattributable_ms, 1)} if unattributable_ms else {},
        )

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top_val = ranked[0]
    top_share = top_val / total
    runner_share = (ranked[1][1] / total) if len(ranked) > 1 else 0.0
    margin = top_share - runner_share

    detail = {k: v for k, v in breakdown.items()}
    if unattributable_ms:
        detail['unattributable_ms'] = round(unattributable_ms, 1)

    # Shares are reported for every candidate with a real stake, even when the
    # window is too close to name a winner: an answer two students genuinely
    # shared still has to divide its marks between them.
    eligible = [
        (sid, val) for sid, val in ranked
        if val / total >= co_speaker_share
    ]
    eligible_total = sum(val for _, val in eligible)
    shares = {
        sid: round(val / eligible_total, 4)
        for sid, val in eligible
    }
    # Rounding and removal of trivial interjections must never leave marks
    # unassigned. Put any four-decimal drift onto the leading contributor.
    if shares:
        leader = eligible[0][0]
        shares[leader] = round(
            shares[leader] + (1.0 - sum(shares.values())),
            4,
        )

    if top_share < min_share or margin < min_margin:
        return Decision(
            student_id=None,
            share=round(top_share, 3),
            margin=round(margin, 3),
            outcome='uncertain',
            co_speakers=tuple(shares),
            breakdown=detail,
            shares=shares,
        )

    return Decision(
        student_id=top_id,
        share=round(top_share, 3),
        margin=round(margin, 3),
        outcome='attributed',
        co_speakers=tuple(s for s in shares if s != top_id),
        breakdown=detail,
        shares=shares,
    )
