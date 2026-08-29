"""Database-facing attribution: gather evidence, resolve, write back.

The pure decision lives in services/resolver.py. This module is the bridge
between it and the ORM: it loads the evidence overlapping an answer window,
calls the resolver, persists the decision, and files the answer against the
resolved student.

Nothing here is on the critical path of a viva: every entry point degrades to
"leave it as the group answered it" rather than raising, because a failure to
identify the speaker must never stop the exam.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from core.models import GroupMember, StudentProfile, VivaAnswer, VivaQuestion
from attribution.models import (
    AnswerAttribution,
    AnswerContribution,
    AttributionStatus,
    EvidenceSource,
    SpeakerEvidence,
    UnknownSpeaker,
)
from .resolver import DEFAULT_WEIGHTS, Evidence, Decision, resolve

logger = logging.getLogger(__name__)

# Answers submitted as text can be typed long after the speaking stopped.
# Cap how far back a window may reach so one slow typist does not swallow the
# previous student's turn.
MAX_WINDOW_S = 300
# Speech that ends a moment before submit still belongs to this answer.
WINDOW_TAIL_S = 2


def is_enabled() -> bool:
    return getattr(settings, 'ATTRIBUTION_ENABLED', True)


def source_weights() -> dict:
    configured = getattr(settings, 'ATTRIBUTION_SOURCE_WEIGHTS', None)
    if not configured:
        return DEFAULT_WEIGHTS
    merged = dict(DEFAULT_WEIGHTS)
    merged.update(configured)
    return merged


def roster_ids(session) -> set[str]:
    """Student ids eligible to be named as the speaker for this session."""
    if session.group_id:
        return {
            str(sid) for sid in GroupMember.objects
            .filter(group_id=session.group_id)
            .values_list('student_id', flat=True)
        }
    if session.student_id:
        return {str(session.student_id)}
    return set()


def answer_window(question: VivaQuestion, answered_at=None):
    """The span of session time an answer to `question` could have been spoken.

    Runs from when the question was put to the student until the answer was
    submitted, clamped to MAX_WINDOW_S so a long pause before typing does not
    reach back into the previous turn.
    """
    end = (answered_at or timezone.now()) + timedelta(seconds=WINDOW_TAIL_S)
    start = question.generated_at or (end - timedelta(seconds=MAX_WINDOW_S))
    earliest = end - timedelta(seconds=MAX_WINDOW_S)
    return max(start, earliest), end


# Candidate keys for unenrolled speakers are namespaced so the resolver — which
# treats every candidate as an opaque string — can rank them alongside students
# without needing to know the difference.
UNKNOWN_PREFIX = 'unknown:'


def candidate_key(row) -> Optional[str]:
    """Resolver candidate key for an evidence row, or None if unattributable."""
    if row.student_id:
        return str(row.student_id)
    if row.unknown_speaker_id:
        return f"{UNKNOWN_PREFIX}{row.unknown_speaker_id}"
    return None


def split_candidate(key: str):
    """Candidate key -> (student_id, unknown_speaker_id); one of them is None."""
    if key.startswith(UNKNOWN_PREFIX):
        return None, key[len(UNKNOWN_PREFIX):]
    return key, None


def load_evidence(session, window_start, window_end) -> list[Evidence]:
    """Evidence rows overlapping the window, as pure resolver inputs."""
    rows = SpeakerEvidence.objects.filter(
        session=session,
        t_start__lt=window_end,
        t_end__gt=window_start,
    )
    return [
        Evidence(
            t_start=r.t_start,
            t_end=r.t_end,
            student_id=candidate_key(r),
            confidence=r.confidence,
            source=r.source,
        )
        for r in rows
    ]


def decide_for_answer(session, question, answered_at=None) -> Decision:
    """Resolve the speaker for one answer window. Never raises."""
    try:
        start, end = answer_window(question, answered_at)
        evidence = load_evidence(session, start, end)
        return resolve(start, end, evidence, weights=source_weights())
    except Exception:
        logger.exception(
            "Attribution resolve failed for session %s question %s",
            getattr(session, 'id', None), getattr(question, 'id', None),
        )
        return Decision(None, 0.0, 0.0, 'error')


def resolve_speaker_id(session, question, requested: str = 'group') -> str:
    """The speaker_id the scoring pipeline should use for this answer.

    Called at submit time. An explicit, valid client choice always wins — that
    is the kiosk dropdown and the examiner acting as a human override. Only
    when the client says nothing (or 'group') does attribution get a vote, and
    only a confident decision displaces the group default.
    """
    if not is_enabled():
        return requested

    valid = roster_ids(session)

    # An explicit client choice is a human statement of fact; record it as
    # manual evidence so the audit trail shows why the vote was bypassed.
    if requested and requested != 'group':
        if str(requested) in valid:
            _record_manual(session, question, requested)
            return str(requested)
        logger.warning(
            "Session %s: speaker_id %s is not on the roster; ignoring.",
            session.id, requested,
        )
        return 'group'

    # Individual sessions have a roster of one — no ambiguity to resolve.
    if not session.group_id and session.student_id:
        return str(session.student_id)

    decision = decide_for_answer(session, question)
    if decision.student_id and decision.student_id in valid:
        return decision.student_id
    # An unenrolled speaker leads this answer. Their marks are still recorded
    # against their UnknownSpeaker (see record_attribution), but the adaptive
    # questioner needs an ability state to steer from and they have none — so
    # the turn runs on group state until an examiner names them.
    return 'group'


def _record_manual(session, question, student_id) -> None:
    """Persist an examiner/kiosk selection as top-weight evidence."""
    try:
        start, end = answer_window(question)
        SpeakerEvidence.objects.get_or_create(
            session=session,
            source=EvidenceSource.MANUAL,
            student_id=student_id,
            t_start=start,
            t_end=end,
            defaults={'confidence': 1.0, 'meta': {'question_id': str(question.id)}},
        )
    except Exception:
        logger.exception("Could not record manual speaker evidence")


@transaction.atomic
def record_attribution(
    answer: VivaAnswer,
    session,
    decision: Decision,
    status: str = AttributionStatus.PROVISIONAL,
) -> Optional[AnswerAttribution]:
    """Persist a decision against an answer, and file the answer accordingly.

    Returns the AnswerAttribution row, or None if persisting failed — the
    caller is mid-viva and must not be interrupted by a bookkeeping error.
    """
    try:
        student_id, unknown_id = (
            split_candidate(decision.student_id) if decision.student_id
            else (None, None)
        )

        attribution, created = AnswerAttribution.objects.get_or_create(
            answer=answer,
            defaults={
                'session': session,
                'student_id': student_id,
                'unknown_speaker_id': unknown_id,
                'provisional_student_id': student_id,
            },
        )
        if not created:
            attribution.session = session
            attribution.student_id = student_id
            attribution.unknown_speaker_id = unknown_id

        attribution.share = decision.share
        attribution.margin = decision.margin
        attribution.outcome = decision.outcome
        attribution.source_breakdown = decision.breakdown
        attribution.co_speakers = list(decision.co_speakers)
        attribution.status = status
        attribution.save()

        _record_contributions(attribution, answer, decision)

        # Keep the answer itself in step, so existing report code that reads
        # VivaAnswer.student sees the resolved speaker without changes. An
        # unknown speaker leaves it null: the answer belongs to nobody on the
        # roster yet, and claiming otherwise would be the guess we refuse.
        if student_id and str(answer.student_id or '') != str(student_id):
            answer.student_id = student_id
            answer.save(update_fields=['student'])

        return attribution
    except Exception:
        logger.exception("Could not record attribution for answer %s", answer.id)
        return None


def _record_contributions(attribution, answer, decision) -> None:
    """Replace this answer's contribution rows from the decision's shares.

    Rewritten wholesale on every resolve so reconciliation cannot leave a
    stale share behind and double-count someone's marks.
    """
    AnswerContribution.objects.filter(attribution=attribution).delete()
    if not decision.shares:
        return

    dominant = decision.student_id
    rows = []
    for key, share in decision.shares.items():
        student_id, unknown_id = split_candidate(key)
        rows.append(AnswerContribution(
            attribution=attribution,
            answer=answer,
            student_id=student_id,
            unknown_speaker_id=unknown_id,
            share=share,
            is_dominant=(key == dominant),
        ))
    AnswerContribution.objects.bulk_create(rows)


def attribute_answer(answer: VivaAnswer, session, question) -> Optional[AnswerAttribution]:
    """Convenience: resolve this answer's window and persist the result."""
    if not is_enabled():
        return None
    decision = decide_for_answer(session, question, answered_at=answer.answered_at)
    return record_attribution(answer, session, decision)


# ---------------------------------------------------------------------------
# Post-hoc reconciliation
# ---------------------------------------------------------------------------


def reconcile_session(session) -> dict:
    """Re-resolve every answer using all evidence, including post-hoc CV.

    Run after CV analysis lands. Answers already CONFIRMED by an examiner are
    never re-filed: a disagreement after sign-off becomes a DISPUTED marker
    for a human to look at, not a silent re-score (see the invariant in
    attribution/models.py).
    """
    stats = {'checked': 0, 'changed': 0, 'disputed': 0, 'unchanged': 0}
    if not is_enabled():
        return stats

    answers = (
        VivaAnswer.objects
        .filter(question__session=session)
        .select_related('question', 'attribution')
    )

    for answer in answers:
        stats['checked'] += 1
        decision = decide_for_answer(session, answer.question, answer.answered_at)
        existing = getattr(answer, 'attribution', None)

        if existing and existing.status == AttributionStatus.CONFIRMED:
            same = str(existing.student_id or '') == str(decision.student_id or '')
            if not same and decision.student_id:
                existing.status = AttributionStatus.DISPUTED
                existing.source_breakdown = decision.breakdown
                existing.save(update_fields=[
                    'status', 'source_breakdown', 'updated_at',
                ])
                stats['disputed'] += 1
            else:
                stats['unchanged'] += 1
            continue

        before = str(existing.student_id or '') if existing else ''
        record_attribution(
            answer, session, decision, status=AttributionStatus.RECONCILED,
        )
        if before != str(decision.student_id or ''):
            stats['changed'] += 1
        else:
            stats['unchanged'] += 1

    logger.info("Attribution reconcile for session %s: %s", session.id, stats)
    return stats


def get_or_create_unknown_speaker(session, track_ref: str) -> UnknownSpeaker:
    """The pseudo-identity for one unrecognised face track.

    Keyed on the CV tracker's reference so the same person keeps one label for
    the whole session — which is the entire point: their marks have to pile up
    in one place for an examiner to hand over intact.
    """
    track_ref = str(track_ref)
    existing = UnknownSpeaker.objects.filter(
        session=session, track_refs__contains=track_ref,
    ).first()
    if existing:
        return existing

    # Labels run A, B, C... in order of first appearance — short enough for an
    # examiner to say out loud while scrubbing the recording.
    used = UnknownSpeaker.objects.filter(session=session).count()
    label = f"Unknown Speaker {chr(ord('A') + used)}" if used < 26 else f"Unknown Speaker {used + 1}"
    return UnknownSpeaker.objects.create(
        session=session, label=label, track_refs=[track_ref],
    )


@transaction.atomic
def resolve_unknown_speaker(unknown, examiner, student_id) -> UnknownSpeaker:
    """Name an unknown speaker, moving their held marks onto a real student.

    Everything already points at this UnknownSpeaker, so identifying them is a
    relabel rather than a recalculation: contributions and attributions are
    repointed and the marks they were holding land on the student who earned
    them. Nothing is rescored.
    """
    valid = roster_ids(unknown.session)
    if str(student_id) not in valid:
        raise ValueError("That student is not part of this session.")

    unknown.resolved_student_id = student_id
    unknown.resolved_by = examiner
    unknown.resolved_at = timezone.now()
    unknown.save(update_fields=[
        'resolved_student', 'resolved_by', 'resolved_at',
    ])

    AnswerContribution.objects.filter(unknown_speaker=unknown).update(
        student_id=student_id,
    )

    # Answers this speaker led now belong to that student, and are marked
    # confirmed — a human just made the call.
    for attribution in AnswerAttribution.objects.filter(
        unknown_speaker=unknown,
    ).select_related('answer'):
        attribution.student_id = student_id
        attribution.status = AttributionStatus.CONFIRMED
        attribution.confirmed_by = examiner
        attribution.confirmed_at = timezone.now()
        attribution.save(update_fields=[
            'student', 'status', 'confirmed_by', 'confirmed_at', 'updated_at',
        ])
        answer = attribution.answer
        answer.student_id = student_id
        answer.save(update_fields=['student'])

    SpeakerEvidence.objects.filter(unknown_speaker=unknown).update(
        student_id=student_id,
    )
    logger.info(
        "Unknown speaker %s resolved to student %s", unknown.label, student_id,
    )
    return unknown


def contribution_shares(session, student) -> Optional[dict]:
    """{answer_id: share} for one student across a session's answers.

    Returns None when this session has no contribution data at all, which the
    caller must read as "fall back to whole-answer credit" rather than "this
    student contributed nothing" — the two are opposite conclusions and the
    difference is somebody's grade. Sessions recorded before attribution
    existed, and sessions where every provider was silent, both land here.

    An unknown speaker's shares only appear once they have been identified;
    until then those marks are held, belonging to nobody.
    """
    if not is_enabled():
        return None

    rows = (
        AnswerContribution.objects
        .filter(attribution__session=session)
        .select_related('unknown_speaker')
    )

    shares: dict = {}
    any_rows = False
    target = str(student.id) if student is not None else None
    for row in rows:
        any_rows = True
        if target is None:
            continue
        if str(row.effective_student_id or '') == target:
            shares[str(row.answer_id)] = row.share

    if not any_rows:
        return None
    return shares


def student_contribution_totals(session) -> dict:
    """Per-student share of every answer, including marks still held unclaimed.

    Returns {'students': {student_id: {answers, total_share}},
             'unknown': [{id, label, answers, total_share}]} — the shape the
    examiner's review panel needs to show what is at stake in naming someone.
    """
    totals: dict = {}
    unknown: dict = {}

    rows = (
        AnswerContribution.objects
        .filter(attribution__session=session)
        .select_related('unknown_speaker')
    )
    for row in rows:
        sid = row.effective_student_id
        if sid:
            bucket = totals.setdefault(str(sid), {'answers': 0, 'total_share': 0.0})
        elif row.unknown_speaker_id:
            bucket = unknown.setdefault(str(row.unknown_speaker_id), {
                'id': str(row.unknown_speaker_id),
                'label': row.unknown_speaker.label,
                'answers': 0,
                'total_share': 0.0,
            })
        else:
            continue
        bucket['answers'] += 1
        bucket['total_share'] = round(bucket['total_share'] + row.share, 3)

    return {'students': totals, 'unknown': list(unknown.values())}


def confirm_attribution(attribution, examiner, student_id=None) -> AnswerAttribution:
    """Examiner confirms or overrides one answer's speaker. Their word is final."""
    if student_id is not None:
        valid = roster_ids(attribution.session)
        if str(student_id) not in valid:
            raise ValueError("That student is not part of this session.")
        attribution.student_id = student_id
        answer = attribution.answer
        answer.student_id = student_id
        answer.save(update_fields=['student'])

    attribution.status = AttributionStatus.CONFIRMED
    attribution.confirmed_by = examiner
    attribution.confirmed_at = timezone.now()
    attribution.save(update_fields=[
        'student', 'status', 'confirmed_by', 'confirmed_at', 'updated_at',
    ])
    return attribution
