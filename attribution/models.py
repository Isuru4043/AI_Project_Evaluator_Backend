"""Speaker attribution — who answered, and on what evidence.

A group viva records everyone at once, so "who answered" cannot be read off
the request. This app collects timestamped speaker evidence from several
independent providers (Agora per-UID audio, live CV lip-motion, post-hoc CV
face recognition, the examiner) and fuses it into one defensible decision per
answer.

INVARIANTS (carried from exam-station-cv):
- Never guess. Ambiguity resolves to student=None ("uncertain"), which the
  examiner sees and decides. A wrong attribution is a grading error.
- Attribution changes WHERE a score is filed, never WHAT the score is.
- The examiner overrides everything, and their choice is final.

Timestamps are stored as absolute UTC datetimes, not offsets into the
recording. Live providers know wall-clock time but not the recording origin,
and answer windows are themselves absolute (question.generated_at ->
answer.answered_at), so absolute time is the only origin every source already
agrees on. Post-hoc CV events are the one exception: they carry offsets into
the recording and are converted at ingest using SessionRecording.
recording_started_at (Agora's sliceStartTime), the same conversion
cv_analysis.services.timeline already performs.
"""

import uuid
import hashlib

from django.db import models

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    StudentProfile,
    VivaAnswer,
)


class EvidenceSource(models.TextChoices):
    """Where a piece of speaker evidence came from.

    Order reflects typical reliability; the actual fusion weights live in
    settings.ATTRIBUTION_SOURCE_WEIGHTS so they can be tuned against a
    labelled pilot session without a redeploy.
    """

    MANUAL = 'manual', 'Examiner / kiosk selection'
    AGORA_STT = 'agora_stt', 'Agora STT voice tag'
    AGORA_VOLUME = 'agora_volume', 'Agora active speaker'
    POSTHOC_CV = 'posthoc_cv', 'Post-hoc CV analysis'
    LIVE_CV = 'live_cv', 'Live CV (lip motion x VAD)'
    SUBMITTER = 'submitter', 'Authenticated submitter'


class BindingMethod(models.TextChoices):
    ARCFACE = 'arcface', 'ArcFace vs enrollment photo'
    SEATING = 'seating', 'Seating order (left to right)'
    MANUAL = 'manual', 'Examiner assigned'


class FaceEnrollmentEmbeddingCache(models.Model):
    """Precomputed private ArcFace vectors for a student's enrollment set."""

    class Status(models.TextChoices):
        PENDING = 'pending', 'Pending'
        READY = 'ready', 'Ready'
        UNUSABLE = 'unusable', 'No usable enrollment'
        FAILED = 'failed', 'Failed'

    student = models.OneToOneField(
        StudentProfile,
        on_delete=models.CASCADE,
        related_name='face_embedding_cache',
        primary_key=True,
    )
    photo_fingerprint = models.CharField(max_length=64, db_index=True)
    embeddings = models.JSONField(default=list, blank=True)
    engine_version = models.CharField(max_length=64, blank=True, default='')
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.PENDING,
    )
    error_message = models.TextField(blank=True, default='')
    updated_at = models.DateTimeField(auto_now=True)

    @staticmethod
    def fingerprint(photo_refs):
        canonical = '\n'.join(sorted(str(value).split('?', 1)[0] for value in photo_refs))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class SpeakerBinding(models.Model):
    """A face position in the physical room bound to a roster student.

    Produced by the seat-binding pass: one still frame from the exam-station
    camera, faces detected by MediaPipe, each crop matched against that
    student's enrollment photo with ArcFace. Live CV then attributes lip
    activity to whichever binding a track sits nearest, so recognition (rare,
    expensive, server-side) is separated from activity (continuous, cheap).

    A binding is never deleted — it is superseded, so evidence recorded before
    a re-bind keeps the mapping that was true when it was captured.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='speaker_bindings',
    )
    # None = a face was detected that matched no enrolled roster member. That
    # is the extra-person case, not a failure — it is surfaced, never guessed.
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='speaker_bindings',
    )
    # Client-side track identifier this binding applies to, when known.
    track_ref = models.CharField(max_length=64, blank=True, default='')
    # Normalized [x0, y0, x1, y1] of the face box in the binding frame.
    bbox = models.JSONField(null=True, blank=True)
    method = models.CharField(
        max_length=20,
        choices=BindingMethod.choices,
        default=BindingMethod.ARCFACE,
    )
    confidence = models.FloatField(default=0.0)
    bound_at = models.DateTimeField(auto_now_add=True)
    # Set when a later binding pass replaces this one.
    superseded_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'Speaker Binding'
        verbose_name_plural = 'Speaker Bindings'
        ordering = ['-bound_at']
        indexes = [
            models.Index(fields=['session', 'superseded_at']),
        ]

    def __str__(self):
        who = self.student_id or 'unknown'
        return f"binding {self.track_ref or self.id} -> {who}"


class SpeakerEvidence(models.Model):
    """One timestamped observation that a student was (or may have been)
    speaking. Append-only: this is the audit trail behind every decision.

    A row with student=None means "someone spoke here but we could not say
    who" — it deliberately contributes nothing to the vote rather than being
    dropped, so an examiner reading the trail can see that the window was
    noisy rather than empty.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='speaker_evidence',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='speaker_evidence',
    )
    # Set instead of `student` when the speaker is a recognisably distinct
    # person who matches no enrolled student. Exactly one of the two is set;
    # both null means "speech we could not attribute at all".
    unknown_speaker = models.ForeignKey(
        'UnknownSpeaker',
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='evidence',
    )
    t_start = models.DateTimeField()
    t_end = models.DateTimeField()
    confidence = models.FloatField(default=1.0)
    source = models.CharField(max_length=20, choices=EvidenceSource.choices)
    # Provider-specific detail: agora uid, track ref, lip score, vad state...
    meta = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Speaker Evidence'
        verbose_name_plural = 'Speaker Evidence'
        ordering = ['t_start']
        indexes = [
            models.Index(fields=['session', 't_start', 't_end']),
            models.Index(fields=['session', 'source']),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=['session', 'source', 'student', 't_start', 't_end'],
                condition=models.Q(
                    student__isnull=False,
                    unknown_speaker__isnull=True,
                ),
                name='uniq_known_speaker_evidence_span',
            ),
            models.UniqueConstraint(
                fields=[
                    'session', 'source', 'unknown_speaker', 't_start', 't_end',
                ],
                condition=models.Q(unknown_speaker__isnull=False),
                name='uniq_unknown_speaker_evidence_span',
            ),
            models.UniqueConstraint(
                fields=['session', 'source', 't_start', 't_end'],
                condition=models.Q(
                    student__isnull=True,
                    unknown_speaker__isnull=True,
                ),
                name='uniq_unattributed_evidence_span',
            ),
        ]

    def __str__(self):
        return f"{self.source} {self.student_id or 'uncertain'} @ {self.t_start}"


class UnknownSpeaker(models.Model):
    """A distinct person in the session who matched no enrolled student.

    Without this, an unenrolled student's answers would fall to the group and
    their individual marks would vanish — punishing them for not having
    uploaded a photo rather than for anything they said. Instead their turns
    accumulate against a stable pseudo-identity ("Unknown Speaker A") that
    holds the marks intact until an examiner says who it was.

    Stability is what makes this work: the same unrecognised face across a
    whole session must land on ONE label, not a new one per turn. That comes
    from the CV tracker — `track_refs` are the face tracks clustered into this
    person — so the marks accumulate in one place.

    Resolving is a pure relabel: the evidence and contributions already exist
    and simply start pointing at a real student.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='unknown_speakers',
    )
    # Human-facing handle: "Unknown Speaker A", "Unknown Speaker B"...
    label = models.CharField(max_length=64)
    # CV track identifiers judged to be this same person.
    track_refs = models.JSONField(default=list, blank=True)
    # Set when an examiner identifies them. Their marks move to this student.
    resolved_student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_from_unknown',
    )
    resolved_by = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='resolved_unknown_speakers',
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    first_seen = models.DateTimeField(null=True, blank=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Unknown Speaker'
        verbose_name_plural = 'Unknown Speakers'
        ordering = ['label']
        constraints = [
            models.UniqueConstraint(
                fields=['session', 'label'], name='uniq_unknown_speaker_label',
            ),
        ]

    def __str__(self):
        if self.resolved_student_id:
            return f"{self.label} -> {self.resolved_student_id}"
        return self.label

    @property
    def is_resolved(self) -> bool:
        return self.resolved_student_id is not None


class AttributionStatus(models.TextChoices):
    PROVISIONAL = 'provisional', 'Provisional (live evidence)'
    RECONCILED = 'reconciled', 'Reconciled (post-hoc evidence)'
    CONFIRMED = 'confirmed', 'Confirmed by examiner'
    DISPUTED = 'disputed', 'Post-hoc disagrees after sign-off'


class AnswerAttribution(models.Model):
    """The decision for one answer, plus everything that produced it.

    Kept separate from VivaAnswer.student so the provisional live decision and
    the post-hoc reconciliation can disagree visibly instead of one silently
    overwriting the other.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    answer = models.OneToOneField(
        VivaAnswer,
        on_delete=models.CASCADE,
        related_name='attribution',
    )
    session = models.ForeignKey(
        EvaluationSession,
        on_delete=models.CASCADE,
        related_name='answer_attributions',
    )
    # None = uncertain. The answer stays filed against the group until a human
    # resolves it.
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='answer_attributions',
    )
    # What the live path decided at submit time, preserved even after
    # reconciliation changes `student` — so a disagreement stays inspectable.
    provisional_student = models.ForeignKey(
        StudentProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='provisional_attributions',
    )
    # Set instead of `student` when the dominant speaker is an unenrolled
    # person. The marks stay attached here until an examiner names them.
    unknown_speaker = models.ForeignKey(
        'UnknownSpeaker',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='answer_attributions',
    )
    share = models.FloatField(default=0.0)   # winner's share of weighted evidence
    margin = models.FloatField(default=0.0)  # lead over the runner-up
    outcome = models.CharField(max_length=20, default='no_evidence')
    # {source: {student_id: weighted_ms}} — drives the examiner's review panel.
    source_breakdown = models.JSONField(default=dict, blank=True)
    # Students above the co-speaker threshold: a collaborative answer.
    co_speakers = models.JSONField(default=list, blank=True)
    status = models.CharField(
        max_length=20,
        choices=AttributionStatus.choices,
        default=AttributionStatus.PROVISIONAL,
    )
    confirmed_by = models.ForeignKey(
        ExaminerProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='confirmed_attributions',
    )
    confirmed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Answer Attribution'
        verbose_name_plural = 'Answer Attributions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['session', 'status']),
        ]

    def __str__(self):
        return f"{self.answer_id} -> {self.student_id or 'uncertain'} [{self.status}]"

    @property
    def needs_review(self) -> bool:
        """True when an examiner should look at this before scores are signed."""
        if self.status == AttributionStatus.CONFIRMED:
            return False
        # An unknown speaker always needs a human: the marks are being held for
        # someone whose name only a person in the room can supply.
        if self.unknown_speaker_id is not None:
            return True
        if self.student_id is None:
            return True
        return (
            self.provisional_student_id is not None
            and self.provisional_student_id != self.student_id
        )


class AnswerContribution(models.Model):
    """How much of one answer each speaker contributed.

    A group answer is often not one person's: two students may build on each
    other. This table preserves those participation shares for examiner review
    while the resolved dominant speaker owns marks for individual criteria.
    Group-criterion marks remain shared by the whole group.

    `share` is the speaker's fraction of the weighted evidence in the answer
    window; shares across one answer sum to 1.0. `is_dominant` marks the
    primary answerer used by individual scoring and the adaptive questioner.
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    attribution = models.ForeignKey(
        AnswerAttribution,
        on_delete=models.CASCADE,
        related_name='contributions',
    )
    answer = models.ForeignKey(
        VivaAnswer,
        on_delete=models.CASCADE,
        related_name='contributions',
    )
    student = models.ForeignKey(
        StudentProfile,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='answer_contributions',
    )
    unknown_speaker = models.ForeignKey(
        UnknownSpeaker,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name='answer_contributions',
    )
    share = models.FloatField(default=0.0)
    is_dominant = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Answer Contribution'
        verbose_name_plural = 'Answer Contributions'
        ordering = ['-share']
        indexes = [
            models.Index(fields=['answer', 'student']),
        ]

    def __str__(self):
        who = self.student_id or self.unknown_speaker_id or 'uncertain'
        return f"{who} contributed {self.share:.0%} of {self.answer_id}"

    @property
    def effective_student_id(self):
        """The student these marks belong to, once an unknown speaker has been
        identified. None while their participation is still unclaimed."""
        if self.student_id:
            return self.student_id
        if self.unknown_speaker_id and self.unknown_speaker.resolved_student_id:
            return self.unknown_speaker.resolved_student_id
        return None
