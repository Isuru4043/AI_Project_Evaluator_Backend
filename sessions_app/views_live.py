"""Live examiner interjection during an in-progress viva.

The AI conducts the viva automatically; an examiner may join at any time
(via the Agora room) and interject with their own questions. Typed questions
are delivered to the student's viva UI (which polls for them), take priority
over the next AI question, and both question and answer are stored on the
session like any other viva Q&A — attributed to the examiner via
``VivaQuestion.question_source = 'examiner'``.

Endpoints (prefixed with /api/ in root urls):
    POST /api/sessions/<id>/live-questions/            examiner asks
    GET  /api/sessions/<id>/live-questions/            examiner lists Q&A
    GET  /api/sessions/<id>/live-questions/pending/    student polls
    POST /api/sessions/<id>/live-questions/<qid>/answer/  student answers
"""

from django.db.models import Max
from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django_q.tasks import async_task

from core.models import (
    EvaluationSession,
    GroupMember,
    StudentProfile,
    VivaAnswer,
    VivaQuestion,
)
from projects.permissions import IsExaminer, IsStudent
from sessions_app.views import _err, _get_examiner_profile, _is_assigned, _ok, _500
from viva_evaluator.permissions import VivaSessionPermission


def _get_session(session_id):
    return (
        EvaluationSession.objects
        .filter(id=session_id)
        .select_related('project', 'student__user', 'group')
        .first()
    )


def _student_profile_in_session(user, session):
    """Return the requesting user's StudentProfile if they belong to this
    session (direct student or group member), else None."""
    profile = StudentProfile.objects.filter(user=user).first()
    if profile is None:
        return None
    if session.student_id and session.student_id == profile.id:
        return profile
    if session.group_id and GroupMember.objects.filter(
        group_id=session.group_id, student=profile,
    ).exists():
        return profile
    return None


def _answered_by(answer) -> str:
    """Who this answer is credited to, for display.

    Two things this has to survive. The custom User model removes
    first_name/last_name/username (they are None on the class) and carries
    full_name instead. And an answer may legitimately have no student: in a
    group viva, speaker attribution leaves an ambiguous answer unattributed
    rather than guessing, so it belongs to the group until an examiner says
    otherwise.
    """
    student = getattr(answer, 'student', None)
    if student is None:
        return 'The group'
    full = (student.user.full_name or '').strip()
    if full and full.lower() != 'none':
        return full
    return student.user.email


def _serialize_question(q, answer=None):
    return {
        'question_id': str(q.id),
        'question_text': q.question_text,
        'question_order': q.question_order,
        'asked_at': q.generated_at,
        'ready': bool((q.question_text or '').strip()),
        'answer': None if answer is None else {
            'answer_text': answer.transcribed_answer,
            'answered_at': answer.answered_at,
            'answered_by': _answered_by(answer),
        },
    }


VOICE_QUESTION_PLACEHOLDER = '[Examiner asked question via voice]'


def _open_examiner_questions(session):
    """Examiner interjections the student still owes an answer to.

    ``closed_at`` is what stops an abandoned interjection from following the
    student for the rest of the viva: the pending poll would keep returning
    it, freezing that student's screen on the examiner panel while the rest
    of the group moved on to the next AI question.
    """
    return VivaQuestion.objects.filter(
        session=session,
        question_source=VivaQuestion.QuestionSource.EXAMINER,
        answers__isnull=True,
        closed_at__isnull=True,
    ).order_by('question_order')


def _close_open_examiner_questions(session, blank_only=False):
    """Stop delivering unanswered interjections, keeping them in the report.

    A blank one was a voice question whose transcript never arrived; it is
    labelled rather than deleted so the report still shows that the examiner
    asked something at that point.
    """
    questions = _open_examiner_questions(session)
    if blank_only:
        questions = questions.filter(question_text='')
    # Resolve to ids first: the answers__isnull join makes this queryset
    # unusable for a direct .update().
    ids = list(questions.values_list('id', flat=True))
    if not ids:
        return 0
    rows = VivaQuestion.objects.filter(id__in=ids)
    rows.filter(question_text='').update(question_text=VOICE_QUESTION_PLACEHOLDER)
    rows.update(closed_at=timezone.now())
    return len(ids)


class LiveQuestionCreateView(APIView):
    """POST /api/sessions/<session_id>/live-questions/  (examiner)"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)
            if session.status != 'in_progress':
                return _err('The viva is not currently in progress.')

            question_text = (request.data.get('question_text') or '').strip()
            if not question_text:
                return _err('question_text is required.')

            # A new interjection replaces a voice draft the examiner started
            # and never completed, so only one question is ever pending.
            _close_open_examiner_questions(session, blank_only=True)

            next_order = (
                VivaQuestion.objects.filter(session=session)
                .aggregate(m=Max('question_order'))['m'] or 0
            ) + 1
            question = VivaQuestion.objects.create(
                session=session,
                project=session.project,
                question_text=question_text,
                question_source=VivaQuestion.QuestionSource.EXAMINER,
                question_order=next_order,
            )
            return _ok(
                'Question sent to the student.',
                _serialize_question(question),
                201,
            )
        except Exception as e:
            return _500(e)


class LiveQuestionListView(APIView):
    """GET /api/sessions/<session_id>/live-questions/  (examiner)"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def get(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)

            questions = (
                VivaQuestion.objects
                .filter(
                    session=session,
                    question_source=VivaQuestion.QuestionSource.EXAMINER,
                )
                .order_by('question_order')
                .prefetch_related('answers__student__user')
            )
            payload = [
                _serialize_question(q, next(iter(q.answers.all()), None))
                for q in questions
            ]
            return _ok('Live questions retrieved.', payload)
        except Exception as e:
            return _500(e)


class LiveQuestionPendingView(APIView):
    """GET /api/sessions/<session_id>/live-questions/pending/  (student poll)

    Returns the oldest examiner question that has no answer yet, or
    ``{'pending': None}``. The student's viva UI shows it before the next
    AI question — the examiner "interrupts" the AI.

    The response also carries ``paused`` and ``examiner_speaking``, which the
    student UI polls several times a second to decide whether to show the AI
    question or the examiner panel. Both were missing before, so the client
    read them as false on every poll and could never see the session resume.

    A question still being dictated has no text yet. Sending it as ``pending``
    would put an empty question on the student's screen, so it is reported as
    ``examiner_speaking`` instead and becomes pending once the text lands.
    """
    permission_classes = [IsAuthenticated, IsStudent]

    def get(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            if _student_profile_in_session(request.user, session) is None:
                return _err('You are not part of this session.', code=403)

            question = _open_examiner_questions(session).first()
            still_dictating = question is not None and not question.question_text.strip()
            return _ok('Pending examiner question.', {
                'pending': (
                    None if question is None or still_dictating
                    else _serialize_question(question)
                ),
                'examiner_speaking': still_dictating,
                'paused': bool(session.examiner_paused),
            })
        except Exception as e:
            return _500(e)


class LiveQuestionAnswerView(APIView):
    """POST /api/sessions/<session_id>/live-questions/<question_id>/answer/"""
    permission_classes = [IsAuthenticated, IsStudent]

    def post(self, request, session_id, question_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            profile = _student_profile_in_session(request.user, session)
            if profile is None:
                return _err('You are not part of this session.', code=403)

            question = VivaQuestion.objects.filter(
                id=question_id,
                session=session,
                question_source=VivaQuestion.QuestionSource.EXAMINER,
            ).first()
            if not question:
                return _err('Examiner question not found.', code=404)
            if question.answers.exists():
                return _err('This question has already been answered.')

            answer_text = (request.data.get('answer_text') or '').strip()
            if not answer_text:
                return _err('answer_text is required.')

            answer, created = VivaAnswer.objects.get_or_create(
                question=question,
                deduplication_key=f"student:{profile.id}",
                defaults={
                    'student': profile,
                    'transcribed_answer': answer_text,
                },
            )
            if not created:
                return _err('This question has already been answered.')
            return _ok(
                'Answer recorded.',
                _serialize_question(question, answer),
                201,
            )
        except Exception as e:
            return _500(e)


# =============================================================================
# Examiner Takeover Flow
# =============================================================================

class ExaminerTakeoverView(APIView):
    """POST /api/sessions/<session_id>/live-questions/takeover/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)
            
            session.examiner_paused = True
            session.save(update_fields=['examiner_paused'])
            
            ai_asked = session.viva_questions.filter(question_source=VivaQuestion.QuestionSource.AI).count()
            
            return _ok('AI paused.', {
                'paused': True,
                'ai_questions_asked': ai_asked,
                'max_ai_questions': session.max_total_questions
            })
        except Exception as e:
            return _500(e)


class ExaminerResumeView(APIView):
    """POST /api/sessions/<session_id>/live-questions/resume/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)
            
            # Handing back to the AI ends the interjection. Without this the
            # student's pending poll keeps returning the examiner's last
            # question, their screen never returns to the AI question, and the
            # group-sync poll stays blocked so they stop advancing with their
            # teammates.
            closed = _close_open_examiner_questions(session)

            session.examiner_paused = False
            session.save(update_fields=['examiner_paused'])

            return _ok('AI resumed.', {'paused': False, 'questions_closed': closed})
        except Exception as e:
            return _500(e)


class ExaminerEndSessionView(APIView):
    """POST /api/sessions/<session_id>/live-questions/end-session/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)
            
            # Label any voice question whose transcript never arrived, and
            # close whatever is still open so nothing is left pending against
            # a finished session.
            VivaQuestion.objects.filter(
                session=session,
                question_source=VivaQuestion.QuestionSource.EXAMINER,
                question_text='',
            ).update(question_text=VOICE_QUESTION_PLACEHOLDER)
            _close_open_examiner_questions(session)
            
            session.status = EvaluationSession.Status.COMPLETED
            session.examiner_paused = False
            session.save(update_fields=['status', 'examiner_paused'])
            
            async_task('viva_evaluator.services.reporting.post_viva_report.generate_post_viva_report', session.id)
            
            return _ok('Session ended.', {'ended': True})
        except Exception as e:
            return _500(e)


class ExaminerSessionStatusView(APIView):
    """GET /api/sessions/<session_id>/live-questions/status/"""
    permission_classes = [IsAuthenticated, VivaSessionPermission]

    def get(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            
            ai_asked = session.viva_questions.filter(question_source=VivaQuestion.QuestionSource.AI).count()
            examiner_asked = session.viva_questions.filter(question_source=VivaQuestion.QuestionSource.EXAMINER).count()
            
            return _ok('Session status retrieved.', {
                'paused': session.examiner_paused,
                'ai_questions_asked': ai_asked,
                'examiner_questions_asked': examiner_asked,
                'max_ai_questions': session.max_total_questions,
                'session_status': session.status
            })
        except Exception as e:
            return _500(e)


class ExaminerCreatePreemptiveQuestionView(APIView):
    """POST /api/sessions/<session_id>/live-questions/preemptive/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def post(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)
            
            # Starting a new voice question abandons an earlier draft that was
            # never completed, so the student is never held by a stale one.
            _close_open_examiner_questions(session, blank_only=True)

            next_order = (
                VivaQuestion.objects.filter(session=session)
                .aggregate(m=Max('question_order'))['m'] or 0
            ) + 1
            question = VivaQuestion.objects.create(
                session=session,
                project=session.project,
                question_text="",  # Blank initially
                question_source=VivaQuestion.QuestionSource.EXAMINER,
                question_order=next_order,
            )
            return _ok(
                'Pre-emptive question created.',
                _serialize_question(question),
                201,
            )
        except Exception as e:
            return _500(e)


class ExaminerUpdatePreemptiveQuestionView(APIView):
    """PATCH /api/sessions/<session_id>/live-questions/<question_id>/"""
    permission_classes = [IsAuthenticated, IsExaminer]

    def patch(self, request, session_id, question_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            ep = _get_examiner_profile(request.user)
            if not ep or not _is_assigned(ep, session.project):
                return _err('You are not assigned to this project.', code=403)
            
            question = VivaQuestion.objects.filter(
                id=question_id,
                session=session,
                question_source=VivaQuestion.QuestionSource.EXAMINER,
            ).first()
            if not question:
                return _err('Examiner question not found.', code=404)
            
            question_text = (request.data.get('question_text') or '').strip()
            if question_text:
                question.question_text = question_text
                question.save(update_fields=['question_text'])
                
            return _ok('Question text updated.', _serialize_question(question))
        except Exception as e:
            return _500(e)
