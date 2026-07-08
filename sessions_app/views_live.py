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
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

from core.models import (
    EvaluationSession,
    GroupMember,
    StudentProfile,
    VivaAnswer,
    VivaQuestion,
)
from projects.permissions import IsExaminer, IsStudent
from sessions_app.views import _err, _get_examiner_profile, _is_assigned, _ok, _500


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


def _serialize_question(q, answer=None):
    return {
        'question_id': str(q.id),
        'question_text': q.question_text,
        'question_order': q.question_order,
        'asked_at': q.generated_at,
        'answer': None if answer is None else {
            'answer_text': answer.transcribed_answer,
            'answered_at': answer.answered_at,
            'answered_by': (
                f"{answer.student.user.first_name} "
                f"{answer.student.user.last_name}".strip()
                or answer.student.user.username
            ),
        },
    }


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
    """
    permission_classes = [IsAuthenticated, IsStudent]

    def get(self, request, session_id):
        try:
            session = _get_session(session_id)
            if not session:
                return _err('Session not found.', code=404)
            if _student_profile_in_session(request.user, session) is None:
                return _err('You are not part of this session.', code=403)

            question = (
                VivaQuestion.objects
                .filter(
                    session=session,
                    question_source=VivaQuestion.QuestionSource.EXAMINER,
                    answers__isnull=True,
                )
                .order_by('question_order')
                .first()
            )
            return _ok(
                'Pending examiner question.',
                {'pending': None if question is None else _serialize_question(question)},
            )
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

            answer = VivaAnswer.objects.create(
                question=question,
                student=profile,
                transcribed_answer=answer_text,
            )
            return _ok(
                'Answer recorded.',
                _serialize_question(question, answer),
                201,
            )
        except Exception as e:
            return _500(e)
