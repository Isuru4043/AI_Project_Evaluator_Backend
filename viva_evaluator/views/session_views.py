from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from urllib.request import urlopen

from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.serializers import (
    SubmissionUploadSerializer,
    SubmissionIndexStatusSerializer,
)

from viva_evaluator.views._helpers import (
    _resolve_session_submission,
    _difficulty_signal_from_score,
    _get_or_create_index_status,
)


class SessionStartView(APIView):
    """
    POST /api/viva/sessions/start/

    Starts a viva session for a submission.
    Generates and returns the first question for the first rubric criterion.

    Request body:
    {
        "session_id": "uuid-of-evaluation-session"
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        session_id = request.data.get('session_id')

        if not session_id:
            return Response(
                {"error": "session_id is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from core.models import EvaluationSession
            from viva_evaluator.services.session_manager import get_session_context
            from viva_evaluator.services.agents import (
                generate_anchored_question, QuestionerInput,
            )
            from viva_evaluator.services.rag.retrieval import retrieve_hybrid_for_turn
            from viva_evaluator.models import VivaQuestionExtension
            from core.models import VivaQuestion

            session = EvaluationSession.objects.get(id=session_id)
            submission = _resolve_session_submission(session)

            if not submission:
                return Response(
                    {"error": "No processed submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Make sure submission is ready
            try:
                index_status = _get_or_create_index_status(submission)
                if index_status.status != SubmissionIndexStatus.IndexStatus.READY:
                    return Response(
                        {"error": "Submission is not ready yet. Please wait for processing to complete."},
                        status=status.HTTP_400_BAD_REQUEST,
                    )
            except Exception:
                return Response(
                    {"error": "No processed submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get session context to find first criterion
            context = get_session_context(session_id)

            # If project has no rubric criteria configured, treat as misconfiguration
            if not context.get('all_criteria'):
                return Response(
                    {"error": "No rubric configured for this project. Please add rubric criteria before starting a viva session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if context['is_complete']:
                return Response(
                    {"error": "This session is already complete."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if session.status == 'in_progress' and session.viva_questions.exists():
                latest_q = session.viva_questions.order_by('question_order').last()
                ext = latest_q.extension if hasattr(latest_q, 'extension') else None
                return Response(
                    {
                        "message": "Session resumed.",
                        "session_id": session_id,
                        "question_id": str(latest_q.id),
                        "question_text": latest_q.question_text,
                        "blooms_level": latest_q.blooms_level,
                        "difficulty": ext.difficulty_level if ext else "medium",
                        "criterion": ext.criteria.criteria_name if ext and ext.criteria else "",
                        "question_number": latest_q.question_order,
                    },
                    status=status.HTTP_200_OK,
                )

            # =================================================================
            # Determine the first topic via the new pipeline
            # =================================================================
            from viva_evaluator.services.pipeline.session_state import load_session_state
            from viva_evaluator.services.pipeline.turn_pipeline import load_rubric, load_viva_topics, pick_next_topic, _grounding_is_weak
            
            state = load_session_state(session)
            rubric = load_rubric(session.project)
            for crit in rubric:
                state.get_or_init_coverage(str(crit['id']), questions_to_ask=int(crit['questions_to_ask']))
                state.get_or_init_bkt(str(crit['id']))
                
            viva_topics = load_viva_topics(session)
            next_topic = pick_next_topic(viva_topics, state, session)
            
            if not next_topic:
                # Should only happen if session budget is 0
                next_topic = viva_topics[0] if viva_topics else {
                    'topic_name': 'General', 'topic_focus': '', 'source_criteria_ids': []
                }

            # =================================================================
            # Hybrid retrieval — pull both FAISS chunks AND KG signals
            # =================================================================
            retrieval = retrieve_hybrid_for_turn(
                submission=submission,
                criterion_name=next_topic['topic_name'],
                criterion_description=next_topic['topic_focus'],
                last_answer='',
                top_k=3,
            )
            retrieved = retrieval['chunks']

            # =================================================================
            # Generate the anchored question via the new pipeline.
            # =================================================================
            question_data = generate_anchored_question(QuestionerInput(
                criterion_name=next_topic['topic_name'],
                criterion_description=next_topic['topic_focus'],
                retrieved_chunks=retrieved,
                kg_signals=retrieval,
                difficulty='medium',
                question_hints=[],
                recent_questions=[],
                is_first_question=True,
                question_number_in_criterion=1,
                weak_grounding=_grounding_is_weak(retrieved),
                session_id=str(session.id),
            ))

            # Save the question to DB
            question_order = context['total_questions_asked'] + 1
            question = VivaQuestion.objects.create(
                session=session,
                question_text=question_data['question_text'],
                blooms_level=question_data.get('blooms_level', 'Understand'),
                question_order=question_order,
                question_source='ai',
                viva_topic_name=next_topic['topic_name'],
                source_criteria_ids=next_topic['source_criteria_ids'],
            )

            # Save extension with criterion and difficulty (using the first criterion)
            from core.models import RubricCriteria
            if next_topic['source_criteria_ids']:
                first_crit_id = next_topic['source_criteria_ids'][0]
                criterion_obj = RubricCriteria.objects.filter(id=first_crit_id).first()
                if criterion_obj:
                    VivaQuestionExtension.objects.create(
                        question=question,
                        criteria=criterion_obj,
                        difficulty_level=question_data.get('difficulty', 'medium'),
                    )

            # Mark session as in progress
            from core.models import EvaluationSession as ES
            session.status = ES.Status.IN_PROGRESS
            from django.utils import timezone
            session.actual_start = timezone.now()
            session.save()

            return Response(
                {
                    "message": "Session started.",
                    "session_id": session_id,
                    "question_id": str(question.id),
                    "question_text": question.question_text,
                    "blooms_level": question.blooms_level,
                    "difficulty": question_data.get('difficulty', 'medium'),
                    "criterion": next_topic['topic_name'],
                    "question_number": question_order,
                    "tier1_passed": question_data.get('tier1_passed', False),
                    "tier1_failures": question_data.get('tier1_failures', []),
                    "critic_passed": question_data.get('critic_passed', True),
                    "critic_critique": question_data.get('critic_critique', ''),
                    "critic_scores": question_data.get('critic_scores', {}),
                },
                status=status.HTTP_200_OK,
            )

        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            from viva_evaluator.services.llm_service import LLMQuotaError
            if isinstance(e, LLMQuotaError):
                return Response(
                    {
                        "error": "The AI service is busy right now (quota limit reached). "
                                 "Please try again in a moment.",
                        "code": "ai_quota_exceeded",
                        "retry_after_seconds": getattr(e, 'retry_after_seconds', None),
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class AnswerSubmitView(APIView):
    """
    POST /api/viva/sessions/<session_id>/answer/

    Student submits their answer to the current question.
    The answer is evaluated, saved, and the next question is generated.

    Request body:
    {
        "question_id": "uuid",
        "answer_text": "student's answer here"
    }
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, session_id):
        question_id = request.data.get('question_id')
        answer_text = request.data.get('answer_text', '').strip()
        speech_metrics = request.data.get('speech_metrics')   # Week 6: optional

        if not question_id or not answer_text:
            return Response(
                {"error": "question_id and answer_text are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            from core.models import (
                EvaluationSession, VivaQuestion,
                VivaAnswer, RubricCriteria,
                GroupMember, StudentProfile,
            )
            from viva_evaluator.models import (
                VivaAnswerExtension, VivaQuestionExtension,
            )
            from viva_evaluator.services.pipeline import (
                process_answer_and_pick_next,
            )

            session = EvaluationSession.objects.get(id=session_id)
            question = VivaQuestion.objects.get(id=question_id, session=session)

            submission = _resolve_session_submission(session)
            student_profile = None
            try:
                student_profile = request.user.student_profile
            except AttributeError:
                pass
            if not student_profile:
                student_profile = session.student
            if not student_profile and session.group:
                first_member = GroupMember.objects.filter(group=session.group).first()
                if first_member:
                    student_profile = first_member.student

            if not submission:
                return Response(
                    {"error": "No submission found for this session."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # =================================================================
            # WEEK 5 — single pipeline call replaces legacy answer_evaluator
            # + session_manager glue. Runs Analyzer → BKT update →
            # termination check → Strategist → Questioner.
            # =================================================================
            result = process_answer_and_pick_next(
                session=session,
                submission=submission,
                prev_question_obj=question,
                student_answer=answer_text,
                speech_metrics=speech_metrics,
            )

            # =================================================================
            # A1 — Response Triage / clarification branch.
            # The student didn't understand the question, so scoring was
            # SUSPENDED. Record the response for the transcript as UNSCORED
            # (ai_answer_score = None), then return a clarified re-ask.
            # No score, no ability change, no turn consumed.
            # =================================================================
            if result.get('clarification'):
                triage = result.get('triage', {})

                # Audit-only record of the (confused) response — unscored.
                VivaAnswer.objects.create(
                    question=question,
                    student=student_profile,
                    transcribed_answer=answer_text,
                    ai_answer_score=None,
                )

                payload = result['clarified_question_payload']
                qd = payload['question_data']
                total_asked = session.viva_questions.count()

                clarified_q = VivaQuestion.objects.create(
                    session=session,
                    question_text=qd['question_text'],
                    blooms_level=qd.get('blooms_level', payload['bloom_level']),
                    question_order=total_asked + 1,
                    question_source='ai',
                    viva_topic_name=payload['topic']['topic_name'],
                    source_criteria_ids=payload['topic']['source_criteria_ids'],
                )
                try:
                    if payload['topic']['source_criteria_ids']:
                        crit_obj = RubricCriteria.objects.get(id=payload['topic']['source_criteria_ids'][0])
                        VivaQuestionExtension.objects.create(
                            question=clarified_q,
                            criteria=crit_obj,
                            difficulty_level=qd.get('difficulty', payload['difficulty']),
                        )
                except RubricCriteria.DoesNotExist:
                    pass

                _is_restate = triage.get('label') == 'GARBLED_TRANSCRIPTION'
                _msg = (
                    "I didn't catch that clearly — could you say your answer again? "
                    "This was not scored."
                    if _is_restate else
                    "It looks like the question may have been unclear. "
                    "Here's a clearer version — this was not scored."
                )
                return Response(
                    {
                        "answer_saved": True,
                        "scored": False,
                        "clarification": True,
                        "clarification_attempt": result.get('clarification_attempt'),
                        "triage": {
                            "label":     triage.get('label'),
                            "rationale": triage.get('rationale'),
                        },
                        "message": _msg,
                        "next_question": {
                            "question_id":     str(clarified_q.id),
                            "question_text":   clarified_q.question_text,
                            "blooms_level":    clarified_q.blooms_level,
                            "difficulty":      qd.get('difficulty', payload['difficulty']),
                            "criterion":       payload['topic']['topic_name'],
                            "question_number": total_asked + 1,
                            "is_clarification": not _is_restate,
                            "is_restate":       _is_restate,
                        },
                    },
                    status=status.HTTP_200_OK,
                )

            analysis = result['analysis']
            soft_score = result['soft_score']
            confidence = result.get('speech_confidence') or {}

            # Build a unified rubric scores block for the response.
            rubric_payload = {
                'correctness': analysis.get('correctness', {}),
                'depth':       analysis.get('depth', {}),
                'consistency': analysis.get('consistency', {}),
                'soft_score':  soft_score,
                'reasoning':   analysis.get('reasoning', ''),
                'gap_identified':       analysis.get('gap_identified', ''),
                'revealed_assumption':  analysis.get('revealed_assumption', ''),
                'contradicts_code_flag': analysis.get('contradicts_code_flag', False),
                'charitable':           analysis.get('charitable', {'applied': False}),
                'consistency_adjustment': analysis.get('consistency_adjustment', {'applied': False}),
                'self_correction':      analysis.get('self_correction', {'applied': False}),
            }

            detailed_analysis = {
                "rubric": rubric_payload,
                "strategy": result.get('strategy', {}),
                "speech_confidence": confidence
            }

            # Persist the answer + extension (audit trail).
            answer = VivaAnswer.objects.create(
                question=question,
                student=student_profile,
                transcribed_answer=answer_text,
                # ai_answer_score is on a 0-10 scale historically; map from soft_score
                ai_answer_score=round(soft_score * 10.0, 2),
            )
            VivaAnswerExtension.objects.create(
                answer=answer,
                llm_score=round(soft_score * 10.0, 2),
                llm_reasoning=analysis.get('reasoning', '') or '',
                next_difficulty_signal=_difficulty_signal_from_score(soft_score),
                detailed_ai_analysis=detailed_analysis,
            )

            # ---- Session complete -------------------------------------------
            if result['session_complete']:
                return Response(
                    {
                        "answer_saved": True,
                        "session_complete": True,
                        "termination_reason": result.get('termination_reason'),
                        "rubric": rubric_payload,
                        "speech_confidence": confidence,
                        "message": "All termination conditions satisfied — session complete.",
                    },
                    status=status.HTTP_200_OK,
                )

            # ---- Save next question -----------------------------------------
            payload = result['next_question_payload']
            next_topic = payload['topic']
            qd = payload['question_data']

            total_asked = session.viva_questions.count()

            next_question = VivaQuestion.objects.create(
                session=session,
                question_text=qd['question_text'],
                blooms_level=qd.get('blooms_level', payload['bloom_level']),
                question_order=total_asked + 1,
                question_source='ai',
                viva_topic_name=next_topic['topic_name'],
                source_criteria_ids=next_topic['source_criteria_ids'],
            )
            try:
                if next_topic['source_criteria_ids']:
                    next_criterion_obj = RubricCriteria.objects.get(id=next_topic['source_criteria_ids'][0])
                    VivaQuestionExtension.objects.create(
                        question=next_question,
                        criteria=next_criterion_obj,
                        difficulty_level=qd.get('difficulty', payload['difficulty']),
                    )
            except RubricCriteria.DoesNotExist:
                pass

            return Response(
                {
                    "answer_saved": True,
                    "session_complete": False,
                    "rubric": rubric_payload,
                    "speech_confidence": confidence,
                    "strategy": {
                        "bloom_level":     payload['bloom_level'],
                        "socratic_intent": payload['socratic_intent'],
                        "p_lt":            round(payload['p_lt'], 3),
                        "rationale":       result['strategy'].get('rationale', ''),
                    },
                    "next_question": {
                        "question_id":     str(next_question.id),
                        "question_text":   next_question.question_text,
                        "blooms_level":    payload['bloom_level'],
                        "difficulty":      payload['difficulty'],
                        "criterion":       next_topic['topic_name'],
                        "question_number": total_asked + 1,
                        "tier1_passed":    qd.get('tier1_passed', False),
                        "tier1_failures":  qd.get('tier1_failures', []),
                        "critic_passed":   qd.get('critic_passed', True),
                        "critic_critique": qd.get('critic_critique', ''),
                        "critic_scores":   qd.get('critic_scores', {}),
                        "attempts":        qd.get('attempts', 1),
                    },
                },
                status=status.HTTP_200_OK,
            )

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except VivaQuestion.DoesNotExist:
            return Response({"error": "Question not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            from viva_evaluator.services.llm_service import LLMQuotaError
            if isinstance(e, LLMQuotaError):
                return Response(
                    {
                        "error": "The AI service is busy right now (quota limit reached). "
                                 "Please try again in a moment.",
                        "code": "ai_quota_exceeded",
                        "retry_after_seconds": getattr(e, 'retry_after_seconds', None),
                    },
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class SessionReportView(APIView):
    """
    GET /api/viva/sessions/<session_id>/report/

    Returns the structured post-viva report. Allowed even before the
    session is COMPLETED so examiners can review mid-session if needed —
    the report just reflects whatever turns have happened so far.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        try:
            from core.models import EvaluationSession
            from viva_evaluator.services.reporting import generate_post_viva_report

            session = EvaluationSession.objects.get(id=session_id)
            report = generate_post_viva_report(session)
            report['session_status'] = session.status
            return Response(report, status=status.HTTP_200_OK)

        except EvaluationSession.DoesNotExist:
            return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class EvaluationSessionCreateView(APIView):
    """
    POST /api/viva/sessions/create/

    Examiner creates a viva session linking project, student, submission.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from viva_evaluator.serializers import (
            EvaluationSessionCreateSerializer,
            EvaluationSessionDetailSerializer,
        )
        serializer = EvaluationSessionCreateSerializer(data=request.data)
        if serializer.is_valid():
            session = serializer.save()
            return Response(
                EvaluationSessionDetailSerializer(session).data,
                status=status.HTTP_201_CREATED,
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SessionListView(APIView):
    """
    GET /api/viva/projects/<project_id>/sessions/

    Returns all sessions for a project.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, project_id):
        from core.models import EvaluationSession, Project
        from viva_evaluator.serializers import EvaluationSessionDetailSerializer
        try:
            project = Project.objects.get(id=project_id)
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found."},
                status=status.HTTP_404_NOT_FOUND,
            )
        sessions = EvaluationSession.objects.filter(
            project=project
        ).order_by('scheduled_start')
        serializer = EvaluationSessionDetailSerializer(sessions, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)


class SessionStatusView(APIView):
    """
    GET /api/viva/sessions/<session_id>/status/

    Returns current status of a session.
    Frontend polls this to know if session is scheduled, in progress, or complete.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        from core.models import EvaluationSession
        try:
            session = EvaluationSession.objects.get(id=session_id)
        except EvaluationSession.DoesNotExist:
            return Response(
                {"error": "Session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Count questions and answers so far
        questions = session.viva_questions.all()
        total_questions = questions.count()
        total_answers = sum(q.answers.count() for q in questions)

        return Response(
            {
                "session_id": str(session.id),
                "status": session.status,
                "phase": session.phase,
                "demo_enabled": session.demo_enabled,
                "demo_completed_at": session.demo_completed_at,
                "scheduled_start": session.scheduled_start,
                "scheduled_end": session.scheduled_end,
                "actual_start": session.actual_start,
                "total_questions_asked": total_questions,
                "total_answers_submitted": total_answers,
            },
            status=status.HTTP_200_OK,
        )


class CurrentQuestionView(APIView):
    """
    GET /api/viva/sessions/<session_id>/current/

    Returns the latest AI-generated question for the session (read-only, no
    generation). In group mode every member's viva UI polls this so that when
    one teammate answers and the AI advances, the others' screens catch up.

    Examiner-interjected questions are delivered through the separate
    live-questions endpoints, so they are excluded here.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        from core.models import EvaluationSession, VivaQuestion

        session = EvaluationSession.objects.filter(id=session_id).first()
        if not session:
            return Response({"error": "Session not found."},
                            status=status.HTTP_404_NOT_FOUND)

        latest_q = (
            session.viva_questions
            .exclude(question_source=VivaQuestion.QuestionSource.EXAMINER)
            .order_by('question_order')
            .last()
        )
        if latest_q is None:
            return Response({"question": None, "session_complete": False},
                            status=status.HTTP_200_OK)

        try:
            ext = latest_q.extension
        except Exception:
            ext = None
        return Response(
            {
                "question": {
                    "question_id": str(latest_q.id),
                    "question_text": latest_q.question_text,
                    "blooms_level": latest_q.blooms_level,
                    "difficulty": ext.difficulty_level if ext else "medium",
                    "criterion": (
                        ext.criteria.criteria_name if ext and ext.criteria else ""
                    ),
                    "question_number": latest_q.question_order,
                },
                "session_complete": session.status == 'completed',
            },
            status=status.HTTP_200_OK,
        )

class SessionDetailedReportView(APIView):
    """
    GET /api/evaluator/sessions/<session_id>/detailed-report/
    
    Returns the comprehensive timeline of the AI viva session for the examiner.
    Includes every question asked, the student's transcribed answer, and the
    granular AI reasoning (gaps identified, strategies, soft scores) stored
    in the detailed_ai_analysis JSON field.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        from core.models import EvaluationSession, VivaQuestion
        import traceback as tb_module

        try:
            try:
                session = EvaluationSession.objects.select_related(
                    'project', 'student__user', 'group'
                ).get(id=session_id)
            except EvaluationSession.DoesNotExist:
                return Response({"error": "Session not found."}, status=status.HTTP_404_NOT_FOUND)

            # Get session info
            session_data = {
                "session_id": str(session.id),
                "project_name": session.project.project_name,
                "status": session.status,
                "scheduled_start": session.scheduled_start,
                "scheduled_end": session.scheduled_end,
                "location_room": session.location_room,
                "student_name": session.student.user.full_name if session.student and getattr(session.student, 'user', None) else None,
                "group_name": session.group.group_name if session.group else None,
            }

            # Get summary report if exists
            report_data = None
            try:
                summary = session.summary_report
                report_data = {
                    "total_ai_score": summary.total_ai_score,
                    "total_final_score": summary.total_final_score,
                    "grade": summary.grade,
                    "overall_feedback": summary.overall_feedback,
                    "emotional_summary": summary.emotional_summary,
                    "integrity_flags_summary": summary.integrity_flags_summary,
                    "scores_status": summary.scores_status,
                    "scores_approved_at": summary.scores_approved_at.isoformat() if summary.scores_approved_at else None,
                }
            except Exception:
                pass

            # Fetch all questions and their answers in chronological order
            questions_qs = (
                session.viva_questions.all()
                .prefetch_related('answers__student__user')
                .order_by('question_order')
            )

            timeline = []
            for q in questions_qs:
                try:
                    q_ext = q.extension
                except Exception:
                    q_ext = None
                
                # Usually one answer per question in the AI loop
                answer = q.answers.order_by('-answered_at').first()
                ans_ext = None
                if answer:
                    try:
                        ans_ext = answer.extension
                    except Exception:
                        ans_ext = None

                timeline.append({
                    "question_id": str(q.id),
                    "question_text": q.question_text,
                    "question_source": q.question_source,
                    "question_order": q.question_order,
                    "blooms_level": q.blooms_level,
                    "difficulty": q_ext.difficulty_level if q_ext else None,
                    "criterion": q_ext.criteria.criteria_name if q_ext and q_ext.criteria else getattr(q, 'viva_topic_name', None),
                    "asked_at": str(getattr(q, 'asked_at', None) or q.generated_at),
                    "answer": {
                        "answer_id": str(answer.id),
                        "transcribed_answer": answer.transcribed_answer,
                        "answered_at": str(answer.answered_at) if answer.answered_at else None,
                        "answered_by": answer.student.user.full_name if getattr(answer, 'student', None) and getattr(answer.student, 'user', None) else None,
                        "llm_score": float(ans_ext.llm_score) if ans_ext and ans_ext.llm_score is not None else None,
                        "examiner_override_score": float(answer.examiner_override_score) if answer.examiner_override_score is not None else None,
                        "examiner_override_note": answer.examiner_override_note,
                        "llm_reasoning": ans_ext.llm_reasoning if ans_ext else None,
                        "detailed_ai_analysis": ans_ext.detailed_ai_analysis if ans_ext else None,
                    } if answer else None
                })

            return Response({
                "session": session_data,
                "report": report_data,
                "timeline": timeline,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({
                "error": str(e),
                "traceback": tb_module.format_exc(),
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

