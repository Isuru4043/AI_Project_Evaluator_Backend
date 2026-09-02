from datetime import timedelta
from unittest.mock import patch

from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from attribution.models import AnswerAttribution, AnswerContribution
from core.models import (
    ExaminerProfile,
    EvaluationSession,
    GroupMember,
    Project,
    ProjectExaminer,
    RubricCategory,
    RubricCriteria,
    SessionSummaryReport,
    StudentGroup,
    StudentProfile,
    User,
    VivaAnswer,
    VivaQuestion,
)
from viva_evaluator.models import VivaAnswerProcessingClaim, VivaQuestionExtension
from viva_evaluator.services.answer_idempotency import (
    IdempotencyConflict,
    acquire_claim,
    request_fingerprint,
)


class AnswerClaimRecoveryTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            project_name="Idempotency recovery",
            evaluation_mode=Project.EvaluationMode.REMOTE,
        )
        self.session = EvaluationSession.objects.create(
            project=self.project,
            scheduled_start=timezone.now(),
            scheduled_end=timezone.now() + timedelta(hours=1),
        )
        self.question = VivaQuestion.objects.create(
            session=self.session,
            project=self.project,
            question_text="Explain the design.",
            question_order=1,
        )
        self.speaker = "student:test-student"
        self.key = f"answer:{self.question.id}:{self.speaker}"
        self.first_hash = request_fingerprint(
            answer_text="First transcript",
            speech_metrics=None,
            speaker_id=self.speaker,
        )
        self.second_hash = request_fingerprint(
            answer_text="Corrected transcript",
            speech_metrics=None,
            speaker_id=self.speaker,
        )

    def create_claim(self):
        return acquire_claim(
            session=self.session,
            question=self.question,
            speaker=self.speaker,
            idempotency_key=self.key,
            request_hash=self.first_hash,
        ).claim

    def reacquire_with_corrected_answer(self, key=None):
        return acquire_claim(
            session=self.session,
            question=self.question,
            speaker=self.speaker,
            idempotency_key=key or self.key,
            request_hash=self.second_hash,
        )

    def test_expired_claim_without_answer_accepts_corrected_transcript(self):
        claim = self.create_claim()
        old_owner = claim.owner_token
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            lease_expires_at=timezone.now() - timedelta(seconds=1),
        )

        result = self.reacquire_with_corrected_answer()

        self.assertEqual(result.action, "process")
        result.claim.refresh_from_db()
        self.assertEqual(result.claim.request_hash, self.second_hash)
        self.assertNotEqual(result.claim.owner_token, old_owner)
        self.assertGreater(result.claim.lease_expires_at, timezone.now())

    def test_failed_claim_without_answer_accepts_new_key_and_transcript(self):
        claim = self.create_claim()
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            status=VivaAnswerProcessingClaim.Status.FAILED,
            error_code="pipeline_error",
        )
        replacement_key = f"replacement:{self.question.id}"

        result = self.reacquire_with_corrected_answer(key=replacement_key)

        self.assertEqual(result.action, "process")
        result.claim.refresh_from_db()
        self.assertEqual(result.claim.idempotency_key, replacement_key)
        self.assertEqual(result.claim.request_hash, self.second_hash)
        self.assertEqual(result.claim.error_code, "")

    def test_expired_claim_with_persisted_answer_remains_immutable(self):
        claim = self.create_claim()
        VivaAnswer.objects.create(
            question=self.question,
            deduplication_key=self.speaker,
            transcribed_answer="First transcript",
        )
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            lease_expires_at=timezone.now() - timedelta(seconds=1),
        )

        with self.assertRaises(IdempotencyConflict):
            self.reacquire_with_corrected_answer()

    def test_completed_claim_with_different_answer_remains_immutable(self):
        claim = self.create_claim()
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            status=VivaAnswerProcessingClaim.Status.COMPLETED,
            response_payload={"answer_saved": True},
        )

        with self.assertRaises(IdempotencyConflict):
            self.reacquire_with_corrected_answer()


class GroupScoringReportTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            project_name="Group scoring",
            is_group_project=True,
            evaluation_mode=Project.EvaluationMode.PHYSICAL,
        )
        self.group = StudentGroup.objects.create(
            project=self.project,
            group_name="Group A",
        )
        self.alice = self._student("alice@example.com", "REG-ALICE")
        self.bob = self._student("bob@example.com", "REG-BOB")
        GroupMember.objects.create(group=self.group, student=self.alice)
        GroupMember.objects.create(group=self.group, student=self.bob)
        self.session = EvaluationSession.objects.create(
            project=self.project,
            group=self.group,
            scheduled_start=timezone.now(),
            scheduled_end=timezone.now() + timedelta(hours=1),
            status=EvaluationSession.Status.COMPLETED,
        )
        category = RubricCategory.objects.create(
            project=self.project,
            category_name="Knowledge",
            weight_percentage=100,
        )
        self.group_criterion = RubricCriteria.objects.create(
            category=category,
            criteria_name="Team architecture",
            max_score=10,
            is_individual=False,
        )
        self.individual_criterion = RubricCriteria.objects.create(
            category=category,
            criteria_name="Individual understanding",
            max_score=10,
            is_individual=True,
        )
        examiner_user = User.objects.create_user(
            email="examiner@example.com",
            password="test-password",
            full_name="Test Examiner",
            role=User.Role.EXAMINER,
        )
        self.examiner = ExaminerProfile.objects.create(user=examiner_user)
        ProjectExaminer.objects.create(
            project=self.project,
            examiner=self.examiner,
            role_in_project=ProjectExaminer.RoleInProject.LEAD,
        )
        self.client = APIClient()
        self.client.force_authenticate(examiner_user)

    @staticmethod
    def _student(email, registration_number):
        user = User.objects.create_user(
            email=email,
            password="test-password",
            full_name=email.split('@')[0].title(),
            role=User.Role.STUDENT,
        )
        return StudentProfile.objects.create(
            user=user,
            registration_number=registration_number,
        )

    def _answer(self, order, criterion, score, student=None):
        question = VivaQuestion.objects.create(
            session=self.session,
            project=self.project,
            question_text=f"Question {order}",
            question_order=order,
        )
        VivaQuestionExtension.objects.create(
            question=question,
            criteria=criterion,
        )
        return VivaAnswer.objects.create(
            question=question,
            student=student,
            ai_answer_score=score,
            transcribed_answer="Test answer",
        )

    def test_group_criteria_are_shared_but_individual_scores_are_separate(self):
        from viva_evaluator.services.scoring_service import ScoringService

        self._answer(1, self.group_criterion, 8, student=self.alice)
        self._answer(2, self.individual_criterion, 9, student=self.alice)
        self._answer(3, self.individual_criterion, 5, student=self.bob)

        alice = ScoringService.aggregate_student_score(self.session, self.alice)
        bob = ScoringService.aggregate_student_score(self.session, self.bob)

        self.assertEqual(alice['percentage'], 85.0)
        self.assertEqual(bob['percentage'], 65.0)
        self.assertNotEqual(alice['grade'], bob['grade'])

    def test_reports_cover_the_complete_roster(self):
        from viva_evaluator.services.session_reports import (
            ensure_participant_reports,
        )

        ensure_participant_reports(self.session)

        report_students = set(
            self.session.summary_reports.values_list('student_id', flat=True)
        )
        self.assertEqual(report_students, {self.alice.id, self.bob.id})

    def test_draft_reports_store_each_students_own_total(self):
        from viva_evaluator.services.session_reports import (
            refresh_draft_summary_reports,
        )

        self._answer(1, self.group_criterion, 8, student=self.alice)
        self._answer(2, self.individual_criterion, 9, student=self.alice)
        self._answer(3, self.individual_criterion, 5, student=self.bob)

        refresh_draft_summary_reports(self.session)

        alice_report = self.session.summary_reports.get(student=self.alice)
        bob_report = self.session.summary_reports.get(student=self.bob)
        self.assertEqual(float(alice_report.total_final_score), 85.0)
        self.assertEqual(float(bob_report.total_final_score), 65.0)
        self.assertEqual(float(alice_report.total_ai_score), 85.0)
        self.assertEqual(float(bob_report.total_ai_score), 65.0)

    def test_unresolved_individual_answer_blocks_finalization_contract(self):
        from viva_evaluator.services.session_reports import (
            unresolved_individual_answers,
        )

        answer = self._answer(1, self.individual_criterion, 7, student=None)

        unresolved = unresolved_individual_answers(self.session)

        self.assertEqual([row.id for row in unresolved], [answer.id])

    def test_joint_individual_answer_credits_each_recognized_contributor(self):
        from viva_evaluator.services.scoring_service import ScoringService
        from viva_evaluator.services.session_reports import (
            unresolved_individual_answers,
        )

        answer = self._answer(1, self.individual_criterion, 8, student=None)
        attribution = AnswerAttribution.objects.create(
            answer=answer,
            session=self.session,
            outcome='uncertain',
            share=0.6,
        )
        AnswerContribution.objects.create(
            attribution=attribution,
            answer=answer,
            student=self.alice,
            share=0.6,
        )
        AnswerContribution.objects.create(
            attribution=attribution,
            answer=answer,
            student=self.bob,
            share=0.4,
        )

        alice = ScoringService.aggregate_student_score(self.session, self.alice)
        bob = ScoringService.aggregate_student_score(self.session, self.bob)

        self.assertEqual(alice['percentage'], 80.0)
        self.assertEqual(bob['percentage'], 80.0)
        self.assertEqual(unresolved_individual_answers(self.session), [])

    def test_assigned_examiner_can_approve_all_group_participant_reports(self):
        self._answer(1, self.group_criterion, 8, student=self.alice)
        self._answer(2, self.individual_criterion, 9, student=self.alice)
        self._answer(3, self.individual_criterion, 5, student=self.bob)

        response = self.client.post(
            f'/api/viva/sessions/{self.session.id}/approve-scores/',
            {},
            format='json',
        )

        self.assertEqual(response.status_code, 200, response.data)
        reports = list(
            self.session.summary_reports.order_by('student_id')
        )
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(report.is_published for report in reports))
        self.assertTrue(all(report.scores_status == 'approved' for report in reports))
        self.assertTrue(all(report.finalized_by_id == self.examiner.id for report in reports))

        # Retrying after a lost response is safe and returns the same success.
        retry = self.client.post(
            f'/api/viva/sessions/{self.session.id}/approve-scores/',
            {},
            format='json',
        )
        self.assertEqual(retry.status_code, 200, retry.data)

    def test_student_report_waits_for_examiner_approval(self):
        self.client.force_authenticate(self.alice.user)

        with patch(
            'viva_evaluator.services.reporting.generate_post_viva_report'
        ) as generate_report:
            response = self.client.get(
                f'/api/viva/sessions/{self.session.id}/report/'
            )

        self.assertEqual(response.status_code, 202, response.data)
        self.assertEqual(response.data['scores_status'], 'draft')
        generate_report.assert_not_called()

    def test_student_can_read_only_own_approved_report(self):
        SessionSummaryReport.objects.create(
            session=self.session,
            student=self.alice,
            total_final_score=85,
            scores_status=SessionSummaryReport.ScoresStatus.APPROVED,
            scores_approved_at=timezone.now(),
            is_published=True,
            published_at=timezone.now(),
        )
        SessionSummaryReport.objects.create(
            session=self.session,
            student=self.bob,
            total_final_score=65,
            scores_status=SessionSummaryReport.ScoresStatus.APPROVED,
            scores_approved_at=timezone.now(),
            is_published=True,
            published_at=timezone.now(),
        )
        self.client.force_authenticate(self.alice.user)

        with patch(
            'viva_evaluator.services.reporting.generate_post_viva_report'
        ) as generate_report:
            response = self.client.get(
                f'/api/viva/sessions/{self.session.id}/report/'
            )

        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(list(response.data['reports']), [str(self.alice.id)])
        self.assertEqual(
            response.data['data']['speaker_id'], str(self.alice.id)
        )
        self.assertEqual(response.data['data']['overall_score'], 0.85)
        self.assertEqual(response.data['data']['scores_status'], 'approved')
        generate_report.assert_not_called()


class DerivedKeyRetryTests(TestCase):
    """A retry after a turn that failed must not strand the student.

    The submit view derives the idempotency key from question and speaker, so
    a second press of Submit reuses it by construction. Speech recognition
    almost never reproduces byte-identical text, so treating that as a key
    reused with a different payload rejected every retry and left the student
    stuck on the question with a conflict error.
    """

    def setUp(self):
        self.project = Project.objects.create(project_name='Derived key retries')
        now = timezone.now()
        self.session = EvaluationSession.objects.create(
            project=self.project,
            scheduled_start=now - timedelta(minutes=5),
            scheduled_end=now + timedelta(minutes=55),
            actual_start=now - timedelta(minutes=4),
            status=EvaluationSession.Status.IN_PROGRESS,
        )
        self.question = VivaQuestion.objects.create(
            session=self.session,
            project=self.project,
            question_text='Explain your caching strategy.',
            question_order=1,
        )
        self.speaker = 'student:derived-key'
        self.key = f'answer:{self.question.id}:{self.speaker}'
        self.first_hash = request_fingerprint(
            answer_text='We cache reads', speech_metrics=None, speaker_id=self.speaker,
        )
        self.second_hash = request_fingerprint(
            answer_text='We cache reads at the edge',
            speech_metrics=None,
            speaker_id=self.speaker,
        )

    def _acquire(self, request_hash, strict_hash=False):
        return acquire_claim(
            session=self.session,
            question=self.question,
            speaker=self.speaker,
            idempotency_key=self.key,
            request_hash=request_hash,
            strict_hash=strict_hash,
        )

    def test_retry_after_a_failure_that_saved_the_answer_is_allowed(self):
        claim = self._acquire(self.first_hash).claim
        VivaAnswer.objects.create(
            question=self.question,
            deduplication_key=self.speaker,
            transcribed_answer='We cache reads',
        )
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            status=VivaAnswerProcessingClaim.Status.FAILED,
            error_code='pipeline_error',
        )

        result = self._acquire(self.second_hash)

        # Persistence dedupes on the speaker key, so re-running only finishes
        # the turn; it never writes a second answer or edits the first.
        self.assertEqual(result.action, 'process')

    def test_resubmitting_an_answered_question_replays_instead_of_erroring(self):
        claim = self._acquire(self.first_hash).claim
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            status=VivaAnswerProcessingClaim.Status.COMPLETED,
            response_payload={'answer_saved': True},
        )

        result = self._acquire(self.second_hash)

        self.assertEqual(result.action, 'replay')
        self.assertEqual(result.claim.response_payload, {'answer_saved': True})

    def test_a_client_chosen_key_still_may_not_change_its_payload(self):
        claim = self._acquire(self.first_hash, strict_hash=True).claim
        VivaAnswerProcessingClaim.objects.filter(pk=claim.pk).update(
            status=VivaAnswerProcessingClaim.Status.COMPLETED,
            response_payload={'answer_saved': True},
        )

        with self.assertRaises(IdempotencyConflict):
            self._acquire(self.second_hash, strict_hash=True)


class IndividualVivaScoringTests(TestCase):
    """A scored individual viva must not total zero.

    Two failures combined to report 0/100 on a viva whose answers were scored.
    The session-level report passes no student, and filtering individual
    criteria by contribution share matched nothing. Separately, attribution
    cleared the answer's owner whenever a window carried no confident
    evidence, which is the normal case for a remote individual viva.
    """

    def setUp(self):
        self.project = Project.objects.create(project_name='Individual viva scoring')
        self.student = self._student('solo@example.com', 'REG-SOLO')
        self.session = EvaluationSession.objects.create(
            project=self.project,
            student=self.student,
            scheduled_start=timezone.now(),
            scheduled_end=timezone.now() + timedelta(hours=1),
            status=EvaluationSession.Status.COMPLETED,
        )
        category = RubricCategory.objects.create(
            project=self.project,
            category_name='Knowledge',
            weight_percentage=100,
        )
        self.criterion = RubricCriteria.objects.create(
            category=category,
            criteria_name='Individual understanding',
            max_score=10,
            is_individual=True,
        )

    @staticmethod
    def _student(email, registration_number):
        user = User.objects.create_user(
            email=email,
            password='test-password',
            full_name='Solo Student',
            role=User.Role.STUDENT,
        )
        return StudentProfile.objects.create(
            user=user, registration_number=registration_number,
        )

    def _answer(self, order, score, *, student, dedup):
        question = VivaQuestion.objects.create(
            session=self.session,
            project=self.project,
            question_text=f'Question {order}',
            question_order=order,
        )
        VivaQuestionExtension.objects.create(question=question, criteria=self.criterion)
        return VivaAnswer.objects.create(
            question=question,
            student=student,
            deduplication_key=dedup,
            ai_answer_score=score,
            transcribed_answer='Test answer',
        )

    def test_session_level_report_scores_individual_criteria(self):
        from viva_evaluator.services.scoring_service import ScoringService

        self._answer(1, 2.2, student=self.student, dedup=f'student:{self.student.id}')
        self._answer(2, 3.1, student=self.student, dedup=f'student:{self.student.id}')

        result = ScoringService.aggregate_student_score(self.session, None)

        self.assertEqual(result['percentage'], 26.5)
        self.assertNotEqual(result['grade'], 'N/A')

    def test_an_answer_whose_owner_was_cleared_still_scores(self):
        from viva_evaluator.services.scoring_service import ScoringService

        # The owner survives in deduplication_key, which attribution never
        # rewrites, so historic answers still reach the right student.
        self._answer(1, 2.2, student=None, dedup=f'student:{self.student.id}')
        self._answer(2, 3.1, student=None, dedup=f'student:{self.student.id}')

        result = ScoringService.aggregate_student_score(self.session, self.student)

        self.assertEqual(result['percentage'], 26.5)

    def test_attribution_never_clears_an_owner_it_cannot_improve_on(self):
        from unittest.mock import patch

        from attribution.services.engine import record_attribution

        answer = self._answer(
            1, 2.2, student=self.student, dedup=f'student:{self.student.id}',
        )
        inconclusive = type('Decision', (), {
            'student_id': None, 'unknown_key': None, 'confidence': 0.0,
            'breakdown': {}, 'co_speakers': [], 'method': 'none',
        })()

        with patch('attribution.services.engine.is_enabled', return_value=True):
            record_attribution(answer, self.session, inconclusive)

        answer.refresh_from_db()
        self.assertEqual(answer.student_id, self.student.id)
