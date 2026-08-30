from datetime import timedelta

from django.test import TestCase
from django.utils import timezone

from core.models import EvaluationSession, Project, VivaAnswer, VivaQuestion
from viva_evaluator.models import VivaAnswerProcessingClaim
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
