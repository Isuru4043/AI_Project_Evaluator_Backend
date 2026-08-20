from dataclasses import FrozenInstanceError
from unittest import TestCase

from viva_evaluator.services.pipeline.contracts import (
    ValidatedQuestion,
    VivaTopicRef,
)


class PipelineContractTests(TestCase):
    def test_topic_contract_round_trips_legacy_pipeline_shape(self):
        raw = {
            "topic_name": "Architecture",
            "topic_focus": "Service boundaries",
            "source_criteria_ids": ["criterion-1", "criterion-2"],
            "suggested_questions": 3,
        }

        topic = VivaTopicRef.from_mapping(raw)

        self.assertEqual(topic.name, "Architecture")
        self.assertEqual(topic.criterion_ids, ("criterion-1", "criterion-2"))
        self.assertEqual(topic.to_pipeline_dict(), raw)

    def test_contracts_are_immutable_at_the_stage_boundary(self):
        topic = VivaTopicRef(
            name="Testing",
            focus="Test strategy",
            criterion_ids=("criterion-1",),
            suggested_questions=2,
        )

        with self.assertRaises(FrozenInstanceError):
            topic.name = "Changed"

    def test_validated_question_has_stable_validation_defaults(self):
        question = ValidatedQuestion(
            question_text="Why did you choose this service boundary?",
            blooms_level="Analyze",
            difficulty="medium",
            tier1_passed=True,
        )

        self.assertEqual(question.tier1_failures, ())
        self.assertIsNone(question.critic_passed)
        self.assertEqual(question.attempts, 1)

