import os
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.agents.questioner import (
    QuestionerInput,
    _build_prompt,
    generate_anchored_question,
)
from viva_evaluator.services.agents.tier1_validator import Tier1Result


class QuestionerPlanningTests(TestCase):
    def _input(self):
        return QuestionerInput(
            criterion_name="Architecture",
            criterion_description="Service boundaries",
            retrieved_chunks=[{
                "text": "The project separates API services.",
                "evidence_id": "submission:test:chunk:1",
            }],
            difficulty="hard",
            target_bloom="Create",
            socratic_intent="probing_evidence",
            intent_prompt_hint="Ask for concrete implementation evidence.",
            question_hints=["Focus on the chosen boundary"],
        )

    def test_prompt_contains_exact_strategy_and_examiner_hints(self):
        prompt = _build_prompt(self._input(), "Create", retry_reason=None)

        self.assertIn("TARGET BLOOM'S LEVEL: Create", prompt)
        self.assertIn("Intent: probing_evidence", prompt)
        self.assertIn("Ask for concrete implementation evidence.", prompt)
        self.assertIn("Focus on the chosen boundary", prompt)
        self.assertIn("source_reference_ids", prompt)
        self.assertIn('"target_bloom": "Create"', prompt)
        self.assertIn('"socratic_intent": "probing_evidence"', prompt)
        self.assertIn("submission:test:chunk:1", prompt)

    @patch(
        "viva_evaluator.services.pipeline.stages.question_validation.validate_question",
        return_value=Tier1Result(
            passed=True,
            failures=[],
            similarity_to_recent=0.1,
            word_count=20,
        ),
    )
    @patch(
        "viva_evaluator.services.llm_service.llm_call",
        return_value={
            "question_text": "In your design, how would you rebuild this boundary for greater resilience?",
            "blooms_level": "Evaluate",
            "difficulty": "medium",
            "source_reference_ids": ["submission:test:chunk:1"],
            "target_bloom": "Create",
            "socratic_intent": "probing_evidence",
        },
    )
    def test_planned_bloom_and_difficulty_override_llm_metadata(
        self,
        _llm_call,
        _validate,
    ):
        result = generate_anchored_question(
            self._input(),
            enable_critic=False,
        )

        self.assertEqual(result["blooms_level"], "Create")
        self.assertEqual(result["difficulty"], "hard")
        self.assertEqual(
            result["source_reference_ids"],
            ["submission:test:chunk:1"],
        )
        self.assertEqual(result["socratic_intent"], "probing_evidence")
