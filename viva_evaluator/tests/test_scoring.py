from unittest import TestCase
from unittest.mock import MagicMock
from decimal import Decimal
from viva_evaluator.services.scoring_service import ScoringService

class ScoringServiceTests(TestCase):
    def test_calculate_grade(self):
        self.assertEqual(ScoringService.calculate_grade(85), 'A')
        self.assertEqual(ScoringService.calculate_grade(75), 'A')
        self.assertEqual(ScoringService.calculate_grade(74.9), 'B')
        self.assertEqual(ScoringService.calculate_grade(65), 'B')
        self.assertEqual(ScoringService.calculate_grade(50), 'C')
        self.assertEqual(ScoringService.calculate_grade(35), 'S')
        self.assertEqual(ScoringService.calculate_grade(34), 'F')
        self.assertEqual(ScoringService.calculate_grade(0), 'F')
        self.assertEqual(ScoringService.calculate_grade(100), 'A')

    def test_get_effective_score_for_answer(self):
        ans_only_ai = MagicMock()
        ans_only_ai.examiner_override_score = None
        ans_only_ai.ai_answer_score = Decimal('8.5')

        ans_override = MagicMock()
        ans_override.examiner_override_score = Decimal('9.0')
        ans_override.ai_answer_score = Decimal('6.0')

        ans_empty = MagicMock()
        ans_empty.examiner_override_score = None
        ans_empty.ai_answer_score = None

        self.assertEqual(ScoringService.get_effective_score_for_answer(ans_only_ai), 8.5)
        self.assertEqual(ScoringService.get_effective_score_for_answer(ans_override), 9.0)
        self.assertIsNone(ScoringService.get_effective_score_for_answer(ans_empty))

class MockCriteria:
    def __init__(self, id_val, name, max_score, is_individual):
        self.id = id_val
        self.criteria_name = name
        self.max_score = Decimal(str(max_score))
        self.is_individual = is_individual
        self.weight_in_category = 1.0

class MockQuestion:
    def __init__(self, extension_criteria, answers):
        self.extension = MagicMock()
        self.extension.criteria = extension_criteria
        self.answers_mock = answers
        
    @property
    def answers(self):
        return self

    def all(self):
        return self.answers_mock

class MockSession:
    def __init__(self, questions):
        self.viva_questions_mock = questions
        
    @property
    def viva_questions(self):
        return self

    def prefetch_related(self, *args):
        return self

    def all(self):
        return self.viva_questions_mock

def test_aggregate_student_score():
    # 1. Setup criteria
    crit_group = MockCriteria('1', 'Group Architecture', 10.0, False)
    crit_indiv = MockCriteria('2', 'Individual Contribution', 10.0, True)
    
    # 2. Setup answers
    # Answer 1: Group question, scored 8.0 by AI
    ans1 = MagicMock()
    ans1.student = None  # Group answer
    ans1.student_id = None
    ans1.ai_answer_score = Decimal('8.0')
    ans1.examiner_override_score = None
    
    # Answer 2: Individual question, scored 5.0 by AI, but overridden to 7.0
    ans2 = MagicMock()
    ans2.student = MagicMock()
    ans2.student.id = 'student_uuid'
    ans2.student_id = 'student_uuid'
    ans2.ai_answer_score = Decimal('5.0')
    ans2.examiner_override_score = Decimal('7.0')
    
    # Answer 3: Clarification re-ask, unscored (should be ignored)
    ans3 = MagicMock()
    ans3.student = None
    ans3.student_id = None
    ans3.ai_answer_score = None
    ans3.examiner_override_score = None
    
    # Answer 4: Individual question answered by ANOTHER student
    ans4 = MagicMock()
    ans4.student = MagicMock()
    ans4.student.id = 'other_uuid'
    ans4.student_id = 'other_uuid'
    ans4.ai_answer_score = Decimal('10.0')
    ans4.examiner_override_score = None
    
    # 3. Setup questions
    q1 = MockQuestion(crit_group, [ans1, ans3])
    q2 = MockQuestion(crit_indiv, [ans2, ans4])
    
    session = MockSession([q1, q2])
    
    # Test aggregation for 'student_uuid'
    student = ans2.student
    
    result = ScoringService.aggregate_student_score(session, student)
    
    # Expected:
    # q1 (group) -> ans1 applies -> 8.0/10
    # q2 (indiv) -> ans2 applies -> 7.0/10
    # ans3 is skipped (no score)
    # ans4 is skipped (wrong student)
    
    assert result['total_possible'] == 20.0
    assert result['total_earned'] == 15.0
    assert result['percentage'] == 75.0
    assert result['grade'] == 'A'
    assert result['per_criteria'] == {'1': 8.0, '2': 7.0}

def test_aggregate_student_score_group_mode():
    crit_group = MockCriteria('1', 'Group Arch', 10.0, False)
    
    ans1 = MagicMock()
    ans1.student = None
    ans1.student_id = None
    ans1.ai_answer_score = Decimal('5.0')
    ans1.examiner_override_score = None
    
    q1 = MockQuestion(crit_group, [ans1])
    session = MockSession([q1])
    
    # Test for "group" (student is None)
    result = ScoringService.aggregate_student_score(session, None)
    
    assert result['total_possible'] == 10.0
    assert result['total_earned'] == 5.0
    assert result['percentage'] == 50.0
    assert result['grade'] == 'C'
    assert result['per_criteria'] == {'1': 5.0}

def test_aggregate_empty_session():
    session = MockSession([])
    result = ScoringService.aggregate_student_score(session, None)
    
    assert result['total_possible'] == 0.0
    assert result['total_earned'] == 0.0
    assert result['percentage'] == 0.0
    assert result['grade'] == 'N/A'
    assert result['per_criteria'] == {}


class ScoringAggregationTests(TestCase):
    def test_student_score(self):
        test_aggregate_student_score()

    def test_group_mode(self):
        test_aggregate_student_score_group_mode()

    def test_empty_session(self):
        test_aggregate_empty_session()

    def test_individual_answer_is_not_shared_with_a_contributor(self):
        criterion = MockCriteria('individual', 'Individual Knowledge', 10.0, True)
        owner = MagicMock(id='owner')
        contributor = MagicMock(id='contributor')
        answer = MagicMock(
            student=owner,
            student_id='owner',
            ai_answer_score=Decimal('8.0'),
            examiner_override_score=None,
        )
        session = MockSession([MockQuestion(criterion, [answer])])

        owner_result = ScoringService.aggregate_student_score(session, owner)
        contributor_result = ScoringService.aggregate_student_score(
            session, contributor,
        )

        self.assertEqual(owner_result['percentage'], 80.0)
        self.assertEqual(contributor_result['grade'], 'N/A')

    def test_ai_total_can_be_calculated_without_examiner_override(self):
        criterion = MockCriteria('individual', 'Individual Knowledge', 10.0, True)
        student = MagicMock(id='student')
        answer = MagicMock(
            student=student,
            student_id='student',
            ai_answer_score=Decimal('5.0'),
            examiner_override_score=Decimal('9.0'),
        )
        session = MockSession([MockQuestion(criterion, [answer])])

        result = ScoringService.aggregate_student_score(
            session,
            student,
            use_examiner_overrides=False,
        )

        self.assertEqual(result['percentage'], 50.0)
