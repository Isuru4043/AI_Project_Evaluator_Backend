import math
from django.db.models import Prefetch

class ScoringService:
    @staticmethod
    def get_effective_score_for_answer(answer):
        """
        Gets the effective score for a VivaAnswer.
        Prioritizes examiner_override_score if it exists.
        Returns None if both override and AI score are None (unscored/clarification).
        Returns score on a 0.0 - 10.0 scale.
        """
        if answer.examiner_override_score is not None:
            return float(answer.examiner_override_score)
        
        if answer.ai_answer_score is not None:
            return float(answer.ai_answer_score)
            
        return None

    @staticmethod
    def calculate_grade(percentage_0_to_100):
        """
        Standardized grade bracket mapping.
        """
        if percentage_0_to_100 >= 75: return 'A'
        if percentage_0_to_100 >= 65: return 'B'
        if percentage_0_to_100 >= 50: return 'C'
        if percentage_0_to_100 >= 35: return 'S'
        return 'F'

    @staticmethod
    def aggregate_student_score(session, student_profile):
        """
        Calculates the overall score and grade for a specific student in a session.
        Respects `is_individual` on RubricCriteria.
        Returns:
            {
                'total_possible': float,
                'total_earned': float,
                'percentage': float, # 0-100
                'grade': str,
                'per_criteria': dict # {criteria_id: score}
            }
        """
        total_possible = 0.0
        total_earned = 0.0
        
        answers = []
        for q in session.viva_questions.prefetch_related('answers', 'extension__criteria').all():
            for a in q.answers.all():
                try:
                    q_ext = q.extension
                    if q_ext and q_ext.criteria:
                        if q_ext.criteria.is_individual:
                            if a.student == student_profile:
                                answers.append((a, q_ext.criteria))
                        else:
                            # Group criteria: applies to all students in the session
                            answers.append((a, q_ext.criteria))
                except Exception:
                    pass

        if not answers:
            return {
                'total_possible': 0.0,
                'total_earned': 0.0,
                'percentage': 0.0,
                'grade': 'N/A',
                'per_criteria': {}
            }
            
        # Group answers by criteria
        criteria_scores = {}
        for a, crit in answers:
            score = ScoringService.get_effective_score_for_answer(a)
            if score is None:
                continue # Skip unscored clarifications
                
            if crit.id not in criteria_scores:
                criteria_scores[crit.id] = {'earned': 0.0, 'samples': 0, 'weight': crit.weight_in_category or 1.0, 'max': crit.max_score}
                
            # Assume score is out of 10. We normalize to max_score.
            normalized_score = (score / 10.0) * float(crit.max_score)
            criteria_scores[crit.id]['earned'] += normalized_score
            criteria_scores[crit.id]['samples'] += 1
            
        if not criteria_scores:
            return {
                'total_possible': 0.0,
                'total_earned': 0.0,
                'percentage': 0.0,
                'grade': 'N/A',
                'per_criteria': {}
            }
            
        per_criteria_result = {}
        for crit_id, data in criteria_scores.items():
            if data['samples'] > 0:
                avg_earned = data['earned'] / data['samples']
                total_earned += avg_earned
                total_possible += float(data['max'])
                per_criteria_result[str(crit_id)] = avg_earned
                
        percentage = 0.0
        if total_possible > 0:
            percentage = round((total_earned / total_possible) * 100, 2)
            
        return {
            'total_possible': total_possible,
            'total_earned': total_earned,
            'percentage': percentage,
            'grade': ScoringService.calculate_grade(percentage),
            'per_criteria': per_criteria_result
        }
