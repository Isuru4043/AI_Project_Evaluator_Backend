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
    def aggregate_student_score(
        session,
        student_profile,
        *,
        use_examiner_overrides=True,
    ):
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
                            # An individual criterion is awarded only to the
                            # resolved primary answerer. Contribution shares
                            # remain useful review evidence, but are not marks:
                            # normalising a partial share by that same share
                            # gave every contributor the whole answer score.
                            if (
                                student_profile is not None
                                and a.student_id == student_profile.id
                            ):
                                answers.append((a, q_ext.criteria, 1.0))
                        else:
                            # Group criteria measure the group, so every member
                            # carries the whole answer regardless of who spoke.
                            answers.append((a, q_ext.criteria, 1.0))
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
        for a, crit, weight in answers:
            score = (
                ScoringService.get_effective_score_for_answer(a)
                if use_examiner_overrides
                else (
                    float(a.ai_answer_score)
                    if a.ai_answer_score is not None
                    else None
                )
            )
            if score is None:
                continue # Skip unscored clarifications

            if crit.id not in criteria_scores:
                criteria_scores[crit.id] = {'earned': 0.0, 'samples': 0, 'weight': crit.weight_in_category or 1.0, 'max': crit.max_score}

            # Assume score is out of 10. We normalize to max_score.
            normalized_score = (score / 10.0) * float(crit.max_score)
            # Both sides carry the weight, so the criterion mean stays a mean:
            # an answer half-spoken counts half as much toward this student's
            # average, without dragging the average itself down.
            criteria_scores[crit.id]['earned'] += normalized_score * weight
            criteria_scores[crit.id]['samples'] += weight

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
