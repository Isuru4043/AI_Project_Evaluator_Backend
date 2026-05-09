import json
from google import genai
from django.conf import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

# How many questions to ask per rubric criterion
QUESTIONS_PER_CRITERION = 3


def get_session_context(session_id: str) -> dict:
    """
    Loads the current state of a viva session from the database.

    Returns everything the adaptive loop needs to decide what to do next:
    - Which criteria have been covered
    - Which criterion is currently active
    - What questions and answers have happened so far
    - Current difficulty level

    Args:
        session_id: UUID of the EvaluationSession

    Returns:
        dict with session state
    """
    from core.models import EvaluationSession, RubricCategory
    from viva_evaluator.models import VivaQuestionExtension

    session = EvaluationSession.objects.get(id=session_id)
    project = session.project

    # Get all rubric criteria for this project in order
    all_criteria = []
    for category in project.rubric_categories.all().order_by('id'):
        for criterion in category.criteria.all().order_by('id'):
            all_criteria.append({
                'id': str(criterion.id),
                'name': criterion.criteria_name,
                'description': criterion.description or '',
                'max_score': float(criterion.max_score),
                'category': category.category_name,
            })

    # Get all questions asked so far in this session
    asked_questions = session.viva_questions.all().order_by('question_order')

    # Figure out which criteria have been fully covered
    covered_criteria_ids = set()
    current_criterion_id = None
    current_criterion_question_count = 0
    current_difficulty = 'medium'

    for question in asked_questions:
        try:
            ext = question.extension
            crit_id = str(ext.criteria_id) if ext.criteria_id else None
            current_difficulty = ext.difficulty_level

            if crit_id:
                # Count how many questions were asked for each criterion
                criterion_questions = [
                    q for q in asked_questions
                    if hasattr(q, 'extension') and
                    str(q.extension.criteria_id) == crit_id
                ]
                if len(criterion_questions) >= QUESTIONS_PER_CRITERION:
                    covered_criteria_ids.add(crit_id)
                else:
                    current_criterion_id = crit_id
                    current_criterion_question_count = len(criterion_questions)
        except Exception:
            pass

    # Get the next difficulty signal from the last answer
    last_question = asked_questions.last()
    if last_question:
        last_answers = last_question.answers.all()
        if last_answers.exists():
            last_answer = last_answers.last()
            try:
                next_diff = last_answer.extension.next_difficulty_signal
                if next_diff == 'higher':
                    current_difficulty = _escalate_difficulty(current_difficulty)
                elif next_diff == 'lower':
                    current_difficulty = _deescalate_difficulty(current_difficulty)
            except Exception:
                pass

    # Find the next uncovered criterion
    next_criterion = None
    for criterion in all_criteria:
        if criterion['id'] not in covered_criteria_ids:
            if current_criterion_id is None or criterion['id'] == current_criterion_id:
                next_criterion = criterion
                break

    # If current criterion is done, move to next
    if current_criterion_question_count >= QUESTIONS_PER_CRITERION:
        for criterion in all_criteria:
            if criterion['id'] not in covered_criteria_ids:
                next_criterion = criterion
                current_criterion_question_count = 0
                current_difficulty = 'medium'  # Reset difficulty for new criterion
                break

    return {
        'session': session,
        'all_criteria': all_criteria,
        'covered_criteria_ids': list(covered_criteria_ids),
        'next_criterion': next_criterion,
        'current_difficulty': current_difficulty,
        'current_criterion_question_count': current_criterion_question_count,
        'total_questions_asked': asked_questions.count(),
        'is_complete': next_criterion is None,
    }


def generate_session_report(session_id: str) -> dict:
    """
    Generates the final XAI report for a completed viva session.

    Aggregates all QA turns, scores per criterion, and asks the LLM
    to write an overall explanation of the student's performance.

    Args:
        session_id: UUID of the EvaluationSession

    Returns:
        dict with full session report
    """
    from core.models import EvaluationSession, RubricCriteria
    from viva_evaluator.models import VivaQuestionExtension, VivaAnswerExtension

    session = EvaluationSession.objects.get(id=session_id)
    questions = session.viva_questions.all().order_by('question_order')

    # Build a transcript and per-criterion scores
    transcript = []
    criteria_scores = {}

    for question in questions:
        q_data = {
            'question': question.question_text,
            'blooms_level': question.blooms_level,
            'answers': [],
        }

        try:
            ext = question.extension
            criteria_id = str(ext.criteria_id) if ext.criteria_id else None
            criteria_name = ext.criteria.criteria_name if ext.criteria else 'General'
        except Exception:
            criteria_id = None
            criteria_name = 'General'

        for answer in question.answers.all():
            a_data = {
                'answer': answer.transcribed_answer,
                'score': float(answer.ai_answer_score) if answer.ai_answer_score else 0,
            }
            try:
                a_data['reasoning'] = answer.extension.llm_reasoning
                a_data['next_difficulty'] = answer.extension.next_difficulty_signal
            except Exception:
                pass

            q_data['answers'].append(a_data)

            # Accumulate scores per criterion
            if criteria_id:
                if criteria_id not in criteria_scores:
                    criteria_scores[criteria_id] = {
                        'name': criteria_name,
                        'scores': [],
                    }
                if answer.ai_answer_score:
                    criteria_scores[criteria_id]['scores'].append(
                        float(answer.ai_answer_score)
                    )

        transcript.append(q_data)

    # Average score per criterion
    per_criterion_summary = {}
    overall_scores = []
    for crit_id, data in criteria_scores.items():
        if data['scores']:
            avg = sum(data['scores']) / len(data['scores'])
            per_criterion_summary[data['name']] = round(avg, 2)
            overall_scores.extend(data['scores'])

    overall_avg = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0

    # Ask LLM for overall XAI explanation
    transcript_text = json.dumps(transcript, indent=2)

    prompt = f"""
You are an academic examiner writing a final evaluation report for a student's viva session.

PER-CRITERION SCORES (out of 10):
{json.dumps(per_criterion_summary, indent=2)}

FULL QA TRANSCRIPT:
{transcript_text[:4000]}

TASK:
Write a professional, honest evaluation report that includes:
1. Overall performance summary (2-3 sentences)
2. Key strengths demonstrated
3. Key gaps or weaknesses identified
4. Recommendation for the examiner

Be specific — reference actual answers from the transcript where relevant.
Be fair but honest. This is to help the examiner make a final decision.

Respond in this exact JSON format with no extra text or markdown:
{{
    "overall_summary": "...",
    "strengths": "...",
    "gaps": "...",
    "examiner_recommendation": "..."
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    xai_report = _parse_json_response(response.text)

    return {
        'session_id': session_id,
        'overall_score': overall_avg,
        'per_criterion_scores': per_criterion_summary,
        'xai_report': xai_report,
        'transcript': transcript,
    }


def _escalate_difficulty(current: str) -> str:
    progression = ['easy', 'medium', 'hard']
    idx = progression.index(current) if current in progression else 1
    return progression[min(idx + 1, 2)]


def _deescalate_difficulty(current: str) -> str:
    progression = ['easy', 'medium', 'hard']
    idx = progression.index(current) if current in progression else 1
    return progression[max(idx - 1, 0)]


def _parse_json_response(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"overall_summary": text, "strengths": "", "gaps": "", "examiner_recommendation": ""}