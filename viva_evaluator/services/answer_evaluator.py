import json
from google import genai
from django.conf import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)
MODEL = "gemini-2.5-flash-lite"


def evaluate_answer(
    question_text: str,
    student_answer: str,
    criteria_name: str,
    criteria_description: str,
    report_text: str,
    current_difficulty: str,
) -> dict:
    """
    Evaluates a student's viva answer semantically using the LLM.

    Does NOT do simple keyword matching — it judges whether the student
    actually understands the concept being asked about, based on the
    meaning of their answer relative to the rubric criterion.

    Args:
        question_text:        The question that was asked.
        student_answer:       What the student said.
        criteria_name:        The rubric criterion being evaluated.
        criteria_description: Examiner's description of the criterion.
        report_text:          The student's submitted report (for context).
        current_difficulty:   Current difficulty level ('easy'/'medium'/'hard').

    Returns:
        dict with keys:
            - llm_score (float 0-10)
            - llm_reasoning (str)
            - next_difficulty ('lower'/'same'/'higher')
            - strengths (str)
            - gaps (str)
    """

    prompt = f"""
You are an academic viva examiner evaluating a student's spoken answer.

RUBRIC CRITERION:
Name: {criteria_name}
Description: {criteria_description}

STUDENT'S REPORT CONTEXT (what they submitted):
{report_text[:2000]}

QUESTION ASKED:
{question_text}

STUDENT'S ANSWER:
{student_answer}

CURRENT DIFFICULTY LEVEL: {current_difficulty}

EVALUATION INSTRUCTIONS:
- Judge the answer based on conceptual understanding, not exact wording
- A correct answer in different words is still correct
- Consider whether the student understands the WHY behind their choices
- Be strict but fair — partial understanding should get partial credit

Score the answer from 0 to 10:
- 0-3: Student shows little to no understanding
- 4-6: Student shows partial understanding with gaps
- 7-8: Student shows solid understanding with minor gaps
- 9-10: Student shows excellent, deep understanding

Based on the score, decide next difficulty:
- Score 0-4: next difficulty should be "lower"
- Score 5-7: next difficulty should be "same"
- Score 8-10: next difficulty should be "higher"

Respond in this exact JSON format with no extra text or markdown:
{{
    "llm_score": 7.5,
    "llm_reasoning": "The student demonstrated understanding of... however they missed...",
    "next_difficulty": "same",
    "strengths": "What the student got right",
    "gaps": "What the student missed or got wrong"
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return _parse_json_response(response.text)


def _parse_json_response(response_text: str) -> dict:
    """Safely parses JSON response from Gemini."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "llm_score": 5.0,
            "llm_reasoning": text,
            "next_difficulty": "same",
            "strengths": "",
            "gaps": "",
        }