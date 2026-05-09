import json
from google import genai
from django.conf import settings

# Initialize client using the new SDK
client = genai.Client(api_key=settings.GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

DIFFICULTY_TO_BLOOMS = {
    "easy":   ["Remember", "Understand"],
    "medium": ["Apply", "Analyze"],
    "hard":   ["Evaluate", "Create"],
}


def generate_first_question(
    report_text: str,
    criteria_name: str,
    criteria_description: str,
    difficulty: str = "medium",
) -> dict:
    """
    Generates the opening viva question for a given rubric criterion.
    """
    blooms_level = DIFFICULTY_TO_BLOOMS.get(difficulty, ["Apply", "Analyze"])[0]

    prompt = f"""
You are an academic viva examiner evaluating a student's final year project report.

RUBRIC CRITERION:
Name: {criteria_name}
Description: {criteria_description}

STUDENT REPORT (excerpt):
{report_text[:3000]}

TASK:
Generate ONE clear, focused viva question that:
- Directly relates to the criterion above
- Tests the student at Bloom's Taxonomy level: {blooms_level}
- Is based on what the student has actually written in their report
- Cannot be answered with a simple yes or no
- Is phrased naturally as a spoken question

Respond in this exact JSON format with no extra text or markdown:
{{
    "question_text": "your question here",
    "blooms_level": "{blooms_level}",
    "difficulty": "{difficulty}"
}}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    return _parse_json_response(response.text)


def generate_followup_question(
    report_text: str,
    criteria_name: str,
    criteria_description: str,
    previous_question: str,
    previous_answer: str,
    next_difficulty: str,
    question_number: int,
) -> dict:
    """
    Generates a follow-up question based on the student's previous answer.
    """
    blooms_level = DIFFICULTY_TO_BLOOMS.get(next_difficulty, ["Apply", "Analyze"])[0]

    prompt = f"""
You are an academic viva examiner conducting an oral examination.

RUBRIC CRITERION:
Name: {criteria_name}
Description: {criteria_description}

STUDENT REPORT (excerpt):
{report_text[:2000]}

CONVERSATION SO FAR:
Question asked: {previous_question}
Student answered: {previous_answer}

TASK:
Generate ONE follow-up viva question that:
- Builds naturally on the student's answer above
- Targets Bloom's Taxonomy level: {blooms_level}
- Probes deeper if the answer was good, or clarifies if the answer was weak
- Stays focused on the criterion: {criteria_name}
- Is different from the previous question
- Cannot be answered with a simple yes or no

This is question number {question_number} for this criterion.

Respond in this exact JSON format with no extra text or markdown:
{{
    "question_text": "your question here",
    "blooms_level": "{blooms_level}",
    "difficulty": "{next_difficulty}"
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
            "question_text": text,
            "blooms_level": "Understand",
            "difficulty": "medium",
        }