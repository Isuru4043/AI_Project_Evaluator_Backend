import json
from google import genai
from django.conf import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"


def extract_rubric_from_text(rubric_text: str) -> dict:
    """
    Sends extracted rubric text to Gemini and gets back a structured rubric.

    Args:
        rubric_text: Plain text extracted from the examiner's rubric PDF/DOCX.

    Returns:
        dict with the full structured rubric ready for preview and saving.
    """

    prompt = f"""
You are an academic system that reads university project rubric documents and extracts their structure.

RUBRIC DOCUMENT TEXT:
{rubric_text[:6000]}

TASK:
Extract the full rubric structure from the above text. Identify:
- Project/module name and description
- Rubric categories (main sections) with their weights
- Individual criteria within each category with their scores and descriptions
- Suggest how many viva questions should be asked per criterion based on its complexity and weight (between 2 and 5)

If the document does not clearly specify weights or scores, make reasonable academic assumptions and note them.

Respond in this exact JSON format with no extra text or markdown:
{{
    "project_name": "name of the project or module",
    "project_description": "brief description of what this project is about",
    "is_group_project": false,
    "academic_year": "2024/2025",
    "rubric_categories": [
        {{
            "category_name": "Category Name",
            "weight_percentage": 30.00,
            "description": "What this category evaluates",
            "criteria": [
                {{
                    "criteria_name": "Criterion Name",
                    "max_score": 10.00,
                    "weight_in_category": 50.00,
                    "description": "What this criterion specifically looks for",
                    "questions_to_ask": 3,
                    "question_hints": [
                        {{
                            "hint_text": "A suggested question area or topic to probe",
                            "order": 1
                        }}
                    ]
                }}
            ]
        }}
    ],
    "extraction_notes": "Any assumptions made or things the examiner should verify"
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
            "error": "Could not parse rubric structure from document.",
            "raw_response": text,
        }