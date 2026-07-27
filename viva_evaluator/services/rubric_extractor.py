import json
from django.conf import settings

from AI_Evaluator_Backend.llm import get_llm


MODEL = settings.GEMINI_MODEL


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

    response = get_llm().models.generate_content(
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


def generate_viva_grouping(project, max_questions: int) -> dict:
    """
    Groups a project's rubric criteria into viva topics to fit within the `max_questions` budget.
    Checks the RubricGroupingCache first. If missing, calls Gemini and saves to cache.
    
    Args:
        project: The Project instance.
        max_questions: The maximum total questions budget.
        
    Returns:
        The RubricGroupingCache instance.
    """
    from core.models import RubricGroupingCache, RubricCriteria
    
    # 1. Check cache
    cache = RubricGroupingCache.objects.filter(project=project, max_questions=max_questions).first()
    if cache:
        return cache
        
    # 2. Extract criteria info
    criteria = RubricCriteria.objects.filter(category__project=project).select_related('category')
    if not criteria.exists():
        raise ValueError("Project has no rubric criteria.")
        
    criteria_list = []
    for c in criteria:
        criteria_list.append({
            "id": str(c.id),
            "name": c.criteria_name,
            "category": c.category.category_name,
            "weight_in_category": float(c.weight_in_category or 0),
            "description": c.description,
        })
        
    # 3. Call Gemini
    prompt = f"""
You are an academic system that structures viva (oral examination) sessions.

Here are {len(criteria_list)} rubric criteria for project '{project.project_name}':
{json.dumps(criteria_list, indent=2)}

The viva session is limited to a MAXIMUM of {max_questions} total questions.
TASK: Group these criteria into "Viva Topics" and intelligently distribute the {max_questions} budget.

Rules:
- Each topic should cover 1 to 4 semantically related criteria.
- Topics should have a natural, open-ended viva question focus.
- Distribute the suggested questions proportionally based on the criteria's combined weight or complexity.
- IMPORTANT: The sum of `suggested_questions` across ALL topics MUST equal exactly {max_questions}.
- Every criterion MUST be included in exactly one topic.

Respond in this EXACT JSON format with no extra text or markdown:
{{
  "viva_topics": [
    {{
      "topic_name": "Topic Name",
      "source_criteria_ids": ["uuid1", "uuid2"],
      "combined_weight_pct": 15.5,
      "suggested_questions": 2,
      "topic_focus": "Description of what to ask about this topic"
    }}
  ]
}}
"""

    response = get_llm().models.generate_content(
        model=MODEL,
        contents=prompt,
    )
    
    grouped_data = _parse_json_response(response.text)
    if "error" in grouped_data:
        raise ValueError(f"Failed to group criteria: {grouped_data['error']}")
        
    # 4. Save to cache
    cache = RubricGroupingCache.objects.create(
        project=project,
        max_questions=max_questions,
        grouped_criteria=grouped_data
    )
    
    return cache
