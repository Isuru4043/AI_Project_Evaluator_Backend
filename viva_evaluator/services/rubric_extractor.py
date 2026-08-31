import json
import logging
import re
import time

from django.conf import settings

from AI_Evaluator_Backend.llm import get_llm

logger = logging.getLogger(__name__)

MODEL = settings.GEMINI_MODEL

# Retry config — matches the resilience pattern in llm_service.py
MAX_RETRIES = 2          # total attempts = MAX_RETRIES + 1 = 3
BACKOFF_BASE_SECS = 1.0  # 1s, 2s exponential backoff


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
- Whether each criterion assesses an individual student's demonstrated knowledge
  or a shared group outcome
- Suggest how many viva questions should be asked per criterion based on its complexity and weight (between 2 and 5)

SCORING-SCOPE RULES:
- Set `is_individual` to true when the viva answer demonstrates one student's
  understanding, explanation, communication, implementation knowledge, design
  reasoning, security analysis, or personal contribution.
- Set it to false only when every member should receive the same mark for a
  genuinely shared artifact or team outcome, regardless of who answers.
- When uncertain, use true. Oral answers should not silently become group marks.

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
                    "is_individual": true,
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

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            response = get_llm().models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            latency_ms = int((time.time() - t0) * 1000)
            raw_text = (response.text or '').strip()

            logger.info(
                'rubric_extract ok model=%s attempt=%d latency=%dms chars_out=%d',
                MODEL, attempt, latency_ms, len(raw_text),
            )

            parsed = _parse_json_response(raw_text)

            # If JSON parsing failed, retry (Gemini sometimes returns prose
            # on the first attempt but JSON on the second)
            if 'error' in parsed and attempt < MAX_RETRIES:
                logger.warning(
                    'rubric_extract json_parse_failed attempt=%d, retrying. preview=%r',
                    attempt, raw_text[:200],
                )
                time.sleep(BACKOFF_BASE_SECS * (2 ** attempt))
                continue

            return parsed

        except Exception as exc:
            last_error = exc
            logger.warning(
                'rubric_extract error attempt=%d model=%s err=%s',
                attempt, MODEL, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_SECS * (2 ** attempt))

    # All retries exhausted
    logger.error('rubric_extract exhausted %d attempts. last_error=%s', MAX_RETRIES + 1, last_error)
    return {
        "error": f"Rubric extraction failed after {MAX_RETRIES + 1} attempts. Please try again.",
    }


# =============================================================================
# JSON parsing — robust against markdown fences, leading/trailing prose.
# Mirrors the battle-tested _parse_json logic from llm_service.py.
# =============================================================================

_FENCE_RE = re.compile(r'^```(?:json)?\s*|\s*```$', re.MULTILINE)


def _parse_json_response(response_text: str) -> dict:
    """Safely parses JSON response from Gemini.

    Handles:
      - Plain JSON
      - Markdown-fenced JSON (```json ... ```)
      - JSON embedded in surrounding prose (extracts first {...})
    """
    text = response_text.strip()
    if not text:
        return {"error": "Empty response from AI model."}

    # Strip markdown fences
    cleaned = _FENCE_RE.sub('', text).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
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
            "is_individual": c.is_individual,
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
- Never combine individual and shared-group criteria in the same topic.

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
