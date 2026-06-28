"""
Brief drafter — produces structured domain briefs for technologies via LLM.

For each technology found in a submission:
  1. Check if an APPROVED brief (T1) already exists in the global DB.
     If yes, reuse it (departmental knowledge accumulation).
  2. Otherwise check for an existing pending T2 draft.
     If yes, reuse it (don't redraft what's already awaiting review).
  3. Otherwise, ask the LLM to draft one and store it as T2 (pending).

This is the "auto-draft + 30-second review" pattern from Phase 0 of the spec.
"""

import logging
from typing import List, Dict, Optional

from viva_evaluator.services.llm_service import llm_call

logger = logging.getLogger(__name__)


# =============================================================================
# Brief schema (what the LLM must produce)
# =============================================================================

BRIEF_SCHEMA_DESCRIPTION = """\
{
  "alternatives": [
    {
      "name":      "<alternative technology name>",
      "trigger":   "<when this alternative would be preferred>",
      "rationale": "<short reason — 1-2 sentences>"
    }
  ],
  "best_practices": [
    "<one best-practice statement, terse>",
    "..."
  ],
  "common_mistakes": [
    "<one common mistake students make>",
    "..."
  ]
}
"""


# =============================================================================
# Public API
# =============================================================================

def draft_briefs_for_submission(
    submission,
    technologies: List[Dict],
) -> List[Dict]:
    """
    For each technology, ensure a brief exists (approved OR pending).

    Args:
        submission:    ProjectSubmission whose extraction triggered this.
        technologies:  Output of tech_extractor.extract_technologies().

    Returns:
        List of dicts:
            {
                'technology': 'PostgreSQL',
                'tier': 1 | 2,
                'brief_id': '<uuid>',
                'status': 'active' | 'pending',
                'created_now': bool,
            }
    """
    from viva_evaluator.models import ApprovedDomainBrief

    results: List[Dict] = []

    for tech in technologies:
        tech_name = tech['name']
        category = tech.get('category', 'unknown')

        # 1. Reuse approved (T1) brief if present
        approved = ApprovedDomainBrief.objects.filter(
            technology__iexact=tech_name,
            status=ApprovedDomainBrief.Status.ACTIVE,
        ).order_by('-approved_at').first()

        if approved:
            logger.info(
                'Brief drafter: reusing T1 brief for %s (id=%s)',
                tech_name, approved.id,
            )
            results.append({
                'technology': tech_name,
                'tier':       approved.tier,
                'brief_id':   str(approved.id),
                'status':     approved.status,
                'created_now': False,
            })
            continue

        # 2. Reuse pending (T2) draft if present
        pending = ApprovedDomainBrief.objects.filter(
            technology__iexact=tech_name,
            status=ApprovedDomainBrief.Status.PENDING,
        ).order_by('-drafted_at').first()

        if pending:
            logger.info(
                'Brief drafter: pending T2 draft already exists for %s', tech_name,
            )
            results.append({
                'technology': tech_name,
                'tier':       pending.tier,
                'brief_id':   str(pending.id),
                'status':     pending.status,
                'created_now': False,
            })
            continue

        # 3. Draft a new T2 brief
        try:
            brief_json = _draft_brief_via_llm(tech_name, category)
        except Exception as exc:
            logger.warning('Brief drafting failed for %s: %s', tech_name, exc)
            continue

        if not brief_json:
            continue

        new_brief = ApprovedDomainBrief.objects.create(
            technology=tech_name,
            brief_json=brief_json,
            status=ApprovedDomainBrief.Status.PENDING,
            scope=ApprovedDomainBrief.Scope.EXAMINER,
            tier=2,
            drafted_for_submission=submission,
        )

        logger.info('Brief drafter: created T2 draft for %s (id=%s)', tech_name, new_brief.id)
        results.append({
            'technology': tech_name,
            'tier':       2,
            'brief_id':   str(new_brief.id),
            'status':     new_brief.status,
            'created_now': True,
        })

    return results


# =============================================================================
# Internals
# =============================================================================

def _draft_brief_via_llm(tech_name: str, category: str) -> Optional[Dict]:
    """Ask the LLM for a structured brief; validate shape before returning."""
    prompt = f"""You are seeding a knowledge base for a viva examination system.

Produce a structured brief about the technology: "{tech_name}"
(category: {category}).

The brief is used by an AI examiner to ask probing questions about the
student's choice of this technology — alternatives they could have picked,
best practices they should know, and common mistakes to challenge.

REQUIRED OUTPUT SHAPE (valid JSON, nothing else):
{BRIEF_SCHEMA_DESCRIPTION}

GUIDELINES:
- 3-5 alternatives covering different use cases (scale, simplicity, ecosystem)
- 4-6 concrete, technical best_practices specific to {tech_name}
- 3-5 common_mistakes that final-year CS students typically make
- Do NOT mention any specific student. This is generic domain knowledge.
- Be concrete; avoid platitudes like "use it well" or "follow good practices".
- Keep each list item short (under 25 words).
"""

    response = llm_call(
        prompt=prompt,
        model='reasoning',
        expect_json=True,
        max_retries=1,
        fallback=None,
    )

    if not isinstance(response, dict):
        return None

    # Defensive: ensure required keys are present and types are right
    alts = response.get('alternatives') or []
    bps = response.get('best_practices') or []
    mistakes = response.get('common_mistakes') or []

    if not (isinstance(alts, list) and isinstance(bps, list) and isinstance(mistakes, list)):
        logger.warning('Brief drafter: LLM returned malformed shape for %s', tech_name)
        return None

    return {
        'alternatives':    alts[:5],
        'best_practices':  bps[:6],
        'common_mistakes': mistakes[:5],
    }
