"""
Code summarizer — turns a list of AST units into one-line summaries
via batched LLM calls.

KEY DESIGN: BATCH BY 10
    A typical FYP repo has 100-300 functions. One LLM call per function
    would be slow and costly. We batch 10 units per call and ask the LLM
    to produce one summary per unit. This reduces N calls to N/10.

OUTPUT:
    Each input unit gets a 'summary' key added:
        {
            'file_path': ...,
            'name': ...,
            'source': ...,
            'summary': "Returns true if the JWT token is valid and not expired."
        }
"""

import json
import logging
from typing import List, Dict

from viva_evaluator.services.llm_service import llm_call

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

BATCH_SIZE = 10                # units per LLM call
MAX_UNITS_PER_REPO = 200       # cap to keep cost predictable
MAX_SOURCE_PER_UNIT = 1500     # truncate huge functions before sending


# =============================================================================
# Public API
# =============================================================================

def summarize_units(units: List[Dict]) -> List[Dict]:
    """
    Add a 'summary' field to each unit. Returns the same list, mutated.

    On LLM failure for a batch, the affected units get a fallback summary
    derived from the function name + signature so retrieval still works.
    """
    if not units:
        return []

    if len(units) > MAX_UNITS_PER_REPO:
        logger.info(
            'Summarizer: capping at %d units (had %d)',
            MAX_UNITS_PER_REPO, len(units),
        )
        # Prioritize: keep function/class units first, drop file units last
        units = sorted(units, key=lambda u: 0 if u['unit_type'] != 'file' else 1)
        units = units[:MAX_UNITS_PER_REPO]

    # Batch
    batches = [units[i:i + BATCH_SIZE] for i in range(0, len(units), BATCH_SIZE)]
    logger.info(
        'Summarizer: %d units in %d batches of %d',
        len(units), len(batches), BATCH_SIZE,
    )

    for batch in batches:
        _summarize_batch(batch)

    return units


# =============================================================================
# Internals
# =============================================================================

def _summarize_batch(batch: List[Dict]) -> None:
    """Mutates each unit in `batch` to add a 'summary' field."""
    prompt = _build_batch_prompt(batch)

    response = llm_call(
        prompt=prompt,
        model='fast',          # cheap model is fine for one-line summaries
        expect_json=True,
        max_retries=1,
        fallback={'summaries': []},
    )

    summaries = response.get('summaries', []) if isinstance(response, dict) else []

    if len(summaries) == len(batch):
        for unit, item in zip(batch, summaries):
            unit['summary'] = (item.get('summary') or '').strip() or _fallback_summary(unit)
    else:
        # Mismatched response — apply fallback to entire batch
        logger.warning(
            'Summarizer: batch returned %d summaries for %d units, using fallback',
            len(summaries), len(batch),
        )
        for unit in batch:
            unit['summary'] = _fallback_summary(unit)


def _build_batch_prompt(batch: List[Dict]) -> str:
    """Build a prompt that asks for N summaries indexed 0..N-1."""
    code_blocks = []
    for idx, unit in enumerate(batch):
        snippet = (unit.get('source') or '').strip()[:MAX_SOURCE_PER_UNIT]
        code_blocks.append(
            f"--- UNIT {idx} ---\n"
            f"File: {unit['file_path']}\n"
            f"Type: {unit['unit_type']}\n"
            f"Name: {unit['name']}\n"
            f"Code:\n{snippet}\n"
        )
    code_section = '\n'.join(code_blocks)

    return f"""You are summarising code units from a student's project for a viva
examination retrieval system.

For each UNIT below, write ONE sentence describing what the unit does.
Focus on PURPOSE, not implementation details:
- What problem does it solve?
- What does it produce or change?
- Mention key technologies / patterns visible (auth, DB query, API call, etc.)

Keep each summary under 25 words. Be concrete. No filler like "This function".

{code_section}

Respond ONLY with valid JSON in this exact shape:
{{
    "summaries": [
        {{"unit_index": 0, "summary": "..."}},
        {{"unit_index": 1, "summary": "..."}},
        ...
    ]
}}

The summaries array must contain exactly {len(batch)} items, one per UNIT
above, in the same order.
"""


def _fallback_summary(unit: Dict) -> str:
    """Last-resort summary built from metadata, no LLM needed."""
    return (
        f"{unit['unit_type'].capitalize()} '{unit['name']}' "
        f"in {unit['file_path']}."
    )
