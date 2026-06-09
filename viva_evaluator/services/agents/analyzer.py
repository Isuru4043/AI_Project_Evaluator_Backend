"""
Analyzer agent — produces a 3D rubric assessment of a student's answer.

OUTPUT (3D rubric):
    correctness: {score: 0..1, evidence_quote: str, evidence_source: str}
    depth:       {score: 0..1, evidence_quote: str, evidence_source: str}
    consistency: {score: 0..1, evidence_quote: str, evidence_source: str}

PLUS:
    contradicts_code_flag: bool — student failed to acknowledge a CONTRADICTS_CODE issue
    gap_identified:        str  — missing concept the student didn't address
    revealed_assumption:   str  — implicit assumption in the answer (Strategist may probe)

CITATION VERIFICATION (Week 5):
    For each dimension, the Analyzer must cite an evidence_quote from the
    retrieved context (or the answer/transcript). After the LLM call, we
    verify each quote actually exists in the cited source via substring or
    embedding similarity. On failure, the dimension is marked unverified
    and downstream BKT excludes it (renormalizes weights).

Replaces the legacy `answer_evaluator.py` which produced a single LLM score.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from viva_evaluator.services.llm_service import llm_call
from viva_evaluator.services.rag.retrieval import format_chunks_for_prompt

logger = logging.getLogger(__name__)


# =============================================================================
# Inputs / outputs
# =============================================================================

@dataclass
class AnalyzerInput:
    question_text: str
    student_answer: str
    criterion_name: str
    criterion_description: str = ''
    retrieved_chunks: List[Dict] = field(default_factory=list)
    contradicts_code_alerts: List[Dict] = field(default_factory=list)
    transcript_recent: List[Dict] = field(default_factory=list)  # last 5 Q/A pairs


# =============================================================================
# Public API
# =============================================================================

def analyze_answer(inp: AnalyzerInput) -> Dict:
    """
    Run the 3D rubric over the student's answer and verify citations.

    Returns dict with this exact shape (used by BKT update + Strategist):
        {
            'correctness': {'score': float, 'evidence_quote': str,
                            'evidence_source': str, 'verified': bool},
            'depth':       {'score': float, 'evidence_quote': str,
                            'evidence_source': str, 'verified': bool},
            'consistency': {'score': float, 'evidence_quote': str,
                            'evidence_source': str, 'verified': bool},
            'contradicts_code_flag': bool,
            'gap_identified':        str,
            'revealed_assumption':   str,
            'soft_score':            float,    # weighted average for BKT input
            'reasoning':             str,      # free-form summary
        }
    """
    prompt = _build_prompt(inp)

    response = llm_call(
        prompt=prompt,
        model='reasoning',
        expect_json=True,
        max_retries=1,
        fallback=_fallback_analysis(),
    )

    # Defensive: ensure shape is correct
    if not isinstance(response, dict):
        response = _fallback_analysis()

    # Verify citations and compute weighted soft score
    verified = _verify_citations(response, inp)
    return verified


# =============================================================================
# Prompt construction
# =============================================================================

def _build_prompt(inp: AnalyzerInput) -> str:
    sources_block = format_chunks_for_prompt(inp.retrieved_chunks, max_chars=2000)

    contradicts_block = ''
    if inp.contradicts_code_alerts:
        items = []
        for alert in inp.contradicts_code_alerts[:3]:
            items.append(
                f"- Code finding: {alert.get('source')} → contradicts report claim: "
                f"{alert.get('target')} (severity: {alert.get('attrs', {}).get('severity', '?')})"
            )
        contradicts_block = (
            "\nKNOWN AUTHORSHIP ALERTS for this submission (set "
            "contradicts_code_flag=true ONLY if the student failed to "
            "acknowledge one of these in their answer):\n"
            + '\n'.join(items)
            + '\n'
        )

    transcript_block = ''
    if inp.transcript_recent:
        snippets = []
        for turn in inp.transcript_recent[-3:]:
            snippets.append(
                f"  Q: {turn.get('question_text', '')[:160]}\n"
                f"  A: {turn.get('answer_text', '')[:160]}"
            )
        transcript_block = (
            "\nRECENT TRANSCRIPT (for consistency check):\n"
            + '\n'.join(snippets)
            + '\n'
        )

    return f"""You are an academic viva examiner scoring a student's spoken answer.

RUBRIC CRITERION:
Name: {inp.criterion_name}
Description: {inp.criterion_description or '(no description)'}

QUESTION ASKED:
{inp.question_text}

STUDENT'S ANSWER (verbatim):
{inp.student_answer}

RETRIEVED SOURCES (the student's report and code — single source of truth):
{sources_block}
{contradicts_block}{transcript_block}
TASK — produce a 3D rubric assessment.

Each dimension is scored from 0.0 to 1.0:
  - correctness: does the answer align with what their report/code shows AND
                 with the rubric's expected concepts?
  - depth:       does it address WHY (not just what)? Engages with tradeoffs?
  - consistency: does it match what they said earlier in the transcript and
                 what's claimed in their submission? (Score 1.0 if no
                 inconsistencies are detectable.)

CITATION RULE (non-negotiable):
For each dimension, you MUST quote a SHORT excerpt (5-30 words) from one of:
  - the retrieved sources block above
  - the student's answer
  - the transcript

The excerpt must support your score. If you cannot find supporting evidence,
output an empty evidence_quote — the system will mark that dimension as
unverified and exclude it from scoring rather than guess.

The evidence_source must be one of:
    "retrieved" | "answer" | "transcript"

Respond ONLY with valid JSON in this exact shape:
{{
    "correctness": {{
        "score": 0.0,
        "evidence_quote": "...",
        "evidence_source": "retrieved"
    }},
    "depth": {{
        "score": 0.0,
        "evidence_quote": "...",
        "evidence_source": "answer"
    }},
    "consistency": {{
        "score": 0.0,
        "evidence_quote": "...",
        "evidence_source": "transcript"
    }},
    "contradicts_code_flag": false,
    "gap_identified": "<one sentence on what was missing>",
    "revealed_assumption": "<one sentence on any implicit assumption>",
    "reasoning": "<2-3 sentence summary of overall assessment>"
}}
"""


# =============================================================================
# Citation verification — programmatic post-LLM check
# =============================================================================

def _verify_citations(response: Dict, inp: AnalyzerInput) -> Dict:
    """
    For each dimension, check that the cited quote actually exists in the
    cited source. Mark dimensions as verified=True/False, then renormalize
    the soft score over only verified dimensions.
    """
    weights = {
        'correctness': 0.50,
        'depth':       0.35,
        'consistency': 0.15,
    }

    # Build searchable text per source
    retrieved_text = ' '.join(c.get('text', '') for c in inp.retrieved_chunks).lower()
    answer_text = (inp.student_answer or '').lower()
    transcript_text = ' '.join(
        (t.get('question_text', '') + ' ' + t.get('answer_text', ''))
        for t in inp.transcript_recent
    ).lower()

    source_lookup = {
        'retrieved':  retrieved_text,
        'answer':     answer_text,
        'transcript': transcript_text,
    }

    verified_total = 0.0
    weight_total = 0.0
    soft_score = 0.0

    for dim, weight in weights.items():
        dim_data = response.get(dim) or {}
        score = float(dim_data.get('score', 0.5)) if isinstance(dim_data, dict) else 0.5
        score = max(0.0, min(1.0, score))
        quote = (dim_data.get('evidence_quote') or '').strip().lower()
        src = (dim_data.get('evidence_source') or '').strip().lower()

        if not quote or src not in source_lookup:
            verified = False
        else:
            verified = _quote_present(quote, source_lookup[src])

        # Persist verification flag back into response for transparency
        response[dim] = {
            'score':           score,
            'evidence_quote':  dim_data.get('evidence_quote', ''),
            'evidence_source': src or '',
            'verified':        verified,
        }

        if verified:
            soft_score += score * weight
            weight_total += weight
            verified_total += weight

    # Renormalize over only verified dimensions
    if weight_total > 0:
        soft_score = soft_score / weight_total
    else:
        # All dimensions unverified — fall back to neutral
        soft_score = 0.5
        logger.warning('Analyzer: all dimensions unverified, using soft_score=0.5')

    response['soft_score'] = max(0.0, min(1.0, soft_score))
    response['verified_weight'] = weight_total

    response.setdefault('contradicts_code_flag', False)
    response.setdefault('gap_identified', '')
    response.setdefault('revealed_assumption', '')
    response.setdefault('reasoning', '')

    logger.info(
        'Analyzer: c=%.2f%s d=%.2f%s con=%.2f%s soft=%.2f',
        response['correctness']['score'], '✓' if response['correctness']['verified'] else '✗',
        response['depth']['score'],       '✓' if response['depth']['verified'] else '✗',
        response['consistency']['score'], '✓' if response['consistency']['verified'] else '✗',
        response['soft_score'],
    )

    return response


def _quote_present(quote: str, source_text: str) -> bool:
    """
    Verify the quote appears in source_text. Two-stage check:
      1. Substring match (fast, exact).
      2. Token overlap fallback for paraphrased citations (≥ 70% words present).
    """
    if not quote or not source_text:
        return False

    quote = quote.strip().strip('"').strip("'")
    if not quote:
        return False

    # Stage 1: direct substring
    if quote in source_text:
        return True

    # Stage 2: word-overlap fallback for slight paraphrasing
    quote_tokens = [w for w in _tokenize(quote) if len(w) > 2]
    if not quote_tokens:
        return False

    source_tokens = set(_tokenize(source_text))
    matched = sum(1 for w in quote_tokens if w in source_tokens)
    return (matched / len(quote_tokens)) >= 0.7


def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r'[a-z0-9_]+', text.lower())


# =============================================================================
# Fallback when LLM fails entirely
# =============================================================================

def _fallback_analysis() -> Dict:
    return {
        'correctness': {'score': 0.5, 'evidence_quote': '', 'evidence_source': ''},
        'depth':       {'score': 0.5, 'evidence_quote': '', 'evidence_source': ''},
        'consistency': {'score': 0.5, 'evidence_quote': '', 'evidence_source': ''},
        'contradicts_code_flag': False,
        'gap_identified': '',
        'revealed_assumption': '',
        'reasoning': 'Analysis unavailable; using neutral score.',
    }
