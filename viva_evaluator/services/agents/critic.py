"""
Critic agent — Tier 2 LLM-based validation of generated questions.

WHEN IT RUNS:
    Only after Tier 1 (programmatic) passes. Tier 1 catches the easy stuff
    (length, anchor regex, single ?, similarity to recent). Tier 2 catches
    the subtler issues that need an LLM judgment:
        - Specificity: does the question reference concrete student work?
        - Bloom alignment: does it actually require the targeted level?
        - Hallucination: does it assert claims not in the retrieved context?

OUTPUT:
    {
        'passed': bool,
        'critique': str,                       # one-sentence reason on fail
        'specificity_score': float (0..1),
        'bloom_alignment_score': float (0..1),
        'hallucination_flag': bool,
    }

INTEGRATION:
    The Questioner runs Tier 1, then if it passes, runs the Critic. On
    failure, the Questioner regenerates with the critique appended to its
    prompt. Maximum 2 Critic retries.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

from viva_evaluator.services.llm_service import llm_call
from viva_evaluator.services.pipeline.contracts import QuestionEvidencePackage

logger = logging.getLogger(__name__)


@dataclass
class CriticInput:
    question_text: str
    target_bloom: str                          # what the Strategist asked for
    target_intent: str                         # the Socratic intent
    evidence_package: Optional[QuestionEvidencePackage] = None
    source_reference_ids: List[str] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)
    module_chunks: Optional[List[Dict]] = None  # Legacy caller compatibility
    student_last_answer: Optional[str] = None  # for "you mentioned ..." checks


# =============================================================================
# Public API
# =============================================================================

def critique_question(inp: CriticInput) -> Dict:
    """
    Run Tier 2 LLM-based critique on a question.

    Returns dict with:
        passed:                bool
        critique:              short reason on fail (empty on pass)
        specificity_score:     0..1
        bloom_alignment_score: 0..1
        hallucination_flag:    bool
    """
    prompt = _build_prompt(inp)

    try:
        response = llm_call(
            prompt=prompt,
            model='fast',
            expect_json=True,
            max_retries=1,
            fallback={
                '_critic_unavailable': True,
                'unavailable_reason': 'critic_llm_call_failed',
            },
            operation='question_critic',
        )
    except Exception as exc:
        logger.exception("critic call failed unexpectedly")
        return _critic_unavailable(f"critic_exception:{type(exc).__name__}")

    if not isinstance(response, dict):
        return _critic_unavailable('critic_response_not_an_object')
    if response.get('_critic_unavailable') is True:
        return _critic_unavailable(
            str(response.get('unavailable_reason') or 'critic_llm_call_failed')
        )

    malformed_reason = _critic_schema_failure(response)
    if malformed_reason:
        logger.warning("critic returned malformed output: %s", malformed_reason)
        return _critic_unavailable(malformed_reason)

    # Defensive normalization
    spec = float(response['specificity_score'])
    bloom = float(response['bloom_alignment_score'])
    conv = float(response['conversational_flow_score'])
    bound = float(response['boundary_check_score'])
    source_support = float(response['source_reference_support_score'])
    halluc = response['hallucination_flag']

    # Critic passes if all quality bars are met
    quality_ok = (
        spec >= 0.5
        and bloom >= 0.5
        and conv >= 0.5
        and bound >= 0.5
        and source_support >= 0.5
        and not halluc
    )
    passed = response['passed'] and quality_ok

    return {
        'passed':                passed,
        'critique':              str(response.get('critique', '') or '')[:300],
        'specificity_score':     spec,
        'bloom_alignment_score': bloom,
        'conversational_flow_score': conv,
        'boundary_check_score':  bound,
        'source_reference_support_score': source_support,
        'hallucination_flag':    halluc,
        '_critic_unavailable':   False,
    }


def _critic_unavailable(reason: str) -> Dict:
    """Return an explicit fail-safe result; unavailability is never a pass."""
    return {
        'passed': False,
        'critique': 'Critic validation was unavailable.',
        'specificity_score': 0.0,
        'bloom_alignment_score': 0.0,
        'conversational_flow_score': 0.0,
        'boundary_check_score': 0.0,
        'source_reference_support_score': 0.0,
        'hallucination_flag': False,
        '_critic_unavailable': True,
        'unavailable_reason': reason[:200],
    }


def _critic_schema_failure(response: Dict[str, Any]) -> str:
    required_booleans = ('passed', 'hallucination_flag')
    required_scores = (
        'specificity_score',
        'bloom_alignment_score',
        'conversational_flow_score',
        'boundary_check_score',
        'source_reference_support_score',
    )
    for field_name in required_booleans:
        if not isinstance(response.get(field_name), bool):
            return f'critic_malformed_{field_name}'
    if not isinstance(response.get('critique'), str):
        return 'critic_malformed_critique'
    for field_name in required_scores:
        value = response.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            return f'critic_malformed_{field_name}'
    return ''


# =============================================================================
# Prompt
# =============================================================================

def _build_prompt(inp: CriticInput) -> str:
    from viva_evaluator.services.rag.retrieval import format_chunks_for_prompt

    if inp.evidence_package is not None:
        sources_block = _format_evidence_package(inp.evidence_package)
    else:
        sources_block = format_chunks_for_prompt(
            inp.retrieved_chunks,
            max_chars=1500,
        )

    last_answer_block = ''
    if inp.student_last_answer and inp.evidence_package is None:
        last_answer_block = (
            f"\nSTUDENT'S LAST ANSWER (verbatim):\n"
            f'"{inp.student_last_answer[:400]}"\n'
        )

    module_block = ''
    if inp.module_chunks and inp.evidence_package is None:
        module_text = format_chunks_for_prompt(inp.module_chunks, max_chars=1500)
        module_block = (
            "\nMODULE MATERIALS (The theoretical boundary for the Viva Session):\n"
            f"{module_text}\n"
        )

    source_reference_block = (
        '\n'.join(f"- {evidence_id}" for evidence_id in inp.source_reference_ids)
        if inp.source_reference_ids
        else "- (none reported)"
    )

    return f"""You are reviewing a viva question generated by another agent.
Your job is to catch quality issues before the question is shown to the student.

QUESTION TO REVIEW:
"{inp.question_text}"

EXAMINER STRATEGY THE QUESTION SHOULD FULFILL:
  Target Bloom level:   {inp.target_bloom}
  Socratic intent:      {inp.target_intent}

COMPLETE EVIDENCE PACKAGE SHARED WITH THE QUESTIONER (the ONLY valid source of
factual claims — anything not present here is hallucinated):

{sources_block}
{last_answer_block}
{module_block}
SOURCE IDS REPORTED BY THE QUESTIONER:
{source_reference_block}

SCORE THE QUESTION ON SIX DIMENSIONS (each 0.0 to 1.0):

1. SPECIFICITY: does the question reference something concrete from the
   student's actual work, or is it a generic question that could be asked
   of any student on this topic?
       1.0 = anchored to specific code/section/student-quote
       0.5 = generic but on-topic
       0.0 = could be asked of any student in any year

2. BLOOM ALIGNMENT: does the question actually require the cognitive level
   of {inp.target_bloom}? A question that can be answered by recall when
   Evaluate was specified should fail this check.
       1.0 = perfectly matches the target Bloom level
       0.5 = roughly the right level but loose
       0.0 = wrong cognitive level entirely

3. HALLUCINATION FLAG (boolean): does the question assert any claim about
   the student's project that CANNOT be verified from the retrieved sources
   above? Examples of hallucinations:
   - Inventing a function name not in the sources
   - Citing a section/table that doesn't exist
   - Claiming the student wrote something they didn't write
   - Asking the student to compare their work against an external alternative 
     technology (e.g., AES-CBC, React vs Angular) that is NOT explicitly 
     mentioned in the retrieved sources.

4. CONVERSATIONAL FLOW: does the question read naturally when spoken aloud? 
   Is it broken into two simple sentences, or is it one massive, overly-dense 
   sentence packed with technical jargon?
       1.0 = Natural, conversational, easy to hear (e.g. 2 short sentences)
       0.5 = Okay, but a bit dense or academic
       0.0 = Hard to parse when spoken, long multi-clause sentence

5. MODULE BOUNDARY (Strict): Does the question expect the student to know technologies
   or concepts that are outside the theoretical boundaries taught in their module?
   If Module Materials are provided, only concepts within them should be expected.
       1.0 = Strictly within the module boundary, or student's own project
       0.0 = Asks about external concepts/frameworks not taught in the module

6. SOURCE-REFERENCE SUPPORT: Do the reported source IDs exist in the evidence
   package, and do they directly support the specific factual anchors used by
   the question? A real ID that does not support the wording must still fail.
       1.0 = every factual anchor is directly supported by the cited evidence
       0.5 = evidence is related but support is indirect
       0.0 = cited evidence is missing, unrelated, or contradicted

Respond ONLY with valid JSON in this exact shape:
{{
    "passed": true,
    "specificity_score": 0.85,
    "bloom_alignment_score": 0.90,
    "conversational_flow_score": 0.85,
    "boundary_check_score": 1.0,
    "source_reference_support_score": 0.90,
    "hallucination_flag": false,
    "critique": "<one-sentence reason if passed=false; empty string if passed=true>"
}}

Set passed=false if any numeric score is below 0.5, or hallucination_flag is
true.
"""


def _format_evidence_package(package: QuestionEvidencePackage) -> str:
    labels = {
        'submission_chunk': 'STUDENT SUBMISSION EVIDENCE',
        'module_chunk': 'MODULE-BOUNDARY EVIDENCE',
        'kg_contradiction': 'KG CONTRADICTION EVIDENCE',
        'kg_alternative': 'KG ALTERNATIVE EVIDENCE',
        'kg_dependency': 'KG DEPENDENCY EVIDENCE',
        'previous_answer': 'PREVIOUS STUDENT ANSWER',
        'presentation_segment': 'PRESENTATION EVIDENCE',
    }
    grouped = []
    for evidence_type, label in labels.items():
        references = package.of_type(evidence_type)
        if not references:
            continue
        lines = [
            f"[{reference.evidence_id}] {reference.content[:900]}"
            for reference in references
        ]
        grouped.append(f"{label}:\n" + '\n'.join(lines))
    return '\n\n'.join(grouped) or '(no attributable evidence available)'
