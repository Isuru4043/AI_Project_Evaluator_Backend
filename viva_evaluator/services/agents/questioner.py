"""
Questioner agent — generates anchored viva questions.

WEEK 1 BEHAVIOR:
    - Takes retrieved chunks (RAG) as the ONLY source of student-specific content.
    - Builds a prompt that REQUIRES anchoring (one of three patterns).
    - Calls llm_service.
    - Validates output with Tier 1 (programmatic checks).
    - Retries once on Tier 1 failure with the failure reason in the prompt.

WEEK 5 EVOLUTION:
    Strategist will pick Bloom level + intent. Critic will run after Tier 1.
    For now we accept difficulty/blooms as inputs and skip Critic.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from viva_evaluator.services.llm_service import llm_call
from viva_evaluator.services.rag.retrieval import format_chunks_for_prompt
from viva_evaluator.services.agents.tier1_validator import (
    validate_question,
    Tier1Result,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Difficulty → Bloom mapping (matches v3 spec; will be replaced by Strategist).
# =============================================================================

DIFFICULTY_TO_BLOOMS = {
    'easy':   'Understand',
    'medium': 'Analyze',
    'hard':   'Evaluate',
}


@dataclass
class QuestionerInput:
    """All context the Questioner needs to produce one question."""

    criterion_name: str
    criterion_description: str = ''
    retrieved_chunks: List[Dict] = field(default_factory=list)
    kg_signals: Optional[Dict] = None                 # Week 3: hybrid retrieval output
    difficulty: str = 'medium'                        # 'easy' | 'medium' | 'hard'
    question_hints: List[str] = field(default_factory=list)
    recent_questions: List[str] = field(default_factory=list)
    previous_question: Optional[str] = None           # for follow-up mode
    previous_answer: Optional[str] = None             # for follow-up mode
    is_first_question: bool = True
    question_number_in_criterion: int = 1
    # A1 Response Triage: when True, re-ask the SAME underlying question in
    # simpler, clearer words because the student found the previous phrasing
    # unclear. clarify_reason carries the triage rationale.
    clarify_mode: bool = False
    clarify_reason: str = ''
    # B3 Weak-retrieval awareness: when True, the submission contains little
    # specific material on this criterion, so ask a BROADER conceptual question
    # instead of inventing specific details the student may never have written.
    weak_grounding: bool = False


# =============================================================================
# Public API — what the views call.
# =============================================================================

def generate_anchored_question(
    inp: QuestionerInput,
    max_retries: int = 1,
    enable_critic: bool = True,
) -> Dict:
    """
    Produce a single anchored viva question with Tier 1 + Tier 2 validation.

    Args:
        inp:           All required context (see QuestionerInput).
        max_retries:   How many times to regenerate if Tier 1 fails. Default 1.
        enable_critic: If True (default), run Tier 2 LLM critique after Tier 1
                       passes. Up to 2 critic retries per v3 spec.

    Returns:
        dict with keys:
            question_text
            blooms_level
            difficulty
            tier1_passed
            tier1_failures
            critic_passed
            critic_critique
            attempts            -- total LLM calls
    """
    blooms = DIFFICULTY_TO_BLOOMS.get(inp.difficulty, 'Analyze')

    prompt = _build_prompt(inp, blooms, retry_reason=None)
    response = llm_call(
        prompt,
        model='reasoning',
        expect_json=True,
        fallback={'question_text': '', 'blooms_level': blooms, 'difficulty': inp.difficulty},
    )
    question_text = (response.get('question_text') or '').strip()

    # ---- Tier 1 validation -------------------------------------------------
    result = validate_question(question_text, recent_questions=inp.recent_questions)
    attempts = 1

    # One Tier 1 retry with failure reasons fed back into the prompt
    if not result.passed and max_retries > 0:
        logger.info(
            'questioner: Tier 1 failed (%s). Retrying with failure context.',
            result.reason_string(),
        )
        retry_prompt = _build_prompt(inp, blooms, retry_reason=result.reason_string())
        retry_response = llm_call(
            retry_prompt,
            model='reasoning',
            expect_json=True,
            fallback={'question_text': question_text, 'blooms_level': blooms, 'difficulty': inp.difficulty},
        )
        retry_text = (retry_response.get('question_text') or '').strip()
        retry_result = validate_question(retry_text, recent_questions=inp.recent_questions)
        attempts = 2

        # Use retry only if it's strictly better
        if retry_result.passed or (not result.passed and len(retry_result.failures) < len(result.failures)):
            question_text = retry_text
            result = retry_result
            response = retry_response

    # ---- Tier 2 (Critic) — only if Tier 1 passed --------------------------
    critic_passed = True
    critic_critique = ''
    critic_data = None
    critic_ran = False

    if enable_critic and result.passed and question_text:
        critic_ran = True
        critic_passed, critic_critique, attempts, response, question_text, result, critic_data = (
            _run_critic_loop(
                inp=inp,
                blooms=blooms,
                question_text=question_text,
                response=response,
                tier1_result=result,
                attempts=attempts,
            )
        )

    return {
        'question_text':   question_text,
        'blooms_level':    response.get('blooms_level', blooms),
        'difficulty':      response.get('difficulty', inp.difficulty),
        'tier1_passed':    result.passed,
        'tier1_failures':  result.failures,
        'critic_ran':      critic_ran,
        'critic_passed':   critic_passed if critic_ran else None,
        'critic_critique': critic_critique,
        'critic_scores':   critic_data or {},
        'attempts':        attempts,
    }


# =============================================================================
# Prompt construction — anchoring is a HARD RULE, not a suggestion.
# =============================================================================

_ANCHORING_RULES = """\
HARD RULES — your question MUST follow ALL of these:

1. ANCHORING (non-negotiable). Your question MUST contain at least ONE
   explicit reference to the student's actual work. Acceptable patterns
   (use any of these naturally in the question — they don't need to be
   at the start):
   (a) Direct quote from the student's report or earlier answer:
       "you mentioned ...", "you described ...", "you wrote ..."
   (b) The student's specific design choice or claim:
       "your choice of X", "your approach to Y", "your <module> architecture"
   (c) A specific code element from the retrieved sources:
       "your <function/class/module name>", "looking at your <component>"

   The question must NOT be a generic question that could be asked of any
   student on this topic. It must reference something concrete that this
   specific student wrote or built, drawn from the retrieved sources below.

2. NEVER reference document locations. The student is in an oral exam and
   does NOT have the report in front of them. You MUST NOT mention:
       - page numbers ("on page 5", "page 12")
       - table numbers ("Table 4.1", "Table 2")
       - figure numbers ("Figure 3", "Fig. 5.2")
       - section numbers ("Section 3.1", "Chapter 4")
       - appendix references
       - citation markers ("[cite: 9]", "[1]")
   Instead, refer to the CONTENT itself: "you described your encryption
   approach", "in your threat model", "your zero trust design".

3. KEEP IT SHORT AND SPOKEN. Target length: 20-40 words. Maximum 60.
   This is an ORAL exam — the student must hold the entire question in
   their head. Long, written-style questions are bad. Examples:
       BAD  (too long, written-style):
         "Considering your Zero Trust goal and the problem of compromised
          servers, how complete and architecturally sound is this single
          countermeasure against all confidentiality threats, particularly
          those involving active server compromise or key management
          vulnerabilities beyond data at rest?"
       GOOD (short, spoken):
         "You said AES-256-GCM gives confidentiality even if the server
          is compromised — what threats does it not cover?"

4. AVOID quoting long phrases from the report. Paraphrase the student's
   idea in plain words instead of pasting their wording back at them.
   At most quote 3-5 words, not full sentences.

5. PLAIN LANGUAGE. Phrase it as a real examiner SPEAKS aloud, not as
   they would write in a paper. The student is a final-year CS student
   so technical terms from THEIR project are fine ("encryption",
   "authentication", "API"). What you must avoid is academic register
   in the question itself.

   THE READ-ALOUD TEST: Imagine asking the question across a table while
   drinking coffee. If it sounds stiff, rewrite it.

       AVOID (stiff/written):              PREFER (spoken/natural):
         "exfiltrate"                        "steal" / "leak" / "take"
         "mitigate"                          "reduce" / "handle" / "fix"
         "ascertain"                         "check" / "figure out"
         "elucidate"                         "explain" / "walk through"
         "ramifications"                     "effects" / "consequences"
         "architecturally sound"             "good design" / "the right call"
         "in totality"                       "overall"
         "vis-à-vis"                         "compared to" / "against"
         "considering the implications of"   "given" / "with"
         "particularly those involving"      "especially when"
         "as it pertains to"                 "for"
         "in the context of"                 "for" / "when"

   General rule: a 1-syllable verb beats a 4-syllable verb when both
   carry the same meaning.

6. PUNCTUATION: end with exactly one '?'. No compound questions.

7. OPEN-ENDED: cannot be answered with yes or no.

8. GROUNDING: every concrete claim about the student's project must come
   from the retrieved sources below. Do NOT invent file names, function
   names, or claims that do not appear in the sources.

9. ONE THING AT A TIME. The question must ask the student to reason about
   exactly ONE thing — one decision, one trade-off, one concept, one
   mechanism. Stacking multiple ideas with "considering X, given Y, with
   Z..." overloads working memory.
       BAD (stacked):
         "Was this the right trade-off, considering the problem of
          balancing automation with human judgment in assessments?"
       GOOD (one thing):
         "Was making the workflow slower the right call here?"
       BAD (stacked):
         "How does X work, given Y, particularly when Z occurs?"
       GOOD (one thing):
         "Walk me through how X handles Z."

   If you find yourself writing "considering the problem of",
   "particularly those involving", or "in the context of balancing",
   you are stacking — strip the framing and ask the core question.

10. UNPACK THE STUDENT'S OWN JARGON. If the student coined a term in
    their report (e.g., "positive friction", "wrapped keys", "TOFU
    pinning"), do NOT just quote it back at them. Either:
    (a) Briefly explain what they meant when you reference it:
          BAD:  "You mentioned 'positive friction' — was that the right call?"
          GOOD: "You designed the workflow so examiners must explicitly
                 approve each AI grade. Was that extra step worth the
                 slowdown?"
    (b) Or skip the jargon and reference the underlying decision:
          GOOD: "Why did you make examiners approve every AI grade
                 manually?"

    The student wrote their report weeks ago. They may not remember
    exact phrasing of every term they invented.
"""


def _build_prompt(
    inp: QuestionerInput,
    blooms_level: str,
    retry_reason: Optional[str],
) -> str:
    sources_block = format_chunks_for_prompt(inp.retrieved_chunks, max_chars=2400)

    # Week 3: render KG signals (CONTRADICTS_CODE alerts, dependencies)
    kg_block = ''
    if inp.kg_signals:
        from viva_evaluator.services.rag.retrieval import format_kg_signals_for_prompt
        kg_text = format_kg_signals_for_prompt(inp.kg_signals)
        if kg_text:
            kg_block = f"\nKNOWLEDGE GRAPH SIGNALS:\n{kg_text}\n"

    if inp.is_first_question or not inp.previous_question:
        conversation_block = '(This is the opening question for this criterion.)'
    else:
        conversation_block = (
            f"PREVIOUS QUESTION (from you):\n{inp.previous_question}\n\n"
            f"STUDENT'S ANSWER (their exact words):\n{inp.previous_answer or '(no answer)'}"
        )

    hints_block = ''
    if inp.question_hints:
        hints_text = '\n'.join(f"- {h}" for h in inp.question_hints)
        hints_block = (
            "\nEXAMINER'S SUGGESTED FOCUS AREAS (use as loose guidelines, not exact wording):\n"
            f"{hints_text}\n"
        )

    retry_block = ''
    if retry_reason:
        retry_block = (
            f"\n⚠ PREVIOUS ATTEMPT FAILED VALIDATION: {retry_reason}\n"
            "Fix these specific issues. Pay particular attention to the "
            "ANCHORING rule and word count.\n"
        )

    # A1: clarification mode — the student didn't understand the previous
    # phrasing. Re-ask the SAME underlying question more simply and clearly.
    clarify_block = ''
    if inp.clarify_mode:
        clarify_block = (
            "\n⚠ CLARIFICATION MODE: The student did NOT understand the previous "
            "question"
            + (f" (reason: {inp.clarify_reason})" if inp.clarify_reason else '')
            + ".\nRe-ask the SAME underlying question about the SAME concept, but "
            "in simpler, clearer, more concrete words. Shorten it. Avoid jargon "
            "and any term the student may not recognise. Do NOT switch to a "
            "different topic, and do NOT make it harder — the goal is purely to "
            "make the question understandable.\n"
        )

    # B3: weak-grounding mode — the report barely covers this criterion. Ask a
    # broad, open question and DO NOT fabricate specific artifacts. Anchor only
    # to the student's project in general terms ("in your project", "your
    # report") which still satisfies the anchoring rule honestly.
    weak_grounding_block = ''
    if inp.weak_grounding:
        weak_grounding_block = (
            "\n⚠ LIMITED SOURCE MATERIAL: the submission contains little specific "
            "content on this criterion. Ask a BROADER, open conceptual question "
            "about how the student approached this topic in their project. Do NOT "
            "invent file names, function names, figures, or specific claims — none "
            "are available. Anchor generally (e.g. 'in your project', 'your "
            "report') rather than to a specific artifact.\n"
        )

    bloom_phrasing = _bloom_phrasing_hint(blooms_level)

    return f"""You are an academic viva examiner conducting an oral examination.

RUBRIC CRITERION:
Name: {inp.criterion_name}
Description: {inp.criterion_description or '(no description)'}

RETRIEVED SOURCES from the student's submission (the ONLY source of truth about
their project — every concrete reference must come from here):

{sources_block}
{kg_block}
CONVERSATION CONTEXT:
{conversation_block}
{hints_block}{retry_block}{clarify_block}{weak_grounding_block}
TARGET BLOOM'S LEVEL: {blooms_level}
PHRASING STYLE FOR THIS LEVEL: {bloom_phrasing}

{_ANCHORING_RULES}

Generate ONE viva question following all the rules above.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "question_text": "your question here",
    "blooms_level": "{blooms_level}",
    "difficulty": "{inp.difficulty}"
}}
"""


def _bloom_phrasing_hint(blooms_level: str) -> str:
    return {
        'Remember':   'recall — "Can you describe what X does in your system?"',
        'Understand': 'explanation — "Can you walk me through how X works?"',
        'Apply':      'application — "How did you apply X to solve Y in your implementation?"',
        'Analyze':    'analysis — "Why does your choice of X behave differently when Z?"',
        'Evaluate':   'judgment — "Was X the right tradeoff given your stated objective Y?"',
        'Create':     'redesign — "If you redesigned this, what would change and why?"',
    }.get(blooms_level, 'analysis-level reasoning')


# =============================================================================
# Tier 2 — Critic loop (Week 6)
# =============================================================================

# Tier 2 critic retry budget. 1 retry (= up to 2 critic evaluations) keeps
# worst-case latency bounded while still catching most quality issues; the
# best-candidate fallback guarantees we never ship a blank question.
CRITIC_MAX_RETRIES = 1


def _run_critic_loop(
    inp: 'QuestionerInput',
    blooms: str,
    question_text: str,
    response: Dict,
    tier1_result: 'Tier1Result',
    attempts: int,
):
    """
    Run the Critic on the candidate question. On fail, regenerate with the
    critique appended to the Questioner prompt. Up to 2 critic retries.

    Returns:
        (critic_passed, critic_critique, attempts, response, question_text,
         tier1_result, critic_data)

    The best (highest-scoring) candidate is returned even if all retries
    fail Tier 2 — we don't want a blank question, just the best we got.
    """
    from viva_evaluator.services.agents.critic import critique_question, CriticInput

    best_text = question_text
    best_response = response
    best_tier1 = tier1_result
    best_critic_score = -1.0
    best_critic_data: Dict = {}
    best_critic_critique = ''
    best_critic_passed = False

    current_text = question_text
    current_response = response
    current_tier1 = tier1_result

    for attempt_idx in range(CRITIC_MAX_RETRIES + 1):
        critic_result = critique_question(CriticInput(
            question_text=current_text,
            target_bloom=blooms,
            target_intent=_intent_label_from_kg(inp.kg_signals),
            retrieved_chunks=inp.retrieved_chunks,
            student_last_answer=inp.previous_answer,
        ))

        critic_score = (
            critic_result['specificity_score']
            + critic_result['bloom_alignment_score']
        ) / 2.0
        if critic_result['hallucination_flag']:
            critic_score *= 0.5  # halve score for hallucination

        if critic_score > best_critic_score:
            best_text = current_text
            best_response = current_response
            best_tier1 = current_tier1
            best_critic_score = critic_score
            best_critic_data = {
                'specificity':     critic_result['specificity_score'],
                'bloom_alignment': critic_result['bloom_alignment_score'],
                'hallucination':   critic_result['hallucination_flag'],
            }
            best_critic_passed = critic_result['passed']
            best_critic_critique = critic_result['critique']

        if critic_result['passed']:
            logger.info(
                'questioner: Critic PASS attempt=%d spec=%.2f bloom=%.2f',
                attempt_idx + 1,
                critic_result['specificity_score'],
                critic_result['bloom_alignment_score'],
            )
            return (
                True, '', attempts, current_response,
                current_text, current_tier1, best_critic_data,
            )

        # Critic failed — log and prepare retry (unless last attempt)
        logger.info(
            'questioner: Critic FAIL attempt=%d critique=%r',
            attempt_idx + 1,
            critic_result['critique'][:120],
        )

        if attempt_idx >= CRITIC_MAX_RETRIES:
            break

        # Retry: append the critique as a retry reason for the Questioner
        retry_reason = f"critic feedback: {critic_result['critique']}"
        retry_prompt = _build_prompt(inp, blooms, retry_reason=retry_reason)
        retry_response = llm_call(
            retry_prompt,
            model='reasoning',
            expect_json=True,
            fallback={
                'question_text': current_text,
                'blooms_level': blooms,
                'difficulty': inp.difficulty,
            },
        )
        retry_text = (retry_response.get('question_text') or '').strip()

        # Re-run Tier 1 on the retry; only proceed if it still passes Tier 1
        retry_tier1 = validate_question(retry_text, recent_questions=inp.recent_questions)
        attempts += 1

        if retry_tier1.passed and retry_text:
            current_text = retry_text
            current_response = retry_response
            current_tier1 = retry_tier1
        else:
            logger.info(
                'questioner: Critic retry failed Tier 1 (%s) — keeping previous candidate.',
                retry_tier1.reason_string(),
            )
            # Don't update current_*; the best-seen so far is preserved
            break

    # All retries exhausted — return best-seen candidate
    logger.info(
        'questioner: Critic exhausted retries, returning best (score=%.2f)',
        best_critic_score,
    )
    return (
        best_critic_passed, best_critic_critique, attempts, best_response,
        best_text, best_tier1, best_critic_data,
    )


def _intent_label_from_kg(kg_signals: Optional[Dict]) -> str:
    """
    Best-effort intent label for the Critic. The Strategist's chosen intent
    isn't currently passed through QuestionerInput; until that's plumbed in,
    we infer one from the KG signals or fall back to a generic label.
    """
    if not kg_signals:
        return 'general_probe'
    if kg_signals.get('contradicts_code_alerts'):
        return 'challenge_contradiction'
    if kg_signals.get('depends_on_topics'):
        return 'exploring_alternatives'
    return 'general_probe'
