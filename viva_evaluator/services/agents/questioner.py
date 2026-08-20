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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Dict

from viva_evaluator.services.rag.retrieval import format_chunks_for_prompt

if TYPE_CHECKING:
    from viva_evaluator.services.pipeline.contracts import QuestionEvidencePackage


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
    module_chunks: List[Dict] = field(default_factory=list)           # Module materials chunks
    kg_signals: Optional[Dict] = None                 # Week 3: hybrid retrieval output
    difficulty: str = 'medium'                        # 'easy' | 'medium' | 'hard'
    target_bloom: Optional[str] = None                 # exact Strategist selection
    socratic_intent: str = ''                         # exact Strategist intent
    intent_prompt_hint: str = ''                      # how to express that intent
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
    session_id: Optional[str] = None
    evidence_package: Optional["QuestionEvidencePackage"] = None


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
    # Compatibility facade for legacy callers.  Generation and validation are
    # intentionally separate stages so a raw candidate can be consumed (for
    # example by speculative TTS) before validation finishes.
    from viva_evaluator.services.pipeline.stages.candidate_generation import (
        generate_question_candidate,
    )
    from viva_evaluator.services.pipeline.stages.question_validation import (
        validate_question_candidate,
    )

    candidate = generate_question_candidate(inp)
    validated = validate_question_candidate(
        inp,
        candidate,
        max_retries=max_retries,
        enable_critic=enable_critic,
    )
    return validated.to_legacy_dict()


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
   (d) A claim or diagram from their presentation slides or spoken transcript:
       "during your demo, you showed ...", "you explained in your presentation that ...", "your slide on <topic>"

   The question must NOT be a generic question that could be asked of any
   student on this topic. It must reference something concrete that this
   specific student wrote, built, showed, or said, drawn from the retrieved sources below.

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

3. KEEP IT SHORT AND SPOKEN. Target length: 15-30 words. Maximum 40.
   This is an ORAL exam — the student must hold the entire question in
   their head. You MUST use exactly TWO sentences if it helps keep it natural:
   Sentence 1 states the context. Sentence 2 asks the question. This creates a 
   natural pause for the Text-to-Speech engine.
       BAD  (too long, written-style, single sentence):
         "Considering your Zero Trust goal and the problem of compromised
          servers, how complete and architecturally sound is this single
          countermeasure against all confidentiality threats?"
       GOOD (short, spoken, two sentences):
         "You said AES-256-GCM protects confidentiality if the server is compromised.
          What threats does it not cover?"

4. AVOID quoting long phrases from the report. Paraphrase the student's
   idea in plain words instead of pasting their wording back at them.
   At most quote 3-5 words, not full sentences.

5. MODULE BOUNDARY (strict). If "Module Materials" context is provided below,
   you MUST restrict the theoretical depth of your question to the concepts 
   covered in those materials. Do NOT expect the student to know technologies
   or concepts outside these materials, UNLESS they specifically mention an
   external technology that is presented as an alternative in the Module Materials.

5. PLAIN CONVERSATIONAL LANGUAGE. Phrase it as a real examiner SPEAKS aloud across a coffee table.
   The student is a final-year CS student, so technical terms from THEIR project are fine ("encryption",
   "authentication", "API"). What you MUST avoid is academic paper register in the question itself.

   THE READ-ALOUD TEST: Imagine asking the question across a table while drinking coffee.
   If it sounds stiff or requires re-reading to understand, IT IS TOO COMPLEX. Rewrite it into plain English.

       BANNED STIFF PHRASES (DO NOT USE):         PREFER CONVERSATIONAL PLAIN WORDS:
         "facilitating the exchange"               "sharing files" / "sending data"
         "remains untrusted"                       "stays out of the loop" / "can't read it"
         "lifecycle of a file"                     "when a file is uploaded"
         "scrambled ciphertext"                    "encrypted files"
         "exfiltrate"                              "steal" / "leak" / "take"
         "mitigate"                                "reduce" / "handle" / "fix"
         "ascertain"                               "check" / "figure out"
         "elucidate"                               "explain" / "walk through"
         "ramifications"                            "effects" / "consequences"
         "architecturally sound"                    "good design" / "the right call"
         "in totality"                              "overall"
         "vis-à-vis"                                "compared to" / "against"
         "considering the implications of"          "given" / "with"
         "particularly those involving"             "especially when"
         "as it pertains to"                        "for"
         "in the context of"                        "for" / "when"

       FEW-SHOT EXAMPLES:
         BAD (Stiff / Paper-style):
           "You described a system where the server only handles scrambled ciphertext. Could you
            walk me through the lifecycle of a file to explain how the server remains untrusted
            while still facilitating the exchange?"
         GOOD (Conversational / Spoken):
           "You mentioned the server only receives encrypted files — walk me through how a user
            uploads a file without the server reading its content?"

   General rule: a 1-syllable verb beats a 4-syllable verb when both carry the same meaning.

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

11. NO EXTERNAL ALTERNATIVES. Do NOT ask the student to compare their choice
    against external technologies, frameworks, or algorithms that they did
    NOT explicitly mention in their report/code and that is not supplied as
    an authorized KG alternative evidence item.
    For example, if they used "AES-256-GCM", do NOT ask "Why didn't you use
    AES-CBC?" unless AES-CBC is explicitly in retrieved evidence. It is unfair
    to test them on an unsupported alternative.
"""


def _build_prompt(
    inp: QuestionerInput,
    blooms_level: str,
    retry_reason: Optional[str],
) -> str:
    from viva_evaluator.services.pipeline.evidence import (
        ensure_question_evidence_package,
    )

    evidence_package = ensure_question_evidence_package(inp)
    sources_block = format_chunks_for_prompt(inp.retrieved_chunks, max_chars=2400)

    # Hybrid retrieval - KG signals
    kg_block = ''
    if inp.kg_signals:
        from viva_evaluator.services.rag.retrieval import format_kg_signals_for_prompt
        kg_text = format_kg_signals_for_prompt(inp.kg_signals)
        if kg_text:
            kg_block = f"\nKNOWLEDGE GRAPH SIGNALS:\n{kg_text}\n"

    demo_block = ''
    presentation_references = evidence_package.of_type("presentation_segment")
    if presentation_references:
        demo_lines = [
            f"- [{reference.evidence_id}] {reference.content}"
            for reference in presentation_references
        ]
        demo_block = (
            "\nPRESENTATION DEMO EVIDENCE (what the student showed or said):\n"
            + "\n".join(demo_lines)
            + "\n"
        )


    if inp.is_first_question or not inp.previous_question:
        conversation_block = '(This is the opening question for this criterion.)'
    else:
        previous_answer_references = evidence_package.of_type("previous_answer")
        previous_answer_id = (
            previous_answer_references[0].evidence_id
            if previous_answer_references
            else "previous-answer:unavailable"
        )
        conversation_block = (
            f"PREVIOUS QUESTION (from you):\n{inp.previous_question}\n\n"
            f"STUDENT'S ANSWER [{previous_answer_id}] (their exact words):\n"
            f"{inp.previous_answer or '(no answer)'}"
        )

    hints_block = ''
    if inp.question_hints:
        hints_text = '\n'.join(f"- {h}" for h in inp.question_hints)
        hints_block = (
            "\nEXAMINER'S SUGGESTED FOCUS AREAS (use as loose guidelines, not exact wording):\n"
            f"{hints_text}\n"
        )

    strategy_block = ''
    if inp.socratic_intent or inp.intent_prompt_hint:
        strategy_block = (
            "\nSOCRATIC QUESTIONING STRATEGY:\n"
            f"Intent: {inp.socratic_intent or 'general_probe'}\n"
            f"Guidance: {inp.intent_prompt_hint or 'Ask one focused follow-up.'}\n"
            "Express this strategy in the question without naming the strategy itself.\n"
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

    module_block = ''
    if inp.module_chunks:
        module_text = format_chunks_for_prompt(inp.module_chunks)
        module_block = (
            "\nMODULE MATERIALS (Strict theoretical boundary — do not ask for knowledge outside this scope):\n"
            f"{module_text}\n"
        )

    evidence_catalog = '\n'.join(
        f"- {reference.evidence_id} ({reference.evidence_type})"
        for reference in evidence_package.references
    ) or "- (no attributable evidence available)"

    return f"""You are an academic viva examiner conducting an oral examination.

RUBRIC CRITERION:
Name: {inp.criterion_name}
Description: {inp.criterion_description or '(no description)'}

RETRIEVED SOURCES from the student's submission (the ONLY source of truth about
their project — every concrete reference must come from here):

{sources_block}
{kg_block}
{demo_block}
{module_block}
AVAILABLE EVIDENCE IDS (return only IDs from this list):
{evidence_catalog}

CONVERSATION CONTEXT:
{conversation_block}
{hints_block}{strategy_block}{retry_block}{clarify_block}{weak_grounding_block}
TARGET BLOOM'S LEVEL: {blooms_level}
PHRASING STYLE FOR THIS LEVEL: {bloom_phrasing}

{_ANCHORING_RULES}

Generate ONE viva question following all the rules above.

For source_reference_ids, list every evidence ID that directly supports a
specific factual anchor in the question. Never invent an ID. When LIMITED
SOURCE MATERIAL applies and the question makes no specific factual claim, an
empty list is allowed.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "question_text": "your question here",
    "source_reference_ids": ["one-or-more IDs from AVAILABLE EVIDENCE IDS"],
    "target_bloom": "{blooms_level}",
    "socratic_intent": "{inp.socratic_intent or 'general_probe'}"
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
