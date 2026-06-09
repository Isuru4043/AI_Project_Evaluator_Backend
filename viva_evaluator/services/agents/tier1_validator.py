"""
Tier 1 validator — programmatic checks on a generated question.

Runs in <100ms with no LLM call. Catches the most common failure modes
before paying for a Critic call (Week 6).

CHECKS:
    1. Word count in [15, 80]
    2. Exactly one question mark
    3. Contains an anchor phrase ("you mentioned" / "in your" / "your code" / ...)
    4. Cosine similarity to recent questions < 0.82 (anti-repetition)
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration — tuneable from settings.py later if needed.
# =============================================================================

MIN_WORDS = 12
MAX_WORDS = 60       # tightened from 80 — viva questions should be spoken-length
MAX_QUESTION_MARKS = 1
SIMILARITY_THRESHOLD = 0.82

ANCHOR_PATTERNS = [
    # Direct references to student's words/work
    r'\byou mentioned\b',
    r'\byou said\b',
    r'\byou described\b',
    r'\byou wrote\b',
    r'\byou explained\b',
    r'\byou stated\b',
    r'\byou (?:claim|claimed|argue|argued|note|noted)\b',

    # References to student's report/document/work
    r'\bin your\b',                # "in your section / report / methodology / code"
    r'\bfrom your\b',
    r'\bwithin your\b',
    r'\baccording to your\b',
    r'\byour report\b',
    r'\byour code\b',
    r'\byour implementation\b',
    r'\byour proposal\b',
    r'\byour design\b',
    r'\byour approach\b',
    r'\byour methodology\b',
    r'\byour solution\b',
    r'\byour project\b',
    # Generic catch: "your <word>" is an anchor to the student's own work
    # (e.g. "your TOFU pinning", "your RSA-OAEP exchange", "your encryption
    # layer"). Genuine over-generic questions are still caught by the
    # Critic's specificity score in Tier 2 — Tier 1 only ensures the
    # question is *about the student*, not about the topic in the abstract.
    r'\byour\s+\w+',

    # References to specific report artifacts (figures, tables, sections)
    r'\b(?:table|figure|fig\.?|section|chapter|diagram|appendix)\s+\d',  # "Table 4.1", "Section 3"
    r'\b(?:listed|shown|described|defined|illustrated|outlined) (?:in|on)\b',  # "listed in Table 4.1"
    r'\bas (?:described|defined|outlined|shown|noted) (?:in|on)\b',

    # Visual anchors
    r'\blooking at your\b',
    r'\bgiven your\b',
    r'\bbased on your\b',
    r'\bconsidering your\b',
    r'\breferring to your\b',
    r'\bregarding your\b',
]
_ANCHOR_RE = re.compile('|'.join(ANCHOR_PATTERNS), re.IGNORECASE)


# =============================================================================
# Document-location patterns — questions must NEVER reference these because
# the student doesn't have the report in front of them during a real viva.
# =============================================================================

_DOC_LOCATION_PATTERNS = [
    r'\bon page\s+\d',                       # "on page 5"
    r'\bpage\s+\d+\s+of\b',                  # "page 5 of"
    r'\b(?:table|figure|fig\.?)\s+\d',       # "Table 4.1", "Figure 3", "Fig. 2"
    r'\bsection\s+\d',                       # "Section 3.1"
    r'\bchapter\s+\d',                       # "Chapter 4"
    r'\bappendix\s+[A-Z\d]',                 # "Appendix A"
    r'\bdiagram\s+\d',                       # "Diagram 2"
    r'\[cite:\s*\d',                         # "[cite: 9]"
    r'\bas (?:shown|stated|described|listed) (?:on|in) page\b',
]
_DOC_LOCATION_RE = re.compile('|'.join(_DOC_LOCATION_PATTERNS), re.IGNORECASE)


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class Tier1Result:
    passed: bool
    failures: List[str]            # human-readable failure reasons
    similarity_to_recent: float    # max cosine similarity vs recent_questions
    word_count: int

    def reason_string(self) -> str:
        return '; '.join(self.failures) if self.failures else 'passed'


# =============================================================================
# Public API
# =============================================================================

def validate_question(
    question_text: str,
    recent_questions: Optional[List[str]] = None,
) -> Tier1Result:
    """
    Run all Tier 1 checks on a candidate question.

    Args:
        question_text:    The generated question string.
        recent_questions: Last N question strings from this session (for anti-repeat).

    Returns:
        Tier1Result with passed flag + list of failure descriptions.
    """
    failures: List[str] = []
    text = (question_text or '').strip()

    # Check 1: word count
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        failures.append(f'too_short ({word_count} words, need {MIN_WORDS})')
    elif word_count > MAX_WORDS:
        failures.append(f'too_long ({word_count} words, max {MAX_WORDS})')

    # Check 2: question mark count
    qmark_count = text.count('?')
    if qmark_count == 0:
        failures.append('missing_question_mark')
    elif qmark_count > MAX_QUESTION_MARKS:
        failures.append(f'compound_question ({qmark_count} question marks)')

    # Check 3: anchoring
    if not _ANCHOR_RE.search(text):
        failures.append('missing_anchor')

    # Check 4: no document-location references (student is in oral exam,
    # has no report in front of them — references like "Table 4.1" or
    # "page 5" make the question impossible to answer without the document).
    doc_loc_match = _DOC_LOCATION_RE.search(text)
    if doc_loc_match:
        failures.append(f'document_location_reference ({doc_loc_match.group(0)!r})')

    # Check 5: similarity to recent
    similarity = _max_similarity(text, recent_questions or [])
    if similarity > SIMILARITY_THRESHOLD:
        failures.append(f'too_similar_to_recent (sim={similarity:.2f})')

    result = Tier1Result(
        passed=not failures,
        failures=failures,
        similarity_to_recent=similarity,
        word_count=word_count,
    )

    if failures:
        logger.info('tier1_validator: FAIL %s for question=%r', failures, text[:120])
    else:
        logger.info('tier1_validator: PASS sim=%.2f words=%d', similarity, word_count)

    return result


# =============================================================================
# Similarity helper — uses SBERT embeddings (already loaded for retrieval).
#
# We cache embeddings of recent questions keyed by their text so that the
# repeated Tier 1 validations within a single turn (initial + Tier 1 retry
# + Critic-loop re-validations) don't re-embed the same strings. Only the
# candidate (which changes each attempt) is embedded fresh.
# =============================================================================

# Bounded LRU-ish cache: text -> normalized embedding (np.ndarray)
_EMB_CACHE: dict = {}
_EMB_CACHE_MAX = 256


def _get_cached_embeddings(texts: List[str]):
    """
    Return embeddings for `texts`, using the module cache for any already
    seen and batch-embedding only the misses.
    """
    from viva_evaluator.services.rag.embeddings import embed_texts
    import numpy as np

    missing = [t for t in texts if t not in _EMB_CACHE]
    if missing:
        vecs = embed_texts(missing)
        for t, v in zip(missing, vecs):
            _EMB_CACHE[t] = v
        # Trim cache if it grows too large
        if len(_EMB_CACHE) > _EMB_CACHE_MAX:
            # drop oldest ~half (dicts preserve insertion order in py3.7+)
            for old_key in list(_EMB_CACHE.keys())[: _EMB_CACHE_MAX // 2]:
                _EMB_CACHE.pop(old_key, None)

    return np.array([_EMB_CACHE[t] for t in texts], dtype='float32')


def _max_similarity(candidate: str, recent: List[str]) -> float:
    """
    Returns the max cosine similarity between `candidate` and any string in `recent`.
    Returns 0.0 if `recent` is empty or embedding fails.

    Recent-question embeddings are cached; only the candidate is embedded
    fresh each call (it changes on every retry).
    """
    if not recent:
        return 0.0

    try:
        import numpy as np

        # Recent questions come from the cache (stable within a turn);
        # candidate is embedded fresh and also cached for any later reuse.
        recent_vecs = _get_cached_embeddings(recent)
        cand_vec = _get_cached_embeddings([candidate])[0]

        if recent_vecs.shape[0] == 0:
            return 0.0

        # Inner product == cosine because vectors are normalized
        sims = recent_vecs @ cand_vec
        return float(np.max(sims))
    except Exception as exc:
        logger.warning('tier1_validator similarity check failed: %s', exc)
        return 0.0
