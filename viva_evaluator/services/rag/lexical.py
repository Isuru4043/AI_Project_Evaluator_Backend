"""
Lexical (BM25) retrieval — keyword search to complement dense embeddings.

WHY (B1 Hybrid retrieval):
    Dense embeddings (SBERT) capture meaning but can miss EXACT tokens that
    matter in technical reports — function names, library names, identifiers
    like `validate_token` or `AES-256-GCM`. BM25 is a classic lexical ranker
    that nails exact-term matches. Fusing both (in retrieval.py) gives results
    that are strong on BOTH meaning and specific terminology.

CACHING:
    A submission's chunks are frozen during a viva, so the BM25 index is built
    once and cached, keyed by (submission_id, chunk_count) — if the chunk count
    changes (e.g., code chunks appended), the key changes and we rebuild.
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# submission_id -> (chunk_count, BM25Okapi, tokenized_corpus)
_BM25_CACHE: dict = {}

_TOKEN_RE = re.compile(r'[a-z0-9_]+')


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or '').lower())


def _get_bm25(submission_id: str, chunks: List[Dict]):
    """Build or fetch a cached BM25 index for this submission's chunks."""
    from rank_bm25 import BM25Okapi

    cached = _BM25_CACHE.get(submission_id)
    if cached is not None and cached[0] == len(chunks):
        return cached[1]

    corpus = [_tokenize(c.get('text', '')) for c in chunks]
    # BM25Okapi requires a non-empty corpus; guard empty docs.
    safe_corpus = [toks if toks else ['__empty__'] for toks in corpus]
    bm25 = BM25Okapi(safe_corpus)
    _BM25_CACHE[submission_id] = (len(chunks), bm25, safe_corpus)
    logger.info('lexical: built BM25 index submission=%s docs=%d', submission_id, len(chunks))
    return bm25


def lexical_search(
    submission_id: str,
    chunks: List[Dict],
    query_text: str,
    k: int = 15,
) -> List[Tuple[int, float]]:
    """
    Return the top-k chunk POSITIONS (index into `chunks`) by BM25 score.

    Returns:
        List of (position, bm25_score) sorted by score desc. Empty on failure.
    """
    if not chunks or not query_text.strip():
        return []

    try:
        bm25 = _get_bm25(submission_id, chunks)
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []
        scores = bm25.get_scores(query_tokens)
    except Exception as exc:
        logger.warning('lexical_search failed: %s', exc)
        return []

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    # Drop zero-score hits (no lexical overlap at all)
    ranked = [(pos, float(s)) for pos, s in ranked[:k] if s > 0.0]
    return ranked


def invalidate(submission_id: str) -> None:
    _BM25_CACHE.pop(str(submission_id), None)
