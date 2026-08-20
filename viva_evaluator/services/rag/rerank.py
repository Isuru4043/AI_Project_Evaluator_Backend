"""
Cross-encoder reranker (B2).

WHY:
    Bi-encoders (SBERT) and BM25 score the query and each chunk INDEPENDENTLY,
    which is fast but imprecise. A cross-encoder reads the (query, chunk) PAIR
    together and scores true relevance — far more accurate for picking the
    final top-k. We retrieve a wide candidate set cheaply (dense + BM25), then
    rerank just those candidates with the cross-encoder.

COST / CONTROL:
    - Lazy-loaded singleton (~80MB model, CPU-friendly MiniLM cross-encoder).
    - Toggle via env RAG_RERANK_ENABLED=false to disable (e.g., for tests or
      latency-sensitive runs). Defaults to enabled.
    - Any failure falls back gracefully to the pre-rerank order.
"""

import hashlib
import json
import logging
import os
from typing import Dict, List

from viva_evaluator.services.rag.cache import (
    BoundedTTLCache,
    env_float,
    env_int,
)

logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv('RAG_RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

_model = None
_load_failed = False
_RERANK_CACHE = BoundedTTLCache(
    max_entries=env_int("RAG_RERANK_CACHE_SIZE", 256),
    ttl_seconds=env_float("RAG_RERANK_CACHE_TTL_SECONDS", 900.0),
)


def reranker_enabled() -> bool:
    return os.getenv('RAG_RERANK_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')


def _get_model():
    global _model, _load_failed
    if _model is None and not _load_failed:
        try:
            from sentence_transformers import CrossEncoder
            logger.info('Loading cross-encoder reranker: %s', RERANK_MODEL)
            _model = CrossEncoder(RERANK_MODEL)
            logger.info('Cross-encoder reranker loaded.')
        except Exception as exc:
            _load_failed = True
            logger.warning('Reranker load failed (%s); reranking disabled.', exc)
    return _model


def rerank_chunks(
    query_text: str,
    chunks: List[Dict],
    top_k: int,
    *,
    cache_namespace: str = "",
) -> List[Dict]:
    """
    Rerank candidate chunks by cross-encoder relevance to query_text and return
    the top_k. Preserves each chunk's existing fields (incl. dense 'score').
    Falls back to the input order (truncated) if reranking is unavailable.
    """
    if not chunks:
        return []
    if not reranker_enabled():
        return [dict(chunk) for chunk in chunks[:top_k]]

    cache_key = _rerank_cache_key(
        query_text,
        chunks,
        top_k,
        cache_namespace,
    )
    hit, cached = _RERANK_CACHE.get(cache_key)
    if hit and cached is not None:
        return [dict(chunk) for chunk in cached]

    model = _get_model()
    if model is None:
        return [dict(chunk) for chunk in chunks[:top_k]]

    try:
        working_chunks = [dict(chunk) for chunk in chunks]
        pairs = [(query_text, c.get('text', '')) for c in working_chunks]
        scores = model.predict(pairs)
        for c, s in zip(working_chunks, scores):
            c['rerank_score'] = float(s)
        ranked = sorted(
            working_chunks,
            key=lambda c: c.get('rerank_score', 0.0),
            reverse=True,
        )[:top_k]
        _RERANK_CACHE.set(cache_key, tuple(dict(chunk) for chunk in ranked))
        return [dict(chunk) for chunk in ranked]
    except Exception as exc:
        logger.warning('rerank_chunks failed (%s); using pre-rerank order.', exc)
        return [dict(chunk) for chunk in chunks[:top_k]]


def invalidate_rerank_cache(cache_namespace: str = "") -> None:
    """Invalidate all reranks, or only those for one submission namespace."""
    normalized = str(cache_namespace or '')
    if not normalized:
        _RERANK_CACHE.invalidate()
        return
    _RERANK_CACHE.invalidate(lambda key: key[0] == normalized)


def _rerank_cache_key(
    query_text: str,
    chunks: List[Dict],
    top_k: int,
    cache_namespace: str,
):
    candidate_payload = [
        {
            'evidence_id': chunk.get('evidence_id'),
            'chunk_idx': chunk.get('chunk_idx'),
            'source': chunk.get('source'),
            'section': chunk.get('section'),
            'text': chunk.get('text', ''),
        }
        for chunk in chunks
    ]
    candidate_digest = hashlib.sha256(
        json.dumps(
            candidate_payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode('utf-8')
    ).hexdigest()
    query_digest = hashlib.sha256(
        str(query_text or '').encode('utf-8')
    ).hexdigest()
    return (
        str(cache_namespace or ''),
        RERANK_MODEL,
        query_digest,
        candidate_digest,
        max(1, int(top_k)),
    )
