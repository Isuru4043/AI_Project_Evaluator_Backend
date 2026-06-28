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

import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

RERANK_MODEL = os.getenv('RAG_RERANK_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

_model = None
_load_failed = False


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


def rerank_chunks(query_text: str, chunks: List[Dict], top_k: int) -> List[Dict]:
    """
    Rerank candidate chunks by cross-encoder relevance to query_text and return
    the top_k. Preserves each chunk's existing fields (incl. dense 'score').
    Falls back to the input order (truncated) if reranking is unavailable.
    """
    if not chunks:
        return []
    if not reranker_enabled():
        return chunks[:top_k]

    model = _get_model()
    if model is None:
        return chunks[:top_k]

    try:
        pairs = [(query_text, c.get('text', '')) for c in chunks]
        scores = model.predict(pairs)
        for c, s in zip(chunks, scores):
            c['rerank_score'] = float(s)
        ranked = sorted(chunks, key=lambda c: c.get('rerank_score', 0.0), reverse=True)
        return ranked[:top_k]
    except Exception as exc:
        logger.warning('rerank_chunks failed (%s); using pre-rerank order.', exc)
        return chunks[:top_k]
