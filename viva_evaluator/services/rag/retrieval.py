"""
Retrieval — semantic search wrapper called by the viva agents.

This is the public RAG API. The Questioner and Analyzer call retrieve_for_turn().
The indexing pipeline calls retrieve_for_indexing() to verify after a build.

Query construction:
    For a viva turn, the query is "criterion_name + last_answer". This pulls
    chunks that match BOTH the topic (via criterion) AND what the student
    just said (so follow-up questions stay anchored to the conversation).
"""

import logging
from typing import List, Dict, Optional

from viva_evaluator.services.rag.embeddings import embed_text
from viva_evaluator.services.rag.vector_store import (
    SubmissionVectorStore,
    load_index_for_submission,
)

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant (standard default).
_RRF_K0 = 60


# =============================================================================
# Public API
# =============================================================================

def retrieve_for_turn(
    submission,
    criterion_name: str,
    criterion_description: str = '',
    last_answer: str = '',
    top_k: int = 3,
) -> List[Dict]:
    """
    Pull the top-k chunks most relevant to the current viva turn, using HYBRID
    retrieval (dense SBERT + BM25 lexical, fused via RRF) followed by an
    optional cross-encoder rerank.

    Pipeline:
        1. Dense search (FAISS)        → wide candidate set, keeps cosine 'score'
        2. Lexical search (BM25)       → wide candidate set on exact terms
        3. Reciprocal Rank Fusion      → merge both rankings
        4. Cross-encoder rerank        → precise final top_k (if enabled)

    Returns:
        List of chunk dicts (each with 'text', 'source', 'section', 'score').
        'score' is preserved as the dense cosine (0 for lexical-only hits) so
        downstream grounding checks (B3) keep working. Empty list if not indexed.
    """
    store = load_index_for_submission(submission)
    if store is None or store.num_chunks == 0:
        logger.info('retrieve_for_turn: no index for submission=%s', submission.id)
        return []

    query = _build_query(criterion_name, criterion_description, last_answer)
    query_vec = embed_text(query)

    # Wide candidate set for fusion/rerank, then narrow to top_k.
    candidate_k = min(max(top_k * 5, 15), store.num_chunks)

    # 1. Dense candidates (chunk dicts with cosine 'score' + 'chunk_idx')
    dense_hits = store.search(query_vec, top_k=candidate_k)

    # 2. Lexical candidates (BM25 positions → chunk dicts)
    lex_hits: List[Dict] = []
    try:
        from viva_evaluator.services.rag.lexical import lexical_search
        for pos, bm25_score in lexical_search(str(submission.id), store.chunks, query, k=candidate_k):
            if 0 <= pos < store.num_chunks:
                c = dict(store.chunks[pos])
                c['bm25_score'] = bm25_score
                c.setdefault('score', 0.0)   # no dense cosine unless also in dense_hits
                lex_hits.append(c)
    except Exception as exc:
        logger.warning('retrieve_for_turn: lexical stage failed (%s); dense-only.', exc)

    # 3. Fuse the two rankings
    fused = _rrf_fuse(dense_hits, lex_hits)

    if not fused:
        return dense_hits[:top_k]

    # 4. Cross-encoder rerank the fused candidates → final top_k
    # Skip reranking for small submissions (<=10 chunks) to eliminate CPU overhead
    try:
        from viva_evaluator.services.rag.rerank import rerank_chunks, reranker_enabled
        if reranker_enabled() and len(fused) > 10:
            return rerank_chunks(query, fused[:candidate_k], top_k)
    except Exception as exc:
        logger.warning('retrieve_for_turn: rerank stage failed (%s); fused order.', exc)

    return fused[:top_k]


def _rrf_fuse(dense_hits: List[Dict], lex_hits: List[Dict]) -> List[Dict]:
    """
    Reciprocal Rank Fusion of two ranked chunk lists, keyed by 'chunk_idx'.

    RRF score = sum over lists of 1 / (k0 + rank). The representative chunk
    prefers the dense copy (so the cosine 'score' is preserved for grounding);
    lexical-only chunks keep score=0.0.
    """
    fused: Dict = {}

    def _key(c: Dict):
        return c.get('chunk_idx', id(c))

    for rank, c in enumerate(dense_hits):
        k = _key(c)
        entry = fused.setdefault(k, {'chunk': dict(c), 'rrf': 0.0})
        entry['rrf'] += 1.0 / (_RRF_K0 + rank)

    for rank, c in enumerate(lex_hits):
        k = _key(c)
        if k in fused:
            # Already have the dense copy (with cosine score); just add weight
            # and carry the bm25 score for transparency.
            fused[k]['rrf'] += 1.0 / (_RRF_K0 + rank)
            fused[k]['chunk'].setdefault('bm25_score', c.get('bm25_score'))
        else:
            entry = fused.setdefault(k, {'chunk': dict(c), 'rrf': 0.0})
            entry['rrf'] += 1.0 / (_RRF_K0 + rank)

    ordered = sorted(fused.values(), key=lambda e: e['rrf'], reverse=True)
    out = []
    for e in ordered:
        chunk = e['chunk']
        chunk['rrf_score'] = round(e['rrf'], 6)
        out.append(chunk)
    return out


def retrieve_for_indexing(
    submission,
    sample_query: str = 'project overview',
    top_k: int = 1,
) -> List[Dict]:
    """
    Sanity-check retrieval immediately after indexing — used by health checks.
    """
    store = load_index_for_submission(submission)
    if store is None:
        return []
    query_vec = embed_text(sample_query)
    return store.search(query_vec, top_k=top_k)


# =============================================================================
# Helpers
# =============================================================================

def _build_query(
    criterion_name: str,
    criterion_description: str,
    last_answer: str,
) -> str:
    """
    Compose a retrieval query from the rubric criterion + recent conversation.
    Truncate last_answer to keep query semantically focused.
    """
    parts = [criterion_name.strip()]
    if criterion_description:
        parts.append(criterion_description.strip())
    if last_answer:
        # Limit to first ~300 chars — the topic is in the early part of an answer
        parts.append(last_answer.strip()[:300])
    return ' '.join(p for p in parts if p)


def format_chunks_for_prompt(chunks: List[Dict], max_chars: int = 2400) -> str:
    """
    Format retrieved chunks as a single string suitable for injecting into
    an LLM prompt. Includes section labels so the agent can cite them.

    Args:
        chunks:    List from retrieve_for_turn().
        max_chars: Truncate total to this many characters.

    Returns:
        Multi-section string.
    """
    if not chunks:
        return '(no relevant content retrieved from the submission)'

    parts: List[str] = []
    used = 0
    for i, c in enumerate(chunks, start=1):
        source = c.get('source', '?')
        section = c.get('section', '?')

        if source == 'figure':
            label = f'figure in section "{section}"'
        elif source == 'code':
            label = f'code in {section}'
        else:
            label = f'report section "{section}"'

        header = f"[Source {i}: {label}]"
        body = (c.get('text') or '').strip()
        block = f"{header}\n{body}"
        if used + len(block) > max_chars and parts:
            break
        parts.append(block)
        used += len(block) + 2  # +2 for the join newlines

    return '\n\n'.join(parts)


# =============================================================================
# Hybrid retrieval — Week 3 addition: combine FAISS chunks with KG signals.
# =============================================================================

def retrieve_hybrid_for_turn(
    submission,
    criterion_name: str,
    criterion_description: str = '',
    last_answer: str = '',
    top_k: int = 3,
) -> Dict:
    """
    Single retrieval call for the viva loop. Returns BOTH semantic chunks
    (FAISS) AND knowledge graph signals (Week 3+).

    The Questioner uses chunks for anchoring and KG signals for content
    direction (e.g., raise an active CONTRADICTS_CODE alert).

    Returns:
        {
            'chunks':                     [chunk dicts from FAISS],
            'contradicts_code_alerts':    [edges with code_finding ↔ report_claim],
            'depends_on_topics':          [unique imported modules],
            'kg_available_for_topic':     bool,
        }
    """
    from viva_evaluator.services.knowledge_graph.kg_store import (
        load_kg_for_submission,
        retrieve_contradicts_code_edges,
    )

    # 1. FAISS chunks (existing)
    chunks = retrieve_for_turn(
        submission=submission,
        criterion_name=criterion_name,
        criterion_description=criterion_description,
        last_answer=last_answer,
        top_k=top_k,
    )

    # 2. KG signals
    contradicts = retrieve_contradicts_code_edges(submission)

    graph = load_kg_for_submission(submission)
    depends_on_topics: List[str] = []
    alternative_edges: List[Dict] = []
    _seen_alts = set()
    if graph is not None:
        for u, v, data in graph.edges(data=True):
            et = data.get('edge_type')
            if et == 'DEPENDS_ON':
                depends_on_topics.append(str(v))
            elif et in ('ALTERNATIVE_TO', 'BETTER_FOR_SCALE'):
                base_tech = data.get('base_tech') or str(v)
                alternative = data.get('alternative') or str(u)
                dedup_key = (base_tech.lower(), alternative.lower())
                if dedup_key in _seen_alts:
                    continue
                _seen_alts.add(dedup_key)
                alternative_edges.append({
                    'edge_type':   et,
                    'base_tech':   base_tech,
                    'alternative': alternative,
                    'rationale':   data.get('rationale', ''),
                    'trigger':     data.get('trigger', ''),
                    'tier':        data.get('tier', 2),
                })
        depends_on_topics = sorted(set(depends_on_topics))[:20]  # cap for prompt size
        # Prefer examiner-approved (tier 1) edges; cap for prompt size.
        alternative_edges.sort(key=lambda e: e.get('tier', 2))
        alternative_edges = alternative_edges[:3]

    kg_available = bool(graph and graph.number_of_edges() > 0)

    return {
        'chunks':                  chunks,
        'contradicts_code_alerts': contradicts,
        'depends_on_topics':       depends_on_topics,
        'alternative_edges':       alternative_edges,
        'kg_available_for_topic':  kg_available,
    }


def format_kg_signals_for_prompt(retrieval_result: Dict) -> str:
    """
    Render KG signals as a short prompt section the Questioner can use.
    Returns '' if there are no relevant signals.
    """
    parts: List[str] = []

    contradicts = retrieval_result.get('contradicts_code_alerts') or []
    if contradicts:
        # The Strategist (Week 5) will pick which alert to surface; for now
        # we mention the highest-severity one.
        ranked = sorted(
            contradicts,
            key=lambda e: 0 if e.get('attrs', {}).get('severity') == 'high' else 1,
        )
        top = ranked[0]
        parts.append(
            "⚠ AUTHORSHIP/INTEGRITY ALERT (priority signal):\n"
            f"  Code finding: '{top['source']}' "
            f"(detail: {top.get('attrs', {}).get('finding_detail', '')[:200]})\n"
            f"  Contradicts report claim: '{top['target']}' "
            f"(excerpt: \"{top.get('attrs', {}).get('claim_excerpt', '')[:200]}\")\n"
            "If this alert is relevant to the current criterion, the question SHOULD\n"
            "challenge this contradiction directly."
        )

    depends_on = retrieval_result.get('depends_on_topics') or []
    if depends_on:
        parts.append(
            "DEPENDENCIES (the student's code imports these):\n"
            f"  {', '.join(depends_on)}"
        )

    # D3: surface concrete alternative technologies so the Questioner can ask a
    # grounded "why X and not Y?" comparison instead of a vague one.
    alternatives = retrieval_result.get('alternative_edges') or []
    if alternatives:
        lines = []
        for e in alternatives:
            base = e.get('base_tech', '')
            alt = e.get('alternative', '')
            rationale = (e.get('rationale') or '').strip()
            if e.get('edge_type') == 'BETTER_FOR_SCALE':
                note = f"{alt} is often preferred at scale"
            else:
                note = f"{alt} is a common alternative"
            if rationale:
                note += f" — {rationale[:160]}"
            lines.append(f"  - The student used {base}. {note}.")
        parts.append(
            "ALTERNATIVE TECHNOLOGIES (you MAY ask why they chose theirs over one "
            "of these — a comparative 'why X and not Y' question):\n"
            + '\n'.join(lines)
        )

    return '\n\n'.join(parts) if parts else ''
