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
    Pull the top-k chunks most relevant to the current viva turn.

    Args:
        submission:            ProjectSubmission.
        criterion_name:        e.g. "Database Design".
        criterion_description: rubric description for added context.
        last_answer:           student's latest answer (empty for first question).
        top_k:                 chunks to return.

    Returns:
        List of chunk dicts (each with 'text', 'source', 'section', 'score').
        Empty list if submission is not yet indexed.
    """
    store = load_index_for_submission(submission)
    if store is None or store.num_chunks == 0:
        logger.info('retrieve_for_turn: no index for submission=%s', submission.id)
        return []

    query = _build_query(criterion_name, criterion_description, last_answer)
    query_vec = embed_text(query)
    return store.search(query_vec, top_k=top_k)


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
    if graph is not None:
        for u, v, data in graph.edges(data=True):
            if data.get('edge_type') == 'DEPENDS_ON':
                depends_on_topics.append(str(v))
        depends_on_topics = sorted(set(depends_on_topics))[:20]  # cap for prompt size

    kg_available = bool(graph and graph.number_of_edges() > 0)

    return {
        'chunks':                  chunks,
        'contradicts_code_alerts': contradicts,
        'depends_on_topics':       depends_on_topics,
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

    return '\n\n'.join(parts) if parts else ''
