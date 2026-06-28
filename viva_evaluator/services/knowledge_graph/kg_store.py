"""
Knowledge graph persistence — serialize NetworkX graphs to/from PostgreSQL.

The graph is stored as JSON on SubmissionIndexStatus.kg_graph_json via
NetworkX's node_link_data() format, which round-trips cleanly.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# In-process cache: submission_id -> NetworkX graph
# The KG is loaded multiple times per turn (retrieval + contradiction edges);
# it never changes during a viva. Cache it, invalidate on save.
# =============================================================================
_KG_CACHE: dict = {}


# =============================================================================
# Edge tier definitions (4-tier confidence model from v3 spec)
# =============================================================================

TIER_EXAMINER_APPROVED = 1   # T1: examiner approved (or AST/SonarQube facts)
TIER_LLM_DRAFT          = 2   # T2: LLM-drafted, awaiting examiner review
TIER_WEB_VALIDATED      = 3   # T3: web-researched, relevance ≥ 0.6
TIER_RAW_WEB            = 4   # T4: raw web search, audit-only


# =============================================================================
# Public API — save / load
# =============================================================================

def save_kg_for_submission(submission, graph) -> int:
    """
    Persist a NetworkX DiGraph to the submission's index status row.

    Returns:
        Number of edges saved.
    """
    import networkx as nx
    from viva_evaluator.models import SubmissionIndexStatus

    if graph is None:
        return 0

    data = nx.node_link_data(graph)
    index_status, _ = SubmissionIndexStatus.objects.get_or_create(submission=submission)
    index_status.kg_graph_json = data
    index_status.save(update_fields=['kg_graph_json'])

    # KG changed — drop stale cache
    _KG_CACHE.pop(str(submission.id), None)

    n_edges = graph.number_of_edges()
    n_nodes = graph.number_of_nodes()
    logger.info(
        'save_kg_for_submission: submission=%s nodes=%d edges=%d',
        submission.id, n_nodes, n_edges,
    )
    return n_edges


def load_kg_for_submission(submission):
    """
    Load the graph for a submission. Returns None if no graph stored.
    Cached in-process; invalidated on save.
    """
    import networkx as nx
    from viva_evaluator.models import SubmissionIndexStatus

    sub_id = str(submission.id)
    if sub_id in _KG_CACHE:
        return _KG_CACHE[sub_id]

    try:
        index_status = SubmissionIndexStatus.objects.get(submission=submission)
    except SubmissionIndexStatus.DoesNotExist:
        return None

    data = index_status.kg_graph_json
    if not data:
        return None

    try:
        graph = nx.node_link_graph(data, directed=True)
        _KG_CACHE[sub_id] = graph
        return graph
    except Exception as exc:
        logger.warning('load_kg_for_submission: failed to deserialize: %s', exc)
        return None


def invalidate_kg_cache(submission) -> None:
    """Drop the cached graph for a submission."""
    _KG_CACHE.pop(str(submission.id), None)


# =============================================================================
# Retrieval — what the viva loop calls each turn
# =============================================================================

def retrieve_edges_for_topics(
    submission,
    topics: List[str],
    edge_types: Optional[List[str]] = None,
    min_tier: int = TIER_RAW_WEB,
) -> List[Dict]:
    """
    Return all edges in the graph involving any of the given topic nodes.

    Args:
        submission:  ProjectSubmission.
        topics:      Node names to look up (e.g., ['jwt', 'sqlite']).
                     Matching is case-insensitive.
        edge_types:  If set, only return edges of these types.
        min_tier:    Lowest tier to return (1 = highest, 4 = audit-only).
                     Default: TIER_RAW_WEB returns everything.

    Returns:
        [
            {
                'edge_type': 'DEPENDS_ON',
                'source':    'auth.views',
                'target':    'jwt',
                'tier':      1,
                'attrs':     {...},  # other edge attributes
            },
            ...
        ]
    """
    graph = load_kg_for_submission(submission)
    if graph is None:
        return []

    topics_lower = {t.lower() for t in topics}
    edges: List[Dict] = []

    for u, v, data in graph.edges(data=True):
        u_lower = str(u).lower()
        v_lower = str(v).lower()
        if u_lower not in topics_lower and v_lower not in topics_lower:
            continue

        edge_type = data.get('edge_type', 'UNKNOWN')
        if edge_types and edge_type not in edge_types:
            continue

        tier = int(data.get('tier', TIER_RAW_WEB))
        if tier > min_tier:
            continue

        edges.append({
            'edge_type': edge_type,
            'source':    u,
            'target':    v,
            'tier':      tier,
            'attrs':     {k: val for k, val in data.items() if k not in ('edge_type', 'tier')},
        })

    return edges


def retrieve_contradicts_code_edges(submission) -> List[Dict]:
    """
    Convenience: return all CONTRADICTS_CODE edges for this submission.
    Highest priority during a viva — the Strategist always checks these.
    """
    graph = load_kg_for_submission(submission)
    if graph is None:
        return []

    edges: List[Dict] = []
    for u, v, data in graph.edges(data=True):
        if data.get('edge_type') == 'CONTRADICTS_CODE':
            edges.append({
                'edge_type': 'CONTRADICTS_CODE',
                'source':    u,    # code finding (e.g., 'hardcoded_credentials')
                'target':    v,    # report claim (e.g., 'secure authentication')
                'tier':      data.get('tier', TIER_EXAMINER_APPROVED),
                'attrs':     {k: val for k, val in data.items()
                              if k not in ('edge_type', 'tier')},
            })
    return edges
