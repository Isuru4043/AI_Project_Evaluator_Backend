"""
Knowledge Graph layer — symbolic retrieval to complement FAISS semantic retrieval.

WEEK 3 SCOPE:
    - DEPENDS_ON edges from import graph (always Tier 1: from AST, no validation)
    - CONTRADICTS_CODE edges from SonarQube findings vs report claims (Tier 1)
    - 4-tier confidence model storage (T1 here, T2/T3 added in Week 4)

WEEK 4 (planned):
    - Auto-drafted domain briefs (T2)
    - Web research edges (T3/T4)
    - Examiner approval workflow

EDGE VOCABULARY (strict):
    - DEPENDS_ON       : source → target (e.g., 'auth.views' → 'jwt')
    - CONTRADICTS_CODE : code finding → report claim
    - ALTERNATIVE_TO   : (Week 4) bidirectional comparable technologies
    - BETTER_FOR_SCALE : (Week 4) directional preferred-for-scale
    - COMMONLY_REPLACES: (Week 4) modern → legacy

PERSISTENCE:
    Each submission's graph is stored as JSON on SubmissionIndexStatus.kg_graph_json
    using NetworkX's node_link_data() format.
"""

from viva_evaluator.services.knowledge_graph.kg_builder import build_kg_for_submission
from viva_evaluator.services.knowledge_graph.kg_store import (
    save_kg_for_submission,
    load_kg_for_submission,
    retrieve_edges_for_topics,
)

__all__ = [
    'build_kg_for_submission',
    'save_kg_for_submission',
    'load_kg_for_submission',
    'retrieve_edges_for_topics',
]
