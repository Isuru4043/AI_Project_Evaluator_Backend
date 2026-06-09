"""
Knowledge graph builder — constructs the per-submission graph.

WEEK 3 INPUTS:
    1. AST imports → DEPENDS_ON edges (verified Tier 1)
    2. SonarQube findings + report claims → CONTRADICTS_CODE edges (Tier 1)

WEEK 4 will add:
    - Auto-drafted technology briefs → ALTERNATIVE_TO / BETTER_FOR_SCALE (T2)
    - Web research → T3/T4 edges
"""

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================

def build_kg_for_submission(
    submission,
    code_index_result: Optional[dict] = None,
    code_submission=None,
    report_chunks: Optional[list] = None,
):
    """
    Build a fresh NetworkX DiGraph from the submission's available signals.

    Args:
        submission:        ProjectSubmission.
        code_index_result: Output of code_indexer.index_code_repo() (optional).
                           Provides imports + files for DEPENDS_ON edges.
        code_submission:   CodeSubmission with sonar_summary populated (optional).
                           Provides SonarQube findings for CONTRADICTS_CODE.
        report_chunks:     Report FAISS chunks (optional). Used to detect
                           security/quality CLAIMS to compare against findings,
                           and to extract technology mentions for briefs.

    Returns:
        NetworkX DiGraph with all detected edges. Always returns a graph
        (possibly empty) — never None.
    """
    import networkx as nx
    from viva_evaluator.services.knowledge_graph.contradiction_detector import (
        detect_contradictions,
    )
    from viva_evaluator.services.knowledge_graph.tech_extractor import (
        extract_technologies,
    )
    from viva_evaluator.services.knowledge_graph.brief_drafter import (
        draft_briefs_for_submission,
    )
    from viva_evaluator.services.knowledge_graph.brief_to_edges import add_brief_edges

    graph = nx.DiGraph()

    # -------------------------------------------------------------------------
    # 1. DEPENDS_ON edges from imports (Tier 1, AST-derived = verified)
    # -------------------------------------------------------------------------
    if code_index_result:
        _add_depends_on_edges(graph, code_index_result)

    # -------------------------------------------------------------------------
    # 2. CONTRADICTS_CODE edges (Tier 1, internal cross-reference)
    # -------------------------------------------------------------------------
    if code_submission and report_chunks:
        try:
            contradictions = detect_contradictions(code_submission, report_chunks)
            for c in contradictions:
                graph.add_edge(
                    c['code_finding'],
                    c['report_claim'],
                    edge_type='CONTRADICTS_CODE',
                    tier=1,
                    severity=c.get('severity', 'medium'),
                    finding_detail=c.get('finding_detail', ''),
                    claim_excerpt=c.get('claim_excerpt', ''),
                )
        except Exception as exc:
            logger.warning('Contradiction detection failed: %s', exc)

    # -------------------------------------------------------------------------
    # 3. Brief-driven edges (Week 4): ALTERNATIVE_TO, BETTER_FOR_SCALE
    #    Auto-drafts briefs for tech we haven't seen before, then materializes
    #    edges from any approved (T1) or pending (T2) briefs.
    # -------------------------------------------------------------------------
    try:
        imports_seen = (code_index_result or {}).get('imports_seen', set())
        technologies = extract_technologies(
            imports_seen=imports_seen,
            report_chunks=report_chunks or [],
        )

        if technologies:
            # Auto-draft any missing briefs (T2). Reuses existing T1/T2.
            draft_briefs_for_submission(submission, technologies)

            # Now wire whatever briefs exist into edges
            add_brief_edges(graph, technologies)
    except Exception as exc:
        logger.warning('Brief-driven edge generation failed: %s', exc)

    logger.info(
        'build_kg_for_submission: nodes=%d edges=%d (CONTRADICTS_CODE=%d)',
        graph.number_of_nodes(),
        graph.number_of_edges(),
        sum(1 for _, _, d in graph.edges(data=True)
            if d.get('edge_type') == 'CONTRADICTS_CODE'),
    )
    return graph


# =============================================================================
# Internals
# =============================================================================

def _add_depends_on_edges(graph, code_index_result: dict) -> None:
    """
    Each unique import becomes a target node. Source = file or module path.

    For the FYP scope we use a coarse model: one DEPENDS_ON edge per
    (file → imported_module) pair. Avoids exploding the graph with one
    edge per (function, import) pair which is rarely useful for viva
    questions.
    """
    files: Set[str] = code_index_result.get('files_touched', set())
    imports: Set[str] = code_index_result.get('imports_seen', set())

    if not files or not imports:
        return

    # Add nodes for every file and every imported module
    for f in files:
        graph.add_node(f, kind='code_module')
    for imp in imports:
        graph.add_node(imp, kind='external_dependency')

    # We don't know which file imported which module without re-walking
    # the AST results. Use the chunk-level data the indexer returns.
    chunks = code_index_result.get('chunks', [])
    seen_pairs: Set[tuple] = set()

    for chunk in chunks:
        file_path = chunk.get('section')
        chunk_imports = chunk.get('imports', [])
        if not file_path or not chunk_imports:
            continue
        for imp in chunk_imports:
            pair = (file_path, imp)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            graph.add_edge(
                file_path, imp,
                edge_type='DEPENDS_ON',
                tier=1,
            )
