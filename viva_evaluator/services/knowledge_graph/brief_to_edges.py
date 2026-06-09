"""
Brief → Edges converter — turns ApprovedDomainBrief rows into KG edges.

Called by kg_builder when assembling the per-submission graph. Reads briefs
of TIER 1 (approved) and TIER 2 (drafted but unreviewed) from the DB and
inserts edges into the NetworkX graph.

EDGE MAPPING:
    For each technology with a brief:
        For each alternative in brief.alternatives:
            If trigger contains 'scale' / 'concurrency' / 'load' / 'high traffic':
                edge: BETTER_FOR_SCALE  (alternative → tech)
            Else:
                edge: ALTERNATIVE_TO     (alternative ↔ tech, both directions)

    Each edge is tagged with the brief's tier (1 = approved, 2 = unreviewed).
"""

import logging
import re
from typing import Set

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

_SCALE_TRIGGERS = re.compile(
    r'\b(?:scale|concurrenc|high\s+(?:load|traffic|throughput|volume)|distributed|cluster)',
    re.IGNORECASE,
)


# =============================================================================
# Public API
# =============================================================================

def add_brief_edges(graph, technologies):
    """
    Mutate `graph` by adding edges derived from briefs about each technology.

    Args:
        graph:        NetworkX DiGraph to extend.
        technologies: list of tech dicts from extract_technologies().

    Returns:
        Tuple (n_edges_added, n_briefs_used) for logging.
    """
    from viva_evaluator.models import ApprovedDomainBrief

    n_edges = 0
    n_briefs = 0

    for tech in technologies:
        tech_name = tech['name']

        # Prefer T1 (approved); fall back to T2 (pending draft)
        brief = ApprovedDomainBrief.objects.filter(
            technology__iexact=tech_name,
            status__in=[
                ApprovedDomainBrief.Status.ACTIVE,
                ApprovedDomainBrief.Status.PENDING,
            ],
        ).order_by('tier', '-drafted_at').first()  # ascending tier → T1 first

        if not brief:
            continue

        n_briefs += 1
        n_edges += _add_alternatives_as_edges(graph, tech_name, brief)

    logger.info(
        'add_brief_edges: %d edges added from %d briefs',
        n_edges, n_briefs,
    )
    return (n_edges, n_briefs)


# =============================================================================
# Internals
# =============================================================================

def _add_alternatives_as_edges(graph, tech_name, brief) -> int:
    """Insert ALTERNATIVE_TO / BETTER_FOR_SCALE edges for one tech."""
    alternatives = (brief.brief_json or {}).get('alternatives', [])
    if not alternatives:
        return 0

    # Make sure both endpoints are nodes
    graph.add_node(tech_name, kind='technology')

    n_added = 0
    seen: Set[tuple] = set()

    for alt in alternatives:
        alt_name = (alt.get('name') or '').strip()
        if not alt_name or alt_name.lower() == tech_name.lower():
            continue

        graph.add_node(alt_name, kind='technology')

        trigger = (alt.get('trigger') or '').strip()
        rationale = (alt.get('rationale') or '').strip()

        is_scale = bool(_SCALE_TRIGGERS.search(trigger))

        edge_attrs = {
            'tier':      brief.tier,
            'trigger':   trigger,
            'rationale': rationale,
            'brief_id':  str(brief.id),
        }

        if is_scale:
            # Directional: alternative → tech (alternative is preferred for scale)
            key = ('BETTER_FOR_SCALE', alt_name, tech_name)
            if key not in seen:
                graph.add_edge(
                    alt_name, tech_name,
                    edge_type='BETTER_FOR_SCALE',
                    **edge_attrs,
                )
                seen.add(key)
                n_added += 1
        else:
            # Bidirectional: ALTERNATIVE_TO
            for src, dst in [(alt_name, tech_name), (tech_name, alt_name)]:
                key = ('ALTERNATIVE_TO', src, dst)
                if key in seen:
                    continue
                graph.add_edge(
                    src, dst,
                    edge_type='ALTERNATIVE_TO',
                    **edge_attrs,
                )
                seen.add(key)
                n_added += 1

    return n_added
