"""
Code indexer — single entry point for code → chunks.

Called by code_analysis after the repo is cloned/extracted. Hooks into the
same FAISS index as the report (one store per submission, mixed sources).

PIPELINE:
    repo_path
       ↓
    ast_parser.parse_repo  → units
       ↓
    code_summarizer.summarize_units  → units with 'summary' field
       ↓
    _units_to_chunks  → chunk dicts ready for vector_store
       ↓
    Returns chunks + the import graph (used later by KG builder)
"""

import logging
from typing import List, Dict

from viva_evaluator.services.code_indexing.ast_parser import parse_repo
from viva_evaluator.services.code_indexing.code_summarizer import summarize_units

logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================

def index_code_repo(
    repo_path: str,
    enable_summaries: bool = True,
) -> Dict:
    """
    Build code chunks for a repo.

    Args:
        repo_path:        Path to extracted source code.
        enable_summaries: If False, skip LLM summarization (uses fallback
                          summaries built from metadata). Useful for tests.

    Returns:
        {
            'chunks':       [chunk dicts ready for FAISS],
            'units_total':  int,
            'imports_seen': set of all imported modules across the repo,
            'files_touched': set of all source file paths,
        }
    """
    units = parse_repo(repo_path)
    if not units:
        return {
            'chunks': [],
            'units_total': 0,
            'imports_seen': set(),
            'files_touched': set(),
        }

    if enable_summaries:
        summarize_units(units)
    else:
        # Fast path: use cheap fallback summaries
        for unit in units:
            unit.setdefault(
                'summary',
                f"{unit['unit_type'].capitalize()} '{unit['name']}' in {unit['file_path']}.",
            )

    chunks = _units_to_chunks(units)

    imports_seen = set()
    files_touched = set()
    for unit in units:
        imports_seen.update(unit.get('imports', []))
        files_touched.add(unit['file_path'])

    logger.info(
        'index_code_repo: %d units → %d chunks across %d files (%d unique imports)',
        len(units), len(chunks), len(files_touched), len(imports_seen),
    )

    return {
        'chunks': chunks,
        'units_total': len(units),
        'imports_seen': imports_seen,
        'files_touched': files_touched,
    }


# =============================================================================
# Internals
# =============================================================================

def _units_to_chunks(units: List[Dict]) -> List[Dict]:
    """
    Convert parsed+summarized units into FAISS-ready chunk dicts.

    Each chunk's 'text' is the embedding input. We include the summary,
    metadata, and a code snippet so retrieval matches both natural-language
    queries ('how does authentication work?') and code-specific queries
    ('verify_token function').
    """
    chunks: List[Dict] = []
    for idx, unit in enumerate(units):
        snippet = (unit.get('source') or '').strip()[:600]

        text_blob = (
            f"File: {unit['file_path']}\n"
            f"{unit['unit_type'].capitalize()}: {unit['name']}\n"
            f"Summary: {unit.get('summary', '')}\n"
            f"Code:\n{snippet}"
        )

        chunks.append({
            'text':       text_blob,
            'source':     'code',
            'section':    unit['file_path'],
            'unit_type':  unit['unit_type'],
            'name':       unit['name'],
            'imports':    unit.get('imports', []),
            'line_start': unit.get('line_start'),
            'line_end':   unit.get('line_end'),
            'chunk_idx':  idx,    # local; report_indexer will renumber globally
            'char_start': 0,
            'char_end':   len(text_blob),
        })

    return chunks
