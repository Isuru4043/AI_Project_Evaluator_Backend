"""
Code indexing — turns a student's code repository into FAISS-ready chunks.

PIPELINE (per submission):
    1. Repo is already cloned/extracted by code_analysis.repo_service
    2. ast_parser walks code files → list of function/class units
    3. code_summarizer batches units → one LLM call per 10 → one-line summaries
    4. code_indexer wraps summaries into chunks tagged source='code'
    5. Chunks are merged with the report's FAISS index (single store per submission)

Each chunk has shape:
    {
        'text':         "File X. Function Y. Summary: Z. Code: ...",
        'source':       'code',
        'section':      '<file_path>',
        'function':     'verify_token',
        'imports':      ['jwt', 'datetime'],
        'chunk_idx':    47,
        ...
    }
"""

from viva_evaluator.services.code_indexing.code_indexer import index_code_repo

__all__ = ['index_code_repo']
