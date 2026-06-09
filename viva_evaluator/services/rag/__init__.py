"""
RAG (Retrieval-Augmented Generation) layer.

This package handles:
    - Chunking: splitting submission text into retrievable pieces
    - Embeddings: turning chunks into vectors via SBERT
    - Vector store: persisting FAISS indices in PostgreSQL
    - Retrieval: semantic search over a submission's content

Used by the Questioner and Analyzer agents to ground every output
in the student's actual report and code.
"""

from viva_evaluator.services.rag.retrieval import retrieve_for_turn, retrieve_for_indexing
from viva_evaluator.services.rag.vector_store import (
    SubmissionVectorStore,
    save_index_for_submission,
    load_index_for_submission,
)

__all__ = [
    'retrieve_for_turn',
    'retrieve_for_indexing',
    'SubmissionVectorStore',
    'save_index_for_submission',
    'load_index_for_submission',
]
