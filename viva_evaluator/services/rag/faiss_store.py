"""
FAISS Store — Loads module-material FAISS indices from Django storage.

The module_indexer saves per-material FAISS blobs + chunk JSON into
``default_storage`` under paths like:
    module_materials/{material_id}_faiss.bin
    module_materials/{material_id}_chunks.json

This module loads those blobs at query time, caches them in-process,
and wraps them in a SubmissionVectorStore so the retrieval layer can
call .search() exactly as it does for submission indices.
"""

import json
import logging
from typing import Optional

import numpy as np

from viva_evaluator.services.rag.vector_store import SubmissionVectorStore

logger = logging.getLogger(__name__)

# In-process cache: index_name -> SubmissionVectorStore
_MODULE_CACHE: dict = {}


def get_faiss_store(index_name: str) -> Optional[SubmissionVectorStore]:
    """
    Load the FAISS store for a module-materials index.

    ``index_name`` is expected to be ``module_materials_{project_id}``.
    We look up every ModuleMaterial for that project whose processing is
    complete, load their individual FAISS blobs from default_storage,
    merge them into a single combined index, and cache the result.

    Returns None if no indexed materials exist.
    """
    cached = _MODULE_CACHE.get(index_name)
    if cached is not None:
        return cached

    try:
        import faiss
        from django.core.files.storage import default_storage
        from core.models import ModuleMaterial
        from viva_evaluator.services.rag.embeddings import EMBEDDING_DIM

        # index_name format: "module_materials_{project_id}"
        prefix = "module_materials_"
        if not index_name.startswith(prefix):
            logger.warning("get_faiss_store: unexpected index_name=%s", index_name)
            return None

        project_id = index_name[len(prefix):]

        materials = ModuleMaterial.objects.filter(
            project_id=project_id,
            processing_status=ModuleMaterial.ProcessingStatus.COMPLETED,
        )

        if not materials.exists():
            return None

        all_chunks = []
        combined_index = faiss.IndexFlatIP(EMBEDDING_DIM)

        for material in materials:
            chunks_path = f"module_materials/{material.id}_chunks.json"
            faiss_path = f"module_materials/{material.id}_faiss.bin"

            if not default_storage.exists(faiss_path) or not default_storage.exists(chunks_path):
                logger.warning(
                    "get_faiss_store: missing files for material=%s", material.id
                )
                continue

            # Load chunks
            with default_storage.open(chunks_path, 'rb') as f:
                chunks = json.loads(f.read().decode('utf-8'))

            # Load FAISS index
            with default_storage.open(faiss_path, 'rb') as f:
                blob = f.read()

            arr = np.frombuffer(blob, dtype=np.uint8)
            mat_index = faiss.deserialize_index(arr)

            # Re-number chunk indices before merging
            base_idx = len(all_chunks)
            for i, chunk in enumerate(chunks):
                chunk['chunk_idx'] = base_idx + i

            # Merge vectors into the combined index
            if mat_index.ntotal > 0:
                vectors = faiss.rev_swig_ptr(
                    mat_index.get_xb(), mat_index.ntotal * EMBEDDING_DIM
                )
                vectors = np.reshape(vectors, (mat_index.ntotal, EMBEDDING_DIM)).copy()
                combined_index.add(vectors)

            all_chunks.extend(chunks)

        if not all_chunks:
            return None

        store = SubmissionVectorStore(combined_index, all_chunks)
        _MODULE_CACHE[index_name] = store
        logger.info(
            "get_faiss_store: loaded %s — %d chunks from %d materials",
            index_name, len(all_chunks), materials.count(),
        )
        return store

    except Exception as e:
        logger.error("get_faiss_store: failed for %s — %s", index_name, e)
        return None


def invalidate_module_cache(project_id: str) -> None:
    """Drop cached store for a project (call after re-indexing materials)."""
    key = f"module_materials_{project_id}"
    _MODULE_CACHE.pop(key, None)
