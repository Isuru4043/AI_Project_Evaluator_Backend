import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from viva_evaluator.services.rag.cache import BoundedTTLCache
from viva_evaluator.services.rag.embeddings import (
    clear_query_embedding_cache,
    embed_text,
)
from viva_evaluator.services.rag.rerank import (
    invalidate_rerank_cache,
    rerank_chunks,
)
from viva_evaluator.services.rag.retrieval import (
    invalidate_kg_signal_cache,
    invalidate_module_retrieval_cache,
    retrieve_kg_signals,
    retrieve_module_materials,
)


class RAGCacheTests(TestCase):
    def tearDown(self):
        clear_query_embedding_cache()
        invalidate_module_retrieval_cache()
        invalidate_kg_signal_cache()
        invalidate_rerank_cache()

    def test_ttl_cache_is_bounded_and_expires(self):
        cache = BoundedTTLCache(max_entries=2, ttl_seconds=5)
        with patch(
            "viva_evaluator.services.rag.cache.time.monotonic",
            return_value=10,
        ):
            cache.set("a", 1)
            cache.set("b", 2)
            self.assertEqual(cache.get("a"), (True, 1))
            cache.set("c", 3)
            self.assertEqual(cache.get("b"), (False, None))

        with patch(
            "viva_evaluator.services.rag.cache.time.monotonic",
            return_value=16,
        ):
            self.assertEqual(cache.get("a"), (False, None))

    def test_query_embedding_is_cached_and_returned_as_a_copy(self):
        clear_query_embedding_cache()
        model = SimpleNamespace(
            encode=MagicMock(
                return_value=np.ones((1, 384), dtype=np.float32)
            )
        )
        with patch(
            "viva_evaluator.services.rag.embeddings._get_model",
            return_value=model,
        ):
            first = embed_text("stable architecture query")
            first[0] = 99
            second = embed_text("stable architecture query")

        model.encode.assert_called_once()
        self.assertEqual(second[0], 1.0)

    def test_module_results_are_cached_until_project_invalidation(self):
        invalidate_module_retrieval_cache()
        store = SimpleNamespace(
            num_chunks=1,
            search=MagicMock(
                return_value=[{"text": "module boundary", "chunk_idx": 1}]
            ),
        )
        with (
            patch(
                "viva_evaluator.services.rag.faiss_store.get_faiss_store",
                return_value=store,
            ),
            patch(
                "viva_evaluator.services.rag.retrieval.embed_text",
                return_value=np.ones(384, dtype=np.float32),
            ),
        ):
            first = retrieve_module_materials("project-1", "architecture", 2)
            first[0]["text"] = "mutated"
            second = retrieve_module_materials("project-1", "architecture", 2)
            invalidate_module_retrieval_cache("project-1")
            retrieve_module_materials("project-1", "architecture", 2)

        self.assertEqual(second[0]["text"], "module boundary")
        self.assertEqual(store.search.call_count, 2)

    def test_kg_summary_is_cached_until_submission_invalidation(self):
        invalidate_kg_signal_cache()
        graph = nx.DiGraph()
        graph.add_edge("service", "redis", edge_type="DEPENDS_ON", tier=1)
        submission = SimpleNamespace(id="submission-1")

        with (
            patch(
                "viva_evaluator.services.knowledge_graph.kg_store."
                "load_kg_for_submission",
                return_value=graph,
            ) as load_graph,
            patch(
                "viva_evaluator.services.knowledge_graph.kg_store."
                "retrieve_contradicts_code_edges",
                return_value=[],
            ) as contradictions,
        ):
            first = retrieve_kg_signals(submission)
            first["depends_on_topics"].append("mutated")
            second = retrieve_kg_signals(submission)
            invalidate_kg_signal_cache(submission.id)
            retrieve_kg_signals(submission)

        self.assertEqual(second["depends_on_topics"], ["redis"])
        self.assertEqual(load_graph.call_count, 2)
        self.assertEqual(contradictions.call_count, 2)

    def test_rerank_result_is_cached_and_namespace_invalidates_it(self):
        invalidate_rerank_cache()
        model = SimpleNamespace(
            predict=MagicMock(return_value=np.array([0.1, 0.9]))
        )
        chunks = [
            {"chunk_idx": 1, "text": "first"},
            {"chunk_idx": 2, "text": "second"},
        ]
        with (
            patch(
                "viva_evaluator.services.rag.rerank.reranker_enabled",
                return_value=True,
            ),
            patch(
                "viva_evaluator.services.rag.rerank._get_model",
                return_value=model,
            ),
        ):
            first = rerank_chunks(
                "query",
                chunks,
                1,
                cache_namespace="submission:1",
            )
            first[0]["text"] = "mutated"
            second = rerank_chunks(
                "query",
                chunks,
                1,
                cache_namespace="submission:1",
            )
            invalidate_rerank_cache("submission:1")
            rerank_chunks(
                "query",
                chunks,
                1,
                cache_namespace="submission:1",
            )

        self.assertEqual(second[0]["text"], "second")
        self.assertEqual(model.predict.call_count, 2)
