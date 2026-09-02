"""Isolated multi-provider model benchmark tooling.

Nothing in this package is used by the live viva pipeline.  Benchmark calls
must be made explicitly through the ``run_model_benchmark`` management
command, which is dry-run by default.
"""

from .contracts import BenchmarkCase, BenchmarkResult, ModelResponse, ModelSpec

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "ModelResponse",
    "ModelSpec",
]
