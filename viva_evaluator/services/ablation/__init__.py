"""
Ablation experiment harness — runs the question-generation pipeline under
different configurations and emits comparable outputs for the dissertation
evaluation chapter.

DESIGN:
    AblationFlags toggles individual contributions:
        - disable_anchoring         : drops the anchoring rule from prompt
        - disable_tier1_validation  : skips programmatic Tier 1 checks
        - disable_critic            : skips Tier 2 LLM critique
        - disable_kg                : zeros out KG signals (FAISS-only)
        - disable_section_aware     : ignores section labels in retrieval

    runner.run_ablation_set takes one (criterion, last_answer, submission)
    pair and produces a per-condition table of question outputs + metrics.

    Two conditions per the v3 spec recommendation:
        Condition A: full system
        Condition B: pick one to disable (most commonly: disable_anchoring)
"""

from viva_evaluator.services.ablation.config import AblationFlags
from viva_evaluator.services.ablation.runner import (
    run_single_ablation,
    run_ablation_set,
)

__all__ = [
    'AblationFlags',
    'run_single_ablation',
    'run_ablation_set',
]
