"""
Ablation runner — generates a question under specific flag conditions.

OUTPUT (per run):
    {
        'condition':        'full_system' | 'no_anchoring' | ...,
        'question_text':    '...',
        'tier1_passed':     bool,
        'tier1_failures':   [...],
        'critic_passed':    bool,
        'critic_scores':    {...},
        'attempts':         int,
        'latency_ms':       int,
        'flags_applied':    {...},
    }

run_ablation_set runs the same input through multiple conditions and
returns a list — that's the core data for the dissertation table.
"""

import logging
import time
from typing import List, Dict, Optional

from viva_evaluator.services.ablation.config import AblationFlags

logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================

def run_single_ablation(
    submission,
    criterion_name: str,
    criterion_description: str,
    flags: AblationFlags,
    last_answer: str = '',
    previous_question: Optional[str] = None,
    difficulty: str = 'medium',
) -> Dict:
    """
    Generate one question under the given flag set and return diagnostics.
    """
    from viva_evaluator.services.rag.retrieval import retrieve_hybrid_for_turn
    from viva_evaluator.services.agents import (
        generate_anchored_question, QuestionerInput,
    )

    t0 = time.time()

    # ---- Retrieval (full or downgraded) -----------------------------------
    retrieval = retrieve_hybrid_for_turn(
        submission=submission,
        criterion_name=criterion_name,
        criterion_description=criterion_description,
        last_answer=last_answer,
        top_k=3,
    )

    # Apply ablations
    if flags.disable_kg:
        retrieval = {
            **retrieval,
            'contradicts_code_alerts': [],
            'depends_on_topics':       [],
            'kg_available_for_topic':  False,
        }

    if flags.disable_section_aware:
        # Strip section labels — simulates Week 1 retrieval quality
        retrieval = {
            **retrieval,
            'chunks': [
                {**c, 'section': 'unknown'} for c in retrieval.get('chunks', [])
            ],
        }

    # ---- Question generation ---------------------------------------------
    qd = generate_anchored_question(
        QuestionerInput(
            criterion_name=criterion_name,
            criterion_description=criterion_description,
            retrieved_chunks=retrieval.get('chunks', []),
            kg_signals=retrieval,
            difficulty=difficulty,
            recent_questions=[],
            previous_question=previous_question,
            previous_answer=last_answer,
            is_first_question=(previous_question is None),
        ),
        max_retries=0 if flags.disable_tier1_validation else 1,
        enable_critic=not flags.disable_critic,
    )

    # ---- Anchoring "ablation" -------------------------------------------
    # The anchoring rule lives in the Questioner prompt and Tier 1 regex.
    # If the user wants this ablated, we report Tier 1 result for transparency
    # but treat the question as "always passing" so it propagates downstream.
    if flags.disable_anchoring:
        # Mark anchoring failures as "ignored" rather than failing the run
        if not qd.get('tier1_passed', True):
            qd['tier1_passed'] = True
            qd['tier1_failures'] = [f + ' (ignored: anchoring disabled)'
                                    for f in qd.get('tier1_failures', [])]

    latency_ms = int((time.time() - t0) * 1000)

    return {
        'condition':       flags.label(),
        'question_text':   qd.get('question_text', ''),
        'blooms_level':    qd.get('blooms_level', ''),
        'tier1_passed':    qd.get('tier1_passed', False),
        'tier1_failures':  qd.get('tier1_failures', []),
        'critic_ran':      qd.get('critic_ran', False),
        'critic_passed':   qd.get('critic_passed', True),
        'critic_critique': qd.get('critic_critique', ''),
        'critic_scores':   qd.get('critic_scores', {}),
        'attempts':        qd.get('attempts', 1),
        'latency_ms':      latency_ms,
        'flags_applied':   flags.__dict__.copy(),
    }


def run_ablation_set(
    submission,
    criterion_name: str,
    criterion_description: str,
    last_answer: str = '',
    previous_question: Optional[str] = None,
    difficulty: str = 'medium',
    conditions: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Run the same input through multiple ablation conditions.

    Args:
        conditions: optional list of dicts that map to AblationFlags. If
                    omitted, runs the recommended FYP set:
                      full_system, no_anchoring, no_kg
    """
    if conditions is None:
        conditions = [
            {},  # full system
            {'disable_anchoring': True},
            {'disable_kg': True},
        ]

    runs: List[Dict] = []
    for cfg in conditions:
        flags = AblationFlags.from_dict(cfg)
        try:
            run = run_single_ablation(
                submission=submission,
                criterion_name=criterion_name,
                criterion_description=criterion_description,
                flags=flags,
                last_answer=last_answer,
                previous_question=previous_question,
                difficulty=difficulty,
            )
        except Exception as exc:
            logger.exception('Ablation run failed for %s: %s', flags.label(), exc)
            run = {
                'condition':     flags.label(),
                'error':         str(exc),
                'flags_applied': flags.__dict__.copy(),
            }
        runs.append(run)

    return runs
