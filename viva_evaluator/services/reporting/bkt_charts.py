"""
BKT trajectory charts — matplotlib line plots returned as base64 PNGs.

DESIGN:
    Headless rendering (matplotlib backend='Agg') so this works on Azure
    App Service / Linux without a display. Each criterion gets one line in
    a single multi-line plot. Threshold reference lines at 0.35 / 0.55 /
    0.75 mark Bloom band boundaries from the v3 spec.
"""

import base64
import io
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def render_bkt_trajectory_png(
    bkt_states_by_criterion: Dict[str, List[float]],
    criterion_name_map: Dict[str, str],
) -> str:
    """
    Render a multi-line plot of P(L_t) trajectories.

    Args:
        bkt_states_by_criterion: {criterion_id: [p_lt history]}
        criterion_name_map:      {criterion_id: human-readable name}

    Returns:
        Base64-encoded PNG (no `data:image/png;base64,` prefix), or empty
        string if matplotlib isn't available or no data is present.
    """
    if not bkt_states_by_criterion:
        return ''

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning('matplotlib not installed — BKT chart skipped.')
        return ''

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    max_turns = 0
    for crit_id, history in bkt_states_by_criterion.items():
        if not history:
            continue
        # Prepend the initial P(L_0) = 0.30 so trajectories show the start point
        series = [0.30] + list(history)
        x = list(range(len(series)))
        max_turns = max(max_turns, len(series))
        label = criterion_name_map.get(crit_id, crit_id)
        ax.plot(x, series, marker='o', linewidth=2, label=label)

    # Bloom band boundaries
    for y, label in [(0.35, 'Apply'), (0.55, 'Analyze'), (0.75, 'Evaluate')]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.7, alpha=0.6)
        ax.text(
            max_turns - 0.5 if max_turns else 0.5, y + 0.01, f'≥ {label}',
            fontsize=8, color='gray', ha='right',
        )

    ax.set_xlabel('Turn (per concept)')
    ax.set_ylabel('P(L_t) — mastery probability')
    ax.set_title('BKT Trajectory by Criterion')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(max_turns - 1, 1))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight')
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')
