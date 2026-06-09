"""
3D rubric radar chart — quick visual aggregation of mean Correctness,
Depth, Consistency per criterion.

OUTPUT:
    Base64-encoded PNG.
"""

import base64
import io
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def render_rubric_radar_png(per_criterion_means: List[Dict]) -> str:
    """
    Args:
        per_criterion_means: list of dicts in this shape:
            [
                {'name': '...', 'correctness': 0.7, 'depth': 0.6,
                 'consistency': 0.85},
                ...
            ]

    Returns:
        Base64 PNG (no prefix), or empty string on failure.
    """
    if not per_criterion_means:
        return ''

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning('matplotlib not installed — radar chart skipped.')
        return ''

    axes_labels = ['Correctness', 'Depth', 'Consistency']
    n_axes = len(axes_labels)
    angles = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=100)

    for entry in per_criterion_means:
        values = [
            float(entry.get('correctness', 0)),
            float(entry.get('depth', 0)),
            float(entry.get('consistency', 0)),
        ]
        values += values[:1]
        ax.plot(angles, values, marker='o', linewidth=2,
                label=entry.get('name', '?'))
        ax.fill(angles, values, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('3D Rubric — Per-Criterion Means', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.grid(True, alpha=0.4)

    buf = io.BytesIO()
    try:
        fig.savefig(buf, format='png', bbox_inches='tight')
    finally:
        plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')
