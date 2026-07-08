"""Active-speaker attribution: lip motion × voice activity → speaking turns.

Pure logic lives here (unit-testable with scripted traces). The CV side only
has to feed WindowObservation values; if lip×VAD ever proves insufficient in
group tests, a learned model (Light-ASD / TalkNet) replaces the decision
function behind the same interface — the TurnSegmenter is unchanged.

HITL invariant: low-confidence windows resolve to UNCERTAIN_SPEAKER; we never
silently guess who spoke.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..contracts.schemas import UNCERTAIN_SPEAKER, AttributionEvent


class LipActivityTracker:
    """Per-student mouth-aspect-ratio history → lip-motion score per window.

    Score = std-dev of MAR over the window, scaled so typical speech lands
    around 0.3–0.8. Silence/closed mouth ≈ 0.
    """

    def __init__(self, window_ms: int = 800, scale: float = 25.0):
        self.window_ms = window_ms
        self.scale = scale
        self._history: dict[str, list[tuple[int, float]]] = {}

    def push(self, student_id: str, t_ms: int, mar: float) -> None:
        hist = self._history.setdefault(student_id, [])
        hist.append((t_ms, mar))
        cutoff = t_ms - self.window_ms
        while hist and hist[0][0] < cutoff:
            hist.pop(0)

    def scores(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for sid, hist in self._history.items():
            if len(hist) >= 3:
                mars = np.array([m for _, m in hist])
                out[sid] = min(1.0, float(mars.std()) * self.scale)
        return out


@dataclass
class WindowObservation:
    """One sliding-window sample of the scene.

    lip_scores: per-student lip-motion activity (mouth-aspect-ratio variance,
    normalized 0..1) for faces with a resolved identity this window.
    """

    t_ms: int
    voice_active: bool
    lip_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SpeakerDecision:
    t_ms: int
    student_id: Optional[str]  # None = silence; UNCERTAIN_SPEAKER = speech, unknown who
    confidence: float


def decide_speaker(
    obs: WindowObservation,
    min_lip_score: float = 0.15,
    min_margin: float = 0.10,
) -> SpeakerDecision:
    """Pick the active speaker for one window.

    - no voice activity → silence (None)
    - voice active, one face clearly moving its mouth → that student
    - voice active but no face passes the bar, or two faces are too close to
      call → UNCERTAIN_SPEAKER (never guess)
    """
    if not obs.voice_active:
        return SpeakerDecision(obs.t_ms, None, 1.0)

    ranked = sorted(obs.lip_scores.items(), key=lambda kv: kv[1], reverse=True)
    if not ranked or ranked[0][1] < min_lip_score:
        return SpeakerDecision(obs.t_ms, UNCERTAIN_SPEAKER, 0.0)

    best_id, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if best - runner_up < min_margin:
        return SpeakerDecision(obs.t_ms, UNCERTAIN_SPEAKER, best - runner_up)

    # Confidence: how decisively this face wins, capped by absolute activity.
    confidence = min(1.0, best) * min(1.0, (best - runner_up) / max(best, 1e-6) + 0.5)
    return SpeakerDecision(obs.t_ms, best_id, round(min(confidence, 1.0), 3))


class TurnSegmenter:
    """Fold per-window decisions into speaking turns.

    - a turn = consecutive windows with the same speaker
    - gaps shorter than merge_gap_ms between same-speaker turns are merged
      (breath pauses)
    - turns shorter than min_turn_ms are dropped (flickers)
    """

    def __init__(
        self,
        window_ms: int,
        min_turn_ms: int = 700,
        merge_gap_ms: int = 600,
    ):
        self.window_ms = window_ms
        self.min_turn_ms = min_turn_ms
        self.merge_gap_ms = merge_gap_ms
        self._open: Optional[dict] = None  # current turn being built
        self._pending: list[AttributionEvent] = []
        self._last_closed: Optional[AttributionEvent] = None

    def push(self, d: SpeakerDecision) -> list[AttributionEvent]:
        """Feed one decision; returns any turns finalized by this push."""
        out: list[AttributionEvent] = []

        if self._open is not None and d.student_id != self._open["student_id"]:
            out.extend(self._close(self._open))
            self._open = None

        if d.student_id is not None:
            if self._open is None:
                self._open = {
                    "student_id": d.student_id,
                    "t_start_ms": d.t_ms,
                    "t_end_ms": d.t_ms + self.window_ms,
                    "confidences": [d.confidence],
                }
            else:
                self._open["t_end_ms"] = d.t_ms + self.window_ms
                self._open["confidences"].append(d.confidence)

        return out

    def flush(self) -> list[AttributionEvent]:
        """End of session: close any open turn and emit what remains."""
        out: list[AttributionEvent] = []
        if self._open is not None:
            out.extend(self._close(self._open))
            self._open = None
        if self._last_closed is not None:
            last = self._last_closed
            if last.t_end_ms - last.t_start_ms >= self.min_turn_ms:
                out.append(last)
            self._last_closed = None
        return out

    def _close(self, turn: dict) -> list[AttributionEvent]:
        event = AttributionEvent(
            t_start_ms=turn["t_start_ms"],
            t_end_ms=turn["t_end_ms"],
            student_id=turn["student_id"],
            confidence=round(
                sum(turn["confidences"]) / len(turn["confidences"]), 3
            ),
        )

        # Merge with the previous closed turn if same speaker and small gap.
        prev = self._last_closed
        if (
            prev is not None
            and prev.student_id == event.student_id
            and event.t_start_ms - prev.t_end_ms <= self.merge_gap_ms
        ):
            merged = AttributionEvent(
                t_start_ms=prev.t_start_ms,
                t_end_ms=event.t_end_ms,
                student_id=prev.student_id,
                confidence=round((prev.confidence + event.confidence) / 2, 3),
            )
            self._last_closed = merged
            return []

        emitted: list[AttributionEvent] = []
        if prev is not None:
            if prev.t_end_ms - prev.t_start_ms >= self.min_turn_ms:
                emitted.append(prev)
            # else: drop flicker
        self._last_closed = event
        return emitted
