"""End-of-session artifact builder (seam 3).

Replays the JSONL event log and folds it into one SessionSummary the report
generator ingests verbatim. Advisory only — the sole scoring-path output is
the attribution timeline (who spoke when).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..contracts.schemas import (
    UNCERTAIN_SPEAKER,
    AttributionEvent,
    BehavioralEvent,
    BehavioralKind,
    IntegrityFlag,
    RecordingRef,
    SessionManifest,
    SessionSummary,
    StudentSummary,
)
from ..events.store import Event, read_events


def build_summary(
    manifest: SessionManifest,
    events: Iterable[Event],
    recording: RecordingRef | None = None,
) -> SessionSummary:
    students = {
        r.student_id: StudentSummary(
            student_id=r.student_id, display_name=r.display_name
        )
        for r in manifest.roster
    }

    timeline: list[AttributionEvent] = []
    session_flags: list[IntegrityFlag] = []
    unattributed_ms = 0
    gaze_samples: dict[str, list[bool]] = {sid: [] for sid in students}

    for ev in events:
        if isinstance(ev, AttributionEvent):
            timeline.append(ev)
            dur = ev.t_end_ms - ev.t_start_ms
            if ev.student_id == UNCERTAIN_SPEAKER:
                unattributed_ms += dur
            elif ev.student_id in students:
                s = students[ev.student_id]
                s.speaking_time_ms += dur
                s.turn_count += 1
        elif isinstance(ev, IntegrityFlag):
            if ev.student_id and ev.student_id in students:
                students[ev.student_id].integrity_flags.append(ev)
            else:
                session_flags.append(ev)
        elif isinstance(ev, BehavioralEvent):
            if (
                ev.kind == BehavioralKind.GAZE_SAMPLE
                and ev.student_id in gaze_samples
            ):
                gaze_samples[ev.student_id].append(bool(ev.payload.get("on_camera")))

    total_attributed = sum(s.speaking_time_ms for s in students.values())
    denom = total_attributed + unattributed_ms
    for s in students.values():
        s.speaking_share = round(s.speaking_time_ms / denom, 4) if denom else 0.0
        samples = gaze_samples[s.student_id]
        if samples:
            s.attention_pct = round(100.0 * sum(samples) / len(samples), 1)

    timeline.sort(key=lambda e: e.t_start_ms)
    return SessionSummary(
        session_id=manifest.session_id,
        mode=manifest.mode,
        per_student=[students[r.student_id] for r in manifest.roster],
        timeline=timeline,
        unattributed_speaking_ms=unattributed_ms,
        session_flags=session_flags,
        recording=recording,
    )


def build_summary_from_log(
    manifest: SessionManifest,
    events_path: Path,
    recording: RecordingRef | None = None,
) -> SessionSummary:
    return build_summary(manifest, read_events(events_path), recording)
