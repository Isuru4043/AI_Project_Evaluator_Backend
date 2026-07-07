"""Versioned contract schemas — the thin seam between this module and the platform.

Everything the platform hands in (SessionManifest) and everything this module
hands out (events, flags, summary, recording reference) is defined here.
The rest of the system does not exist yet; when it does, it adapts to these
shapes (schema_version lets both sides evolve).

INVARIANTS (see CLAUDE.md):
- Nothing here is a score. AttributionEvent routes answers to students;
  behavioral/integrity outputs are advisory annotations for the examiner.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

# Sentinel student_id used when the active speaker cannot be attributed with
# confidence. Never silently guess (HITL invariant).
UNCERTAIN_SPEAKER = "uncertain"


def ms_to_timecode(ms: int) -> str:
    """Milliseconds → 'HH:MM:SS' video timecode (floor to whole seconds)."""
    if ms < 0:
        raise ValueError(f"negative timestamp: {ms}")
    total_s = ms // 1000
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class SessionMode(str, Enum):
    INDIVIDUAL = "individual"
    GROUP = "group"


class RosterEntry(BaseModel):
    student_id: str
    display_name: str


class SessionManifest(BaseModel):
    """Seam 1: session clock/ID. Handed to us by the platform (or generated
    locally in standalone mode with the same shape). This module never mints
    its own session identity."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    mode: SessionMode
    roster: list[RosterEntry] = Field(min_length=1)
    t0_utc: datetime
    notes: Optional[str] = None

    def student_ids(self) -> set[str]:
        return {r.student_id for r in self.roster}


# ---------------------------------------------------------------------------
# Events (append-only JSONL during the session)
# ---------------------------------------------------------------------------


class AttributionEvent(BaseModel):
    """A speaking turn attributed to a roster student (or UNCERTAIN_SPEAKER).

    The ONLY output of this module that touches the scoring path — it routes
    an answer's score to the right student without changing its value.
    """

    type: Literal["attribution"] = "attribution"
    t_start_ms: int
    t_end_ms: int
    student_id: str  # roster id or UNCERTAIN_SPEAKER
    confidence: float = Field(ge=0.0, le=1.0)


class BehavioralKind(str, Enum):
    GAZE_SAMPLE = "gaze_sample"          # payload: {"on_camera": bool}
    OFF_SCREEN_GLANCE = "off_screen_glance"  # sustained look-away; payload: {"duration_ms": int}
    ABSENCE = "absence"                  # payload: {"duration_ms": int}


class BehavioralEvent(BaseModel):
    """Advisory-only behavioral observation. Never an input to any score."""

    type: Literal["behavioral"] = "behavioral"
    t_ms: int
    kind: BehavioralKind
    student_id: Optional[str] = None  # None when not attributable to a face
    payload: dict = Field(default_factory=dict)


class IntegrityKind(str, Enum):
    UNKNOWN_FACE = "unknown_face"
    EXTRA_PERSON = "extra_person"
    STUDENT_ABSENT = "student_absent"


class IntegrityFlag(BaseModel):
    """Timecoded evidence pointer for HUMAN review of the recording.

    Never auto-triggers a penalty; surfaced as a supplementary note so the
    lecturer can jump to `video_timecode` in the session recording.
    """

    type: Literal["integrity_flag"] = "integrity_flag"
    t_ms: int
    video_timecode: str  # HH:MM:SS into the session recording
    kind: IntegrityKind
    note: str
    student_id: Optional[str] = None
    snapshot_path: Optional[str] = None  # sparse evidence still, not video

    @classmethod
    def at(
        cls,
        t_ms: int,
        kind: IntegrityKind,
        note: str,
        video_offset_ms: int = 0,
        **kw,
    ) -> "IntegrityFlag":
        return cls(
            t_ms=t_ms,
            video_timecode=ms_to_timecode(max(t_ms - video_offset_ms, 0)),
            kind=kind,
            note=note,
            **kw,
        )


# ---------------------------------------------------------------------------
# End-of-session artifact (Seam 3: consumed by the report generator)
# ---------------------------------------------------------------------------


class RecordingRef(BaseModel):
    """Full-session A/V recording for post-hoc lecturer review, plus the
    mapping between session time and video time:

        video_ms = t_session_ms - video_offset_ms
    """

    path: str
    container: str = "mp4"
    video_offset_ms: int = 0  # session time at video frame 0
    duration_ms: Optional[int] = None
    drift_ms_at_end: Optional[int] = None  # measured A/V-vs-clock drift, QA only

    def timecode_for(self, t_session_ms: int) -> str:
        return ms_to_timecode(max(t_session_ms - self.video_offset_ms, 0))


class StudentSummary(BaseModel):
    student_id: str
    display_name: str
    speaking_time_ms: int = 0
    speaking_share: float = Field(default=0.0, ge=0.0, le=1.0)
    turn_count: int = 0
    attention_pct: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    integrity_flags: list[IntegrityFlag] = Field(default_factory=list)


class SessionSummary(BaseModel):
    """The single versioned artifact the report generator ingests verbatim
    as an advisory annex."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    mode: SessionMode
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    per_student: list[StudentSummary]
    timeline: list[AttributionEvent] = Field(default_factory=list)
    unattributed_speaking_ms: int = 0  # time attributed to UNCERTAIN_SPEAKER
    session_flags: list[IntegrityFlag] = Field(default_factory=list)  # not tied to one student
    recording: Optional[RecordingRef] = None
