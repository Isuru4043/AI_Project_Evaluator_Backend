"""Behavioral analyzers — advisory only, run on the low-rate tick (2–5 FPS).

Pure state machines over per-tick face observations; the CV layer feeds
FaceTickObservation values, tests feed synthetic ones. None of this output
enters scoring (invariant 2); integrity flags are evidence pointers with
video timecodes for the lecturer to review in the recording.

Head-pose stability was dropped (duplicates gaze-on-task); solvePnP remains
a documented fallback only if iris gaze proves too noisy in calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..contracts.schemas import (
    BehavioralEvent,
    BehavioralKind,
    IntegrityFlag,
    IntegrityKind,
)


@dataclass
class FaceTickObservation:
    """What the face pipeline knows about one tick of the session.

    ``gaze_on_camera`` and ``visible_ids`` are deliberately separate. A student
    who turns far enough that the mesh loses their face is *not looking at the
    camera* — that has to keep their look-away spell running — but they are
    also not *present*, which is the presence analyzer's business. Folding the
    two together (presence == "has a gaze entry") meant the most blatant
    look-aways silently reset the gaze timer and were never flagged.
    """

    t_ms: int
    # student_id -> gaze on-camera (True/False). May include a student whose
    # face was not detected this tick, recorded as False.
    gaze_on_camera: dict[str, bool] = field(default_factory=dict)
    unknown_face_count: int = 0  # faces present but matching no roster identity
    # Students whose face was actually detected. None means "same as the gaze
    # keys", which is what every caller predating the split meant.
    visible_ids: Optional[set[str]] = None

    @property
    def present(self) -> set[str]:
        if self.visible_ids is not None:
            return set(self.visible_ids)
        return set(self.gaze_on_camera)


class GazeAnalyzer:
    """Per-student on/off-camera sampling + sustained off-screen glances.

    Two thresholds, because looking away is not by itself suspicious — a
    student thinking through an answer glances off constantly:

    * ``glance_threshold_ms`` (1.0s) marks an off-spell as a *glance* — an
      advisory BehavioralEvent, counted in the summary so the examiner sees
      "8 look-aways" without being shown eight of anything.
    * ``flag_threshold_ms`` (2.0s) is the point an off-spell becomes long enough
      to be worth watching, and earns one timecoded IntegrityFlag.

    Both are edge-triggered per off-spell, so a 40-second stare produces one
    glance and one flag, not hundreds.
    """

    def __init__(
        self,
        glance_threshold_ms: int = 1000,
        flag_threshold_ms: int = 2000,
        video_offset_ms: int = 0,
    ):
        self.glance_threshold_ms = glance_threshold_ms
        self.flag_threshold_ms = flag_threshold_ms
        self.video_offset_ms = video_offset_ms
        self._off_since: dict[str, int] = {}  # student_id -> t_ms gaze went off
        self._glance_open: set[str] = set()   # glance already emitted for this off-spell
        self._flag_open: set[str] = set()     # flag already emitted for this off-spell

    def push(self, obs: FaceTickObservation) -> list[BehavioralEvent | IntegrityFlag]:
        events: list[BehavioralEvent | IntegrityFlag] = []
        present = obs.present
        for sid, on_camera in obs.gaze_on_camera.items():
            events.append(
                BehavioralEvent(
                    t_ms=obs.t_ms,
                    kind=BehavioralKind.GAZE_SAMPLE,
                    student_id=sid,
                    payload={"on_camera": bool(on_camera)},
                )
            )
            if on_camera:
                self._off_since.pop(sid, None)
                self._glance_open.discard(sid)
                self._flag_open.discard(sid)
                continue

            start = self._off_since.setdefault(sid, obs.t_ms)
            off_for = obs.t_ms - start
            if off_for >= self.glance_threshold_ms and sid not in self._glance_open:
                self._glance_open.add(sid)
                events.append(
                    BehavioralEvent(
                        t_ms=start,
                        kind=BehavioralKind.OFF_SCREEN_GLANCE,
                        student_id=sid,
                        payload={"duration_ms": off_for},
                    )
                )
            # While the face is not visible the spell keeps running — the time
            # still counts against attention — but the marker belongs to the
            # presence analyzer, which fires sooner and says the truer thing.
            # Emitting both would put two dots on one incident.
            if (
                off_for >= self.flag_threshold_ms
                and sid not in self._flag_open
                and sid in present
            ):
                self._flag_open.add(sid)
                events.append(
                    IntegrityFlag.at(
                        t_ms=start,
                        kind=IntegrityKind.GAZE_OFF_SCREEN,
                        note=(
                            f"Looked away from the screen for "
                            f"{self.flag_threshold_ms / 1000:g}s+ — review recording"
                        ),
                        video_offset_ms=self.video_offset_ms,
                        student_id=sid,
                    )
                )
        return events


class PresenceAnalyzer:
    """Absence intervals per student + unknown/extra-person integrity flags.

    Presence is read from ``obs.present`` (the faces actually detected), never
    from the gaze map, which may carry a False entry for a student precisely
    because they are not visible.

    ``absence_threshold_ms`` (1.5s) is short on purpose: turning away from the
    camera for a second and a half during an oral exam is worth a timecoded
    pointer. Flags are edge-triggered (one per incident, at incident start) so
    a 5-minute intrusion produces one timecoded note, not 600 events.
    """

    def __init__(
        self,
        roster_ids: list[str],
        absence_threshold_ms: int = 1500,
        video_offset_ms: int = 0,
    ):
        self.roster_ids = list(roster_ids)
        self.absence_threshold_ms = absence_threshold_ms
        self.video_offset_ms = video_offset_ms
        self._missing_since: dict[str, int] = {}
        self._absence_flagged: set[str] = set()
        self._unknown_present = False

    def push(self, obs: FaceTickObservation) -> list[BehavioralEvent | IntegrityFlag]:
        out: list[BehavioralEvent | IntegrityFlag] = []
        present = obs.present

        for sid in self.roster_ids:
            if sid in present:
                start = self._missing_since.pop(sid, None)
                if start is not None and sid in self._absence_flagged:
                    self._absence_flagged.discard(sid)
                    out.append(
                        BehavioralEvent(
                            t_ms=start,
                            kind=BehavioralKind.ABSENCE,
                            student_id=sid,
                            payload={"duration_ms": obs.t_ms - start},
                        )
                    )
            else:
                start = self._missing_since.setdefault(sid, obs.t_ms)
                gone_for = obs.t_ms - start
                if (
                    gone_for >= self.absence_threshold_ms
                    and sid not in self._absence_flagged
                ):
                    self._absence_flagged.add(sid)
                    out.append(
                        IntegrityFlag.at(
                            t_ms=start,
                            kind=IntegrityKind.STUDENT_ABSENT,
                            note=(
                                f"Student {sid} turned away from or left the "
                                f"camera for "
                                f"{self.absence_threshold_ms / 1000:g}s+ — "
                                f"review recording"
                            ),
                            video_offset_ms=self.video_offset_ms,
                            student_id=sid,
                        )
                    )

        if obs.unknown_face_count > 0 and not self._unknown_present:
            self._unknown_present = True
            kind = (
                IntegrityKind.EXTRA_PERSON
                if len(present) >= len(self.roster_ids)
                else IntegrityKind.UNKNOWN_FACE
            )
            out.append(
                IntegrityFlag.at(
                    t_ms=obs.t_ms,
                    kind=kind,
                    note=(
                        "Unrecognized person in frame "
                        f"({obs.unknown_face_count}) — review recording"
                    ),
                    video_offset_ms=self.video_offset_ms,
                )
            )
        elif obs.unknown_face_count == 0:
            self._unknown_present = False

        return out
