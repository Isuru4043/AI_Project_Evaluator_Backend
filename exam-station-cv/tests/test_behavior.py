"""Behavioral analyzers with synthetic tick observations."""

from exam_cv.behavior.analyzers import (
    FaceTickObservation,
    GazeAnalyzer,
    PresenceAnalyzer,
)
from exam_cv.contracts.schemas import (
    BehavioralEvent,
    BehavioralKind,
    IntegrityFlag,
    IntegrityKind,
)

TICK = 333


def ticks(specs):
    """specs: list of (t_ms, gaze_map, unknown_count)"""
    return [
        FaceTickObservation(t_ms=t, gaze_on_camera=g, unknown_face_count=u)
        for t, g, u in specs
    ]


class TestGazeAnalyzer:
    def test_samples_emitted(self):
        g = GazeAnalyzer()
        events = g.push(FaceTickObservation(0, {"s1": True, "s2": False}))
        kinds = [(e.student_id, e.payload["on_camera"]) for e in events]
        assert ("s1", True) in kinds and ("s2", False) in kinds

    def test_sustained_glance_flagged_once(self):
        g = GazeAnalyzer(glance_threshold_ms=1000)
        glances = []
        for t in range(0, 3000, TICK):
            for e in g.push(FaceTickObservation(t, {"s1": False})):
                if e.kind == BehavioralKind.OFF_SCREEN_GLANCE:
                    glances.append(e)
        assert len(glances) == 1
        assert glances[0].t_ms == 0  # anchored at glance start

    def test_short_glance_not_flagged(self):
        g = GazeAnalyzer(glance_threshold_ms=3000)
        events = []
        for t, on in [(0, False), (TICK, False), (2 * TICK, True)]:
            events += g.push(FaceTickObservation(t, {"s1": on}))
        assert not [
            e
            for e in events
            if isinstance(e, BehavioralEvent)
            and e.kind == BehavioralKind.OFF_SCREEN_GLANCE
        ]

    def test_sustained_look_away_raises_one_integrity_flag(self):
        """The cheating-relevant signal: a long stare off-screen becomes ONE
        timecoded pointer, anchored at the moment the student looked away."""
        g = GazeAnalyzer(glance_threshold_ms=1000, flag_threshold_ms=3000)
        out = []
        for t in range(60_000, 90_000, TICK):
            out += g.push(FaceTickObservation(t, {"s1": False}))

        flags = [e for e in out if isinstance(e, IntegrityFlag)]
        assert len(flags) == 1
        assert flags[0].kind == IntegrityKind.GAZE_OFF_SCREEN
        assert flags[0].student_id == "s1"
        assert flags[0].t_ms == 60_000          # anchored at look-away start
        assert flags[0].video_timecode == "00:01:00"

    def test_thinking_glance_stays_below_the_flag_threshold(self):
        """A 4s glance is a student thinking, not evidence — it counts as a
        glance but must not raise an integrity flag."""
        g = GazeAnalyzer(glance_threshold_ms=3000, flag_threshold_ms=6000)
        out = []
        for t in range(0, 4000, TICK):
            out += g.push(FaceTickObservation(t, {"s1": False}))
        out += g.push(FaceTickObservation(4000, {"s1": True}))

        assert not [e for e in out if isinstance(e, IntegrityFlag)]
        glances = [
            e
            for e in out
            if isinstance(e, BehavioralEvent)
            and e.kind == BehavioralKind.OFF_SCREEN_GLANCE
        ]
        assert len(glances) == 1

    def test_flag_refires_on_a_separate_look_away(self):
        g = GazeAnalyzer(glance_threshold_ms=1000, flag_threshold_ms=2000)
        out = []
        for t in range(0, 3000, TICK):          # spell 1
            out += g.push(FaceTickObservation(t, {"s1": False}))
        out += g.push(FaceTickObservation(3000, {"s1": True}))   # back on screen
        for t in range(4000, 7000, TICK):       # spell 2
            out += g.push(FaceTickObservation(t, {"s1": False}))

        flags = [e for e in out if isinstance(e, IntegrityFlag)]
        assert len(flags) == 2
        assert [f.t_ms for f in flags] == [0, 4000]


class TestPresenceAnalyzer:
    def test_absence_flag_then_return_event(self):
        p = PresenceAnalyzer(["s1"], absence_threshold_ms=1000)
        out = []
        for t in range(0, 2000, TICK):
            out += p.push(FaceTickObservation(t, {}))  # s1 gone
        flags = [e for e in out if isinstance(e, IntegrityFlag)]
        assert len(flags) == 1
        assert flags[0].kind == IntegrityKind.STUDENT_ABSENT
        assert flags[0].student_id == "s1"
        assert flags[0].video_timecode == "00:00:00"

        back = p.push(FaceTickObservation(2000, {"s1": True}))
        absences = [
            e
            for e in back
            if isinstance(e, BehavioralEvent) and e.kind == BehavioralKind.ABSENCE
        ]
        assert len(absences) == 1
        assert absences[0].payload["duration_ms"] == 2000

    def test_unknown_face_edge_triggered(self):
        p = PresenceAnalyzer(["s1"])
        out = []
        for t in range(0, 5 * TICK, TICK):
            out += p.push(FaceTickObservation(t, {"s1": True}, unknown_face_count=1))
        flags = [e for e in out if isinstance(e, IntegrityFlag)]
        assert len(flags) == 1  # one incident, one note
        assert flags[0].kind == IntegrityKind.EXTRA_PERSON

    def test_unknown_face_reflagged_after_clearing(self):
        p = PresenceAnalyzer(["s1"])
        out = []
        out += p.push(FaceTickObservation(0, {"s1": True}, unknown_face_count=1))
        out += p.push(FaceTickObservation(TICK, {"s1": True}, unknown_face_count=0))
        out += p.push(FaceTickObservation(2 * TICK, {"s1": True}, unknown_face_count=1))
        flags = [e for e in out if isinstance(e, IntegrityFlag)]
        assert len(flags) == 2

    def test_video_timecode_respects_offset(self):
        p = PresenceAnalyzer(["s1"], absence_threshold_ms=0, video_offset_ms=2000)
        out = p.push(FaceTickObservation(62_000, {}))
        flags = [e for e in out if isinstance(e, IntegrityFlag)]
        assert flags[0].video_timecode == "00:01:00"
