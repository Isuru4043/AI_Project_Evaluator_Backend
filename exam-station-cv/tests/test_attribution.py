"""Speaker decision + turn segmentation with scripted traces."""

from exam_cv.contracts.schemas import UNCERTAIN_SPEAKER
from exam_cv.speaker.attribution import (
    LipActivityTracker,
    SpeakerDecision,
    TurnSegmenter,
    WindowObservation,
    decide_speaker,
)

W = 800  # window_ms used throughout


class TestDecideSpeaker:
    def test_silence(self):
        d = decide_speaker(WindowObservation(0, voice_active=False, lip_scores={"s1": 0.9}))
        assert d.student_id is None

    def test_clear_speaker(self):
        d = decide_speaker(
            WindowObservation(0, True, {"s1": 0.7, "s2": 0.05, "s3": 0.02})
        )
        assert d.student_id == "s1"
        assert d.confidence > 0.5

    def test_voice_but_no_lips_is_uncertain(self):
        d = decide_speaker(WindowObservation(0, True, {"s1": 0.05, "s2": 0.03}))
        assert d.student_id == UNCERTAIN_SPEAKER

    def test_too_close_to_call_is_uncertain(self):
        d = decide_speaker(WindowObservation(0, True, {"s1": 0.50, "s2": 0.45}))
        assert d.student_id == UNCERTAIN_SPEAKER

    def test_no_faces_is_uncertain(self):
        d = decide_speaker(WindowObservation(0, True, {}))
        assert d.student_id == UNCERTAIN_SPEAKER


def run_trace(trace: list, **kw) -> list:
    """trace: list of (t_ms, student_id|None, confidence)"""
    seg = TurnSegmenter(window_ms=W, **kw)
    out = []
    for t, sid, conf in trace:
        out.extend(seg.push(SpeakerDecision(t, sid, conf)))
    out.extend(seg.flush())
    return out


class TestTurnSegmenter:
    def test_single_turn(self):
        trace = [(i * W, "s1", 0.9) for i in range(5)] + [(5 * W, None, 1.0)]
        turns = run_trace(trace)
        assert len(turns) == 1
        assert turns[0].student_id == "s1"
        assert turns[0].t_start_ms == 0
        assert turns[0].t_end_ms == 5 * W

    def test_speaker_change_closes_turn(self):
        trace = [(0, "s1", 0.9), (W, "s1", 0.9), (2 * W, "s2", 0.8), (3 * W, "s2", 0.8)]
        turns = run_trace(trace)
        assert [t.student_id for t in turns] == ["s1", "s2"]

    def test_flicker_dropped(self):
        # one lone 800ms window below min_turn_ms=1000 between two real turns
        trace = (
            [(i * W, "s1", 0.9) for i in range(3)]
            + [(3 * W, "s2", 0.6)]
            + [((4 + i) * W, "s1", 0.9) for i in range(3)]
        )
        turns = run_trace(trace, min_turn_ms=1000, merge_gap_ms=0)
        assert [t.student_id for t in turns] == ["s1", "s1"]

    def test_breath_pause_merged(self):
        # s1 speaks, 1-window silence gap (800ms > merge default 600 → use 900)
        trace = (
            [(i * W, "s1", 0.9) for i in range(3)]
            + [(3 * W, None, 1.0)]
            + [((4 + i) * W, "s1", 0.9) for i in range(3)]
        )
        turns = run_trace(trace, merge_gap_ms=900)
        assert len(turns) == 1
        assert turns[0].t_start_ms == 0
        assert turns[0].t_end_ms == 7 * W

    def test_uncertain_becomes_its_own_turn(self):
        trace = [(0, "s1", 0.9), (W, "s1", 0.9)] + [
            (2 * W, UNCERTAIN_SPEAKER, 0.1),
            (3 * W, UNCERTAIN_SPEAKER, 0.1),
        ]
        turns = run_trace(trace)
        assert [t.student_id for t in turns] == ["s1", UNCERTAIN_SPEAKER]


class TestLipActivityTracker:
    def test_static_mouth_scores_zero(self):
        lips = LipActivityTracker(window_ms=800)
        for t in range(0, 800, 66):
            lips.push("s1", t, 0.05)
        assert lips.scores().get("s1", 0.0) < 0.05

    def test_moving_mouth_scores_high(self):
        lips = LipActivityTracker(window_ms=800)
        for i, t in enumerate(range(0, 800, 66)):
            lips.push("s1", t, 0.05 if i % 2 else 0.35)
        assert lips.scores()["s1"] > 0.5

    def test_window_expiry(self):
        lips = LipActivityTracker(window_ms=800)
        for i, t in enumerate(range(0, 800, 66)):
            lips.push("s1", t, 0.05 if i % 2 else 0.35)
        for t in range(900, 1700, 66):  # go quiet; old samples expire
            lips.push("s1", t, 0.05)
        assert lips.scores()["s1"] < 0.1
