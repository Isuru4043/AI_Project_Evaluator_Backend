"""JSONL store roundtrip + summary builder folding."""

from exam_cv.contracts.schemas import (
    UNCERTAIN_SPEAKER,
    AttributionEvent,
    BehavioralEvent,
    BehavioralKind,
    IntegrityFlag,
    IntegrityKind,
    RecordingRef,
)
from exam_cv.events.store import append_event, read_events
from exam_cv.report.summary import build_summary


class TestStore:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "events.jsonl"
        events = [
            AttributionEvent(t_start_ms=0, t_end_ms=4000, student_id="s1", confidence=0.9),
            BehavioralEvent(t_ms=100, kind=BehavioralKind.GAZE_SAMPLE, student_id="s1",
                            payload={"on_camera": True}),
            IntegrityFlag(t_ms=5000, video_timecode="00:00:05",
                          kind=IntegrityKind.UNKNOWN_FACE, note="check"),
        ]
        for e in events:
            append_event(path, e)
        back = list(read_events(path))
        assert [type(e).__name__ for e in back] == [
            "AttributionEvent", "BehavioralEvent", "IntegrityFlag",
        ]
        assert back[0].student_id == "s1"


class TestEmptySession:
    """A session that produced no events (nobody in frame, no speech) must
    still finalize into a valid summary — the JSONL file may not exist."""

    def test_read_missing_file_is_empty(self, tmp_path):
        assert list(read_events(tmp_path / "does_not_exist.jsonl")) == []

    def test_summary_from_missing_log(self, individual_manifest, tmp_path):
        from exam_cv.report.summary import build_summary_from_log

        summary = build_summary_from_log(
            individual_manifest, tmp_path / "no_events.jsonl"
        )
        assert summary.per_student[0].speaking_time_ms == 0
        assert summary.timeline == []


class TestSummary:
    def test_folding(self, group_manifest):
        events = [
            AttributionEvent(t_start_ms=0, t_end_ms=10_000, student_id="s1", confidence=0.9),
            AttributionEvent(t_start_ms=12_000, t_end_ms=18_000, student_id="s2", confidence=0.8),
            AttributionEvent(t_start_ms=20_000, t_end_ms=24_000, student_id="s1", confidence=0.85),
            AttributionEvent(t_start_ms=25_000, t_end_ms=27_000,
                             student_id=UNCERTAIN_SPEAKER, confidence=0.1),
            # gaze: s1 3/4 on camera
            *[
                BehavioralEvent(t_ms=t, kind=BehavioralKind.GAZE_SAMPLE, student_id="s1",
                                payload={"on_camera": on})
                for t, on in [(0, True), (300, True), (600, True), (900, False)]
            ],
            IntegrityFlag(t_ms=26_000, video_timecode="00:00:26",
                          kind=IntegrityKind.EXTRA_PERSON, note="check"),  # session-level
            IntegrityFlag(t_ms=13_000, video_timecode="00:00:13",
                          kind=IntegrityKind.STUDENT_ABSENT, note="s3 gone", student_id="s3"),
        ]
        summary = build_summary(group_manifest, events)

        s1, s2, s3 = summary.per_student
        assert s1.speaking_time_ms == 14_000 and s1.turn_count == 2
        assert s2.speaking_time_ms == 6_000 and s2.turn_count == 1
        assert s3.speaking_time_ms == 0

        # shares: denominator includes uncertain time (22s total)
        assert abs(s1.speaking_share - 14 / 22) < 1e-3
        assert summary.unattributed_speaking_ms == 2_000

        assert s1.attention_pct == 75.0
        assert s3.integrity_flags[0].kind == IntegrityKind.STUDENT_ABSENT
        assert len(summary.session_flags) == 1
        assert summary.timeline[0].t_start_ms == 0  # sorted

    def test_recording_ref_carried(self, individual_manifest):
        rec = RecordingRef(path="session_x.mp4", video_offset_ms=1500)
        summary = build_summary(individual_manifest, [], recording=rec)
        assert summary.recording.video_offset_ms == 1500

    def test_uncertain_never_lands_on_a_student(self, group_manifest):
        events = [
            AttributionEvent(t_start_ms=0, t_end_ms=5000,
                             student_id=UNCERTAIN_SPEAKER, confidence=0.0)
        ]
        summary = build_summary(group_manifest, events)
        assert all(s.speaking_time_ms == 0 for s in summary.per_student)
        assert summary.unattributed_speaking_ms == 5000
