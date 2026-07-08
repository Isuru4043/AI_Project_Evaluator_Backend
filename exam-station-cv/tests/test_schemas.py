"""Contract schemas: timecode math, manifest shape, artifact roundtrip."""

import pytest
from pydantic import ValidationError

from exam_cv.contracts.manifest import standalone_manifest
from exam_cv.contracts.schemas import (
    SCHEMA_VERSION,
    AttributionEvent,
    IntegrityFlag,
    IntegrityKind,
    RecordingRef,
    SessionMode,
    SessionSummary,
    StudentSummary,
    ms_to_timecode,
)


class TestTimecode:
    def test_zero(self):
        assert ms_to_timecode(0) == "00:00:00"

    def test_full(self):
        assert ms_to_timecode(3_723_000) == "01:02:03"

    def test_floors_sub_second(self):
        assert ms_to_timecode(14 * 60_000 + 32_999) == "00:14:32"

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            ms_to_timecode(-1)


class TestRecordingRef:
    def test_timecode_uses_offset(self):
        rec = RecordingRef(path="x.mp4", video_offset_ms=2000)
        # session t=62s → video t=60s
        assert rec.timecode_for(62_000) == "00:01:00"

    def test_timecode_clamps_before_video_start(self):
        rec = RecordingRef(path="x.mp4", video_offset_ms=5000)
        assert rec.timecode_for(1000) == "00:00:00"


class TestIntegrityFlag:
    def test_at_computes_video_timecode(self):
        flag = IntegrityFlag.at(
            t_ms=872_000 + 3000,
            kind=IntegrityKind.EXTRA_PERSON,
            note="review recording",
            video_offset_ms=3000,
        )
        assert flag.video_timecode == "00:14:32"


class TestManifest:
    def test_standalone_individual(self):
        m = standalone_manifest(["Alice"])
        assert m.mode == SessionMode.INDIVIDUAL
        assert len(m.roster) == 1

    def test_standalone_group(self):
        m = standalone_manifest(["Alice", "Bob"])
        assert m.mode == SessionMode.GROUP

    def test_empty_roster_rejected(self):
        with pytest.raises(ValueError):
            standalone_manifest([])


class TestSummaryRoundtrip:
    def test_json_roundtrip(self, group_manifest):
        summary = SessionSummary(
            session_id=group_manifest.session_id,
            mode=group_manifest.mode,
            per_student=[
                StudentSummary(student_id="s1", display_name="Alice", turn_count=2)
            ],
            timeline=[
                AttributionEvent(
                    t_start_ms=0, t_end_ms=4000, student_id="s1", confidence=0.9
                )
            ],
        )
        parsed = SessionSummary.model_validate_json(summary.model_dump_json())
        assert parsed.schema_version == SCHEMA_VERSION
        assert parsed.timeline[0].student_id == "s1"

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            AttributionEvent(t_start_ms=0, t_end_ms=1, student_id="s1", confidence=1.5)
