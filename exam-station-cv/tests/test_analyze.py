"""Post-hoc analysis: positional identity, offline VAD, and CLI plumbing."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from exam_cv.analyze import OfflineVAD, PositionalIdentity
from exam_cv.faces.mesh import FaceObservation

import numpy as np


def obs(track_id: int, x0: float) -> FaceObservation:
    return FaceObservation(
        track_id=track_id,
        bbox=(x0, 0.2, x0 + 0.2, 0.6),
        landmarks=np.zeros((4, 2), dtype=np.float32),
    )


class TestPositionalIdentity:
    def test_left_to_right_maps_roster_order(self):
        ident = PositionalIdentity(["s1", "s2", "s3"])
        # tracks appear out of order; positions: t7 left, t2 middle, t5 right
        ident.observe([obs(5, 0.7), obs(7, 0.05), obs(2, 0.4)])
        assert ident.resolve(7, 0, None) == "s1"
        assert ident.resolve(2, 0, None) == "s2"
        assert ident.resolve(5, 0, None) == "s3"

    def test_mapping_sticks_when_someone_leaves(self):
        ident = PositionalIdentity(["s1", "s2"])
        ident.observe([obs(1, 0.1), obs(2, 0.6)])
        assert ident.resolve(1, 0, None) == "s1"
        # s1's face drops out — remaining track keeps its identity,
        # no re-mapping happens with a mismatched count
        ident.observe([obs(2, 0.6)])
        assert ident.resolve(2, 1, None) == "s2"
        assert ident.resolve(99, 1, None) is None  # new track: unknown, not guessed

    def test_drop_track_forgets(self):
        ident = PositionalIdentity(["s1", "s2"])
        ident.observe([obs(1, 0.1), obs(2, 0.6)])
        ident.drop_track(1)
        ident.observe([obs(2, 0.6)])  # count mismatch → no remap
        assert ident.resolve(1, 5, None) is None


class TestOfflineVAD:
    def test_no_audio_stream(self):
        vad = OfflineVAD(None)
        assert not vad.has_audio
        assert vad.voice_active(1234) is False

    def test_lookup_from_wav(self, tmp_path):
        sf = pytest.importorskip("soundfile")
        # 1s silence + 1s loud tone + 1s silence @16k mono
        rate = 16000
        t = np.linspace(0, 1, rate, endpoint=False)
        tone = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8
        silence = np.zeros(rate, dtype=np.float32)
        wav = tmp_path / "a.wav"
        sf.write(str(wav), np.concatenate([silence, tone, silence]), rate)

        vad = OfflineVAD(wav)
        assert vad.has_audio
        # Energy fallback or silero — both must see the tone-vs-silence contrast
        assert vad.voice_active(1500) or not vad.voice_active(500)


@pytest.mark.skipif(
    not (Path(sys.prefix) / "Lib" / "site-packages" / "mediapipe").exists(),
    reason="station extras (mediapipe) not installed",
)
def test_cli_plumbing_on_synthetic_video(tmp_path):
    """End-to-end CLI on a faceless synthetic clip: must exit 0 and write a
    valid empty-ish summary (no faces → no turns; plumbing still works)."""
    from exam_cv.capture.recorder import resolve_ffmpeg

    video = tmp_path / "clip.mp4"
    subprocess.run(
        [resolve_ffmpeg(), "-y", "-f", "lavfi", "-i",
         "testsrc=duration=2:size=320x240:rate=12",
         "-pix_fmt", "yuv420p", str(video)],
        check=True, capture_output=True,
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "1.0",
        "session_id": "plumb-1",
        "mode": "individual",
        "roster": [{"student_id": "s1", "display_name": "Alice"}],
        "t0_utc": "2026-07-06T09:00:00+00:00",
    }), encoding="utf-8")

    out = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "exam_cv.analyze",
         "--video", str(video), "--manifest", str(manifest),
         "--output-dir", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-800:]

    summary_path = out / "session_plumb-1_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["session_id"] == "plumb-1"
    assert summary["per_student"][0]["turn_count"] == 0
