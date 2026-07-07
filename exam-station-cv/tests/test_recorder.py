"""FfmpegRecorder smoke test — skipped only when ffmpeg cannot be located.

Run this on the exam-station machine (ffmpeg required there anyway).
"""

import numpy as np
import pytest

from exam_cv.capture.recorder import FfmpegRecorder, resolve_ffmpeg


def _ffmpeg_available() -> bool:
    try:
        resolve_ffmpeg()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _ffmpeg_available(), reason="ffmpeg not installed"
)

W, H, FPS = 320, 240, 15.0


def test_video_only_recording(tmp_path):
    rec = FfmpegRecorder("t1", tmp_path, W, H, FPS)
    rec.start()
    for i in range(30):  # 2s of frames
        img = np.full((H, W, 3), i * 8 % 255, dtype=np.uint8)
        rec.write_frame(img, t_ms=1000 + int(i * 1000 / FPS))  # first frame at t=1s
    ref = rec.finalize()

    assert ref.video_offset_ms == 1000  # session t of frame 0
    assert (tmp_path / "session_t1.mp4").exists()
    assert (tmp_path / "session_t1.mp4").stat().st_size > 0
    # timecode mapping: session t=61s → video 00:01:00
    assert ref.timecode_for(61_000) == "00:01:00"
    # drift QA: 30 frames at nominal fps vs wall span should be ~0 here
    assert abs(ref.drift_ms_at_end) < 100


def test_mux_with_wav(tmp_path):
    import wave

    wav_path = tmp_path / "audio.wav"
    with wave.open(str(wav_path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        t = np.linspace(0, 2, 32000)
        w.writeframes((np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16).tobytes())

    rec = FfmpegRecorder("t2", tmp_path, W, H, FPS)
    rec.start()
    for i in range(30):
        rec.write_frame(np.zeros((H, W, 3), dtype=np.uint8), t_ms=int(i * 1000 / FPS))
    ref = rec.finalize(wav_path=wav_path)

    final = tmp_path / "session_t2.mp4"
    assert final.exists() and final.stat().st_size > 0
    assert not (tmp_path / "session_t2_video.mp4").exists()  # intermediate cleaned
