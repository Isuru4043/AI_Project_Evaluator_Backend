"""Session recording: full A/V of the viva for post-hoc lecturer review.

Design: video frames (the same tee the analyzer sees) are piped raw into an
ffmpeg subprocess encoding H.264; audio is written to a WAV alongside by
AudioCapture; at session end a second, fast ffmpeg pass muxes video + WAV
into the final session_<id>.mp4. Two single-input pipes are far more robust
than one dual-input pipe, and the mux pass is stream-copy (seconds).

The recording starts at session t0, so video time == session time minus
video_offset_ms (the delay between t0 and the first piped frame). Integrity
flags carry HH:MM:SS timecodes computed from that mapping so the lecturer
can jump straight to a flagged moment.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ..contracts.schemas import RecordingRef


def resolve_ffmpeg() -> str:
    """Locate the ffmpeg executable robustly.

    Order: EXAM_CV_FFMPEG env override → PATH → known winget install location.
    Depending on PATH alone is fragile on Windows (a shell opened before the
    install won't see it), so we fall back to the winget package dir.
    """
    override = os.getenv("EXAM_CV_FFMPEG")
    if override and Path(override).exists():
        return override

    found = shutil.which("ffmpeg")
    if found:
        return found

    # winget installs Gyan.FFmpeg here; the version dir varies.
    local = os.getenv("LOCALAPPDATA", "")
    if local:
        patterns = [
            os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
            os.path.join(
                local, "Microsoft", "WinGet", "Packages",
                "Gyan.FFmpeg*", "ffmpeg-*", "bin", "ffmpeg.exe",
            ),
        ]
        for pat in patterns:
            hits = sorted(glob.glob(pat))
            if hits:
                return hits[-1]

    raise FileNotFoundError(
        "ffmpeg not found. Install it (winget install Gyan.FFmpeg), then open a "
        "NEW terminal, or set EXAM_CV_FFMPEG to the ffmpeg.exe path."
    )


class FfmpegRecorder:
    def __init__(
        self,
        session_id: str,
        output_dir: Path,
        width: int,
        height: int,
        fps: float,
        ffmpeg_bin: str | None = None,
    ):
        self.session_id = session_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_bin = ffmpeg_bin or resolve_ffmpeg()

        self.video_path = self.output_dir / f"session_{session_id}_video.mp4"
        self.final_path = self.output_dir / f"session_{session_id}.mp4"
        self._proc: Optional[subprocess.Popen] = None
        self._first_frame_t_ms: Optional[int] = None
        self._last_frame_t_ms: Optional[int] = None
        self._frames_written = 0

    def start(self) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            str(self.video_path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_frame(self, image: np.ndarray, t_ms: int) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("recorder not started")
        if self._first_frame_t_ms is None:
            self._first_frame_t_ms = t_ms
        self._last_frame_t_ms = t_ms
        self._proc.stdin.write(image.tobytes())
        self._frames_written += 1

    def finalize(self, wav_path: Path | None = None) -> RecordingRef:
        """Close the video pipe, correct the frame-rate mismatch, mux audio.

        A webcam rarely delivers exactly the nominal fps, but the raw stream
        was encoded assuming constant `self.fps`. Left uncorrected the video
        plays too fast/slow and drifts out of sync with the (real-time) audio.
        We measure the true average rate from wall-clock frame timestamps and
        rescale the video's timestamps with ffmpeg `-itsscale` — a stream copy,
        no re-encode — so the video duration matches wall-clock and the audio.
        """
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.close()
            self._proc.wait(timeout=120)
        offset = self._first_frame_t_ms or 0

        wall_ms = None
        itsscale = 1.0
        if (
            self._last_frame_t_ms is not None
            and self._first_frame_t_ms is not None
            and self._frames_written > 1
        ):
            # wall span of all frames (last frame still shows for ~1/fps)
            wall_ms = (self._last_frame_t_ms - self._first_frame_t_ms) + int(
                1000 / self.fps
            )
            encoded_ms = self._frames_written * 1000 / self.fps
            if encoded_ms > 0:
                itsscale = wall_ms / encoded_ms

        has_audio = wav_path is not None and Path(wav_path).exists()
        if has_audio:
            cmd = [
                self.ffmpeg_bin, "-y",
                "-itsscale", f"{itsscale:.6f}", "-i", str(self.video_path),
                "-i", str(wav_path),
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                str(self.final_path),
            ]
        else:
            cmd = [
                self.ffmpeg_bin, "-y",
                "-itsscale", f"{itsscale:.6f}", "-i", str(self.video_path),
                "-c:v", "copy",
                str(self.final_path),
            ]
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.video_path.unlink(missing_ok=True)

        # Residual drift after correction (QA; should be ~0).
        drift = None
        if wall_ms is not None:
            corrected_ms = int(self._frames_written * 1000 / self.fps * itsscale)
            drift = corrected_ms - wall_ms

        return RecordingRef(
            path=str(self.final_path),
            video_offset_ms=offset,
            duration_ms=wall_ms,
            drift_ms_at_end=drift,
        )
