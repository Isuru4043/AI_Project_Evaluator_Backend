"""Post-hoc analysis of a recorded viva session (seam-2 alternative path).

The live viva records in the browser (Agora owns the webcam), so the CV
engine analyzes the recording afterwards:

    python -m exam_cv.analyze --video rec.webm --manifest manifest.json \
        --output-dir out/

Pipeline: ffmpeg extracts the audio track → offline VAD builds a
voice-activity timeline → video frames drive the SAME SessionRunner used
live (FakeCamera-style source with real file timing) → the standard
summary artifact lands in --output-dir as session_<id>_summary.json.

Timecodes: frame t_ms is derived from the real frame index / file fps, so
every event timestamp and integrity-flag timecode is directly seekable in
the analyzed recording (video_offset_ms = 0).

Group identity (single-camera recordings): there is no enrollment step
post-hoc, so v1 uses SEATING ORDER — students sit left→right in roster
order. When the number of visible faces doesn't match the roster the
mapping is left sticky/unknown rather than guessed (HITL invariant).
Enrollment-snapshot identity can replace this behind the same interface.
"""

from __future__ import annotations

import argparse
import bisect
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

from .capture.camera import Frame
from .contracts.manifest import load_manifest
from .contracts.schemas import SessionMode
from .service import RunnerConfig, SessionRunner, _UnusedEmbedder


class VideoFileFrames:
    """Frame source over a recorded file with REAL timing.

    Reads the container fps and yields every k-th frame so the effective
    rate is ~target_fps, with t_ms computed from the true frame index —
    event timestamps stay aligned to the recording.
    """

    def __init__(self, video_path: Path, target_fps: float = 12.0):
        import cv2  # lazy

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {video_path}")
        file_fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not 1.0 <= file_fps <= 120.0:  # webm sometimes reports 0/1000
            file_fps = 30.0
        self.file_fps = file_fps
        self._stride = max(1, round(file_fps / target_fps))
        self.fps = file_fps / self._stride
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[Frame]:
        index = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            if index % self._stride == 0:
                yield Frame(t_ms=int(index * 1000 / self.file_fps), image=image)
            index += 1

    def close(self) -> None:
        self._cap.release()


class OfflineVAD:
    """Voice-activity lookup built once from the recording's audio track."""

    BLOCK_MS = 96  # 3 × 512 samples @ 16 kHz — matches live AudioCapture

    def __init__(self, wav_path: Optional[Path]):
        self._starts: list[int] = []
        self._active: list[bool] = []
        if wav_path is None:
            return

        import numpy as np
        import soundfile as sf  # lazy

        from .capture.audio import SAMPLE_RATE, AudioBlock, EnergyVAD

        try:
            from .capture.audio import SileroVAD

            vad = SileroVAD()
        except Exception:
            vad = EnergyVAD()

        samples, rate = sf.read(str(wav_path), dtype='float32')
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        assert rate == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz wav, got {rate}"

        block = SAMPLE_RATE * self.BLOCK_MS // 1000
        for start in range(0, len(samples) - block + 1, block):
            t_ms = start * 1000 // SAMPLE_RATE
            chunk = samples[start:start + block]
            self._starts.append(t_ms)
            self._active.append(vad.is_speech(AudioBlock(t_ms=t_ms, samples=chunk)))

    def voice_active(self, t_ms: int) -> bool:
        if not self._starts:
            return False
        i = bisect.bisect_right(self._starts, t_ms) - 1
        return self._active[max(i, 0)]

    @property
    def has_audio(self) -> bool:
        return bool(self._starts)


class PositionalIdentity:
    """Seating-order identity for single-camera group recordings.

    When exactly len(roster) faces are visible, tracks map to students
    left→right in roster order and the mapping sticks to each track.
    Otherwise unmapped tracks resolve to None (unknown) — never guessed.
    Fed by SeatingMesh below; same interface as IdentityResolver.
    """

    def __init__(self, roster_ids: list[str]):
        self.roster_ids = list(roster_ids)
        self._centers: dict[int, float] = {}
        self._sticky: dict[int, str] = {}

    def observe(self, observations) -> None:
        self._centers = {
            o.track_id: (o.bbox[0] + o.bbox[2]) / 2 for o in observations
        }
        if len(self._centers) == len(self.roster_ids):
            for sid, tid in zip(
                self.roster_ids, sorted(self._centers, key=self._centers.get)
            ):
                self._sticky[tid] = sid

    def resolve(self, track_id, t_ms, crop_provider) -> Optional[str]:
        return self._sticky.get(track_id)

    def drop_track(self, track_id) -> None:
        self._sticky.pop(track_id, None)


class SeatingMesh:
    """MeshPipeline wrapper that feeds PositionalIdentity each frame."""

    def __init__(self, mesh, identity: PositionalIdentity):
        self._mesh = mesh
        self._identity = identity
        self.tracker = mesh.tracker

    def process_frame(self, image):
        obs = self._mesh.process_frame(image)
        self._identity.observe(obs)
        return obs

    def process_tick(self, image):
        obs = self._mesh.process_tick(image)
        self._identity.observe(obs)
        return obs

    def crop(self, image, obs, pad=0.2):
        return self._mesh.crop(image, obs, pad)

    def close(self):
        self._mesh.close()


def extract_audio(video_path: Path, out_dir: Path) -> Optional[Path]:
    """Extract mono 16 kHz wav; returns None when there's no audio stream."""
    from .capture.recorder import resolve_ffmpeg

    wav = out_dir / 'audio_16k.wav'
    result = subprocess.run(
        [resolve_ffmpeg(), '-y', '-i', str(video_path),
         '-vn', '-ac', '1', '-ar', '16000', str(wav)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
        return None
    return wav


def analyze(
    video_path: Path,
    manifest_path: Path,
    output_dir: Path,
    target_fps: float = 12.0,
):
    from .faces.mesh import MeshPipeline

    manifest = load_manifest(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Re-analysis must not append onto a previous run's event log.
    stale = output_dir / f"session_{manifest.session_id}_events.jsonl"
    stale.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix='exam_cv_audio_') as tmp:
        wav = extract_audio(video_path, Path(tmp))
        vad = OfflineVAD(wav)

    frames = VideoFileFrames(video_path, target_fps=target_fps)
    mesh = MeshPipeline(max_faces=max(5, len(manifest.roster) + 1))

    if manifest.mode == SessionMode.GROUP:
        identity = PositionalIdentity([r.student_id for r in manifest.roster])
        mesh = SeatingMesh(mesh, identity)
    else:
        from .faces.identity import EnrollmentGallery, IdentityResolver

        identity = IdentityResolver(EnrollmentGallery(), _UnusedEmbedder())

    runner = SessionRunner(
        manifest=manifest,
        frames=frames,
        mesh=mesh,
        identity_resolver=identity,
        output_dir=output_dir,
        # No audio stream → fall back to lips-only speech detection.
        voice_active_fn=vad.voice_active if vad.has_audio else None,
        recorder=None,  # the recording already exists; we're analyzing it
        audio=None,
        config=RunnerConfig(),
    )
    return runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc CV/behavioral analysis of a recorded viva",
    )
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--target-fps', type=float, default=12.0)
    args = parser.parse_args()

    summary = analyze(args.video, args.manifest, args.output_dir, args.target_fps)
    print(f"analysis complete: session {summary.session_id}", flush=True)
    for s in summary.per_student:
        print(
            f"  {s.display_name}: {s.turn_count} turns, "
            f"{s.speaking_time_ms / 1000:.1f}s speaking, "
            f"attention={s.attention_pct}, flags={len(s.integrity_flags)}",
            flush=True,
        )
    sys.exit(0)


if __name__ == '__main__':
    main()
