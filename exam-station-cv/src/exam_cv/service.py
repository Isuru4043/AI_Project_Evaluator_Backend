"""Session runner: the two-rate loop, wired per the performance rules.

    frame rate (10–15 FPS): unrefined mesh → identity (embed only when due)
                            → lip activity → [window] speaker decision → turns
    tick rate  (2–5 FPS):   refined mesh (iris) → gaze/presence analyzers
    end of session:         flush turns → finalize recording → summary → sink

All components are injected so tests/eval clips run the identical loop with
FakeCamera, a fake embedder, and no recorder.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .behavior.analyzers import FaceTickObservation, GazeAnalyzer, PresenceAnalyzer
from .capture.camera import FrameSource
from .contracts.manifest import load_manifest, standalone_manifest
from .contracts.schemas import (
    IntegrityFlag,
    RecordingRef,
    SessionManifest,
    SessionMode,
)
from .contracts.sink import ArtifactSink, FileSink
from .events.store import append_event
from .report.summary import build_summary_from_log
from .speaker.attribution import (
    LipActivityTracker,
    TurnSegmenter,
    WindowObservation,
    decide_speaker,
)


@dataclass
class RunnerConfig:
    window_ms: int = 800       # speaker-decision window
    tick_ms: int = 333         # behavioral tick (~3 FPS)
    min_turn_ms: int = 700
    merge_gap_ms: int = 600


class SessionRunner:
    def __init__(
        self,
        manifest: SessionManifest,
        frames: FrameSource,
        mesh,                      # faces.mesh.MeshPipeline (or fake)
        identity_resolver,         # faces.identity.IdentityResolver (or fake)
        output_dir: Path,
        voice_active_fn=None,      # (t_ms) -> bool; None = assume speech when lips move
        recorder=None,             # capture.recorder.FfmpegRecorder or None
        audio=None,                # capture.audio.AudioCapture or None
        sink: Optional[ArtifactSink] = None,
        config: RunnerConfig = RunnerConfig(),
    ):
        self.manifest = manifest
        self.frames = frames
        self.mesh = mesh
        self.identity = identity_resolver
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.voice_active_fn = voice_active_fn
        self.recorder = recorder
        self.audio = audio
        self.sink = sink or FileSink(self.output_dir)
        self.cfg = config

        self.events_path = self.output_dir / f"session_{manifest.session_id}_events.jsonl"
        self._individual = manifest.mode == SessionMode.INDIVIDUAL

    def run(self):
        from .faces.mesh import (  # local import keeps fakes numpy-only
            coarse_gaze_on_camera,
            iris_gaze_on_camera,
            mouth_aspect_ratio,
        )

        roster_ids = [r.student_id for r in self.manifest.roster]
        lips = LipActivityTracker(window_ms=self.cfg.window_ms)
        segmenter = TurnSegmenter(
            window_ms=self.cfg.window_ms,
            min_turn_ms=self.cfg.min_turn_ms,
            merge_gap_ms=self.cfg.merge_gap_ms,
        )
        gaze = GazeAnalyzer()
        video_offset_ms = 0
        presence = PresenceAnalyzer(roster_ids, video_offset_ms=video_offset_ms)

        next_window = self.cfg.window_ms
        next_tick = 0
        speaker_candidates: set[str] = set(roster_ids)

        if self.recorder is not None:
            self.recorder.start()
        if self.audio is not None:
            self.audio.start()

        try:
            for frame in self.frames.frames():
                if self.recorder is not None:
                    self.recorder.write_frame(frame.image, frame.t_ms)

                # ---- frame-rate path (attribution) --------------------------
                observations = self.mesh.process_frame(frame.image)
                for lost in getattr(self.mesh.tracker, "lost_last_update", []):
                    self.identity.drop_track(lost)

                identified: dict[str, "FaceObservation"] = {}
                unknown_faces = 0
                for obs in observations:
                    if self._individual and len(observations) == 1:
                        sid = roster_ids[0]  # roster of one: no embedding needed
                    else:
                        sid = self.identity.resolve(
                            obs.track_id,
                            frame.t_ms,
                            crop_provider=lambda o=obs: self.mesh.crop(frame.image, o),
                        )
                    if sid is None:
                        unknown_faces += 1
                    else:
                        identified[sid] = obs
                        lips.push(sid, frame.t_ms, mouth_aspect_ratio(obs.landmarks))

                if frame.t_ms >= next_window:
                    voice = (
                        self.voice_active_fn(frame.t_ms)
                        if self.voice_active_fn is not None
                        else bool(lips.scores())
                    )
                    decision = decide_speaker(
                        WindowObservation(
                            t_ms=next_window - self.cfg.window_ms,
                            voice_active=voice,
                            lip_scores=lips.scores(),
                        )
                    )
                    if decision.student_id is not None:
                        speaker_candidates = {decision.student_id} & set(roster_ids) or set(roster_ids)
                    for turn in segmenter.push(decision):
                        append_event(self.events_path, turn)
                    next_window += self.cfg.window_ms

                # ---- tick-rate path (behavioral, advisory) ------------------
                if frame.t_ms >= next_tick:
                    tick_obs = self.mesh.process_tick(frame.image)
                    gaze_map: dict[str, bool] = {}
                    tick_unknown = 0
                    for obs in tick_obs:
                        if self._individual and len(tick_obs) == 1:
                            sid = roster_ids[0]
                        else:
                            sid = self.identity.resolve(
                                obs.track_id,
                                frame.t_ms,
                                crop_provider=lambda o=obs: self.mesh.crop(frame.image, o),
                            )
                        if sid is None:
                            tick_unknown += 1
                            continue
                        # Rule 5: iris-grade gaze only for speaker candidates;
                        # everyone else gets the coarse head-pose proxy.
                        if sid in speaker_candidates:
                            gaze_map[sid] = iris_gaze_on_camera(obs.landmarks)
                        else:
                            gaze_map[sid] = coarse_gaze_on_camera(obs.landmarks)

                    face_tick = FaceTickObservation(
                        t_ms=frame.t_ms,
                        gaze_on_camera=gaze_map,
                        unknown_face_count=max(unknown_faces, tick_unknown),
                    )
                    for ev in gaze.push(face_tick):
                        append_event(self.events_path, ev)
                    for ev in presence.push(face_tick):
                        append_event(self.events_path, ev)
                    next_tick += self.cfg.tick_ms

            # ---- end of session ------------------------------------------
            for turn in segmenter.flush():
                append_event(self.events_path, turn)
        finally:
            self.frames.close()
            if self.audio is not None:
                self.audio.close()

        recording: Optional[RecordingRef] = None
        if self.recorder is not None:
            wav = getattr(self.audio, "wav_path", None) if self.audio else None
            recording = self.recorder.finalize(wav_path=wav)

        summary = build_summary_from_log(self.manifest, self.events_path, recording)
        self.sink.publish(
            summary,
            events_path=self.events_path,
            recording_path=Path(recording.path) if recording else None,
        )
        return summary


# ---------------------------------------------------------------------------
# CLI entrypoint (live exam station)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="VivaSense exam-station CV service")
    parser.add_argument("--manifest", type=Path, help="platform-issued session manifest JSON")
    parser.add_argument(
        "--students", nargs="*", help="standalone mode: student display names"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./sessions"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        manifest = load_manifest(args.manifest)
    elif args.students:
        manifest = standalone_manifest(args.students)
    else:
        parser.error("provide --manifest or --students")

    import time

    from .capture.audio import AudioCapture, EnergyVAD, SileroVAD
    from .capture.camera import Camera
    from .capture.recorder import FfmpegRecorder
    from .faces.identity import EnrollmentGallery, IdentityResolver
    from .faces.mesh import MeshPipeline

    t0 = time.monotonic()
    out = args.output_dir / manifest.session_id
    out.mkdir(parents=True, exist_ok=True)

    camera = Camera(device_index=args.camera, t0_monotonic=t0)
    mesh = MeshPipeline(max_faces=max(5, len(manifest.roster) + 1))

    # ArcFace is only needed to tell students apart in group mode. Individual
    # mode shortcuts identity to the single roster entry and never embeds, so
    # skip loading the recognition model (and its download) entirely.
    gallery = EnrollmentGallery()
    if manifest.mode == SessionMode.GROUP:
        from .faces.identity import ArcFaceEmbedder

        embedder = ArcFaceEmbedder()
        _enroll_interactively(manifest, camera, mesh, gallery, embedder=embedder)
    else:
        embedder = _UnusedEmbedder()
    resolver = IdentityResolver(gallery, embedder)

    recorder = None
    if not args.no_record:
        recorder = FfmpegRecorder(
            manifest.session_id, out, camera.width, camera.height, camera.fps
        )

    wav_path = out / f"session_{manifest.session_id}_audio.wav"
    audio = AudioCapture(wav_path=wav_path, t0_monotonic=t0)
    audio.wav_path = wav_path
    try:
        vad = SileroVAD()
    except Exception:
        vad = EnergyVAD()

    voice_state = {"active": False}

    def voice_active(t_ms: int) -> bool:
        while True:
            block = audio.get_block(timeout=0.0)
            if block is None:
                break
            voice_state["active"] = vad.is_speech(block)
        return voice_state["active"]

    runner = SessionRunner(
        manifest=manifest,
        frames=camera,
        mesh=mesh,
        identity_resolver=resolver,
        output_dir=out,
        voice_active_fn=voice_active,
        recorder=recorder,
        audio=audio,
    )
    summary = runner.run()
    print(f"Session complete. Artifact: {out}")
    for s in summary.per_student:
        print(
            f"  {s.display_name}: {s.turn_count} turns, "
            f"{s.speaking_time_ms/1000:.0f}s speaking, "
            f"{len(s.integrity_flags)} flags"
        )


class _UnusedEmbedder:
    """Placeholder for individual mode, where identity is the single roster
    entry and no embedding ever happens. Raises loudly if that assumption
    is ever violated."""

    def embed(self, face_crop_bgr):
        raise RuntimeError(
            "embedder called in individual mode — identity should shortcut "
            "to the sole roster entry without embedding"
        )


def _enroll_interactively(manifest, camera, mesh, gallery, embedder, snapshots: int = 3) -> None:
    """Group-mode enrollment: each student faces the camera alone; we capture
    N snapshots and store their embeddings. Console-guided for v1."""
    frame_iter = camera.frames()
    for entry in manifest.roster:
        input(f"Enrollment: {entry.display_name} alone in frame, then press Enter…")
        captured = 0
        for frame in frame_iter:
            faces = mesh.process_frame(frame.image)
            if len(faces) != 1:
                continue
            gallery.enroll(
                entry.student_id, embedder.embed(mesh.crop(frame.image, faces[0]))
            )
            captured += 1
            if captured >= snapshots:
                break
        print(f"  enrolled {entry.display_name} ({captured} snapshots)")


if __name__ == "__main__":
    main()
