"""Time-boxed live smoke test of the real CV pipeline (~6s), then finalize."""
import sys, time, traceback
from pathlib import Path

from exam_cv.contracts.manifest import standalone_manifest
from exam_cv.service import SessionRunner, RunnerConfig, _UnusedEmbedder

DURATION_S = 6
OUT = Path(sys.argv[1])


class TimedCamera:
    """Wrap the real Camera; stop after DURATION_S so run() returns cleanly."""
    def __init__(self, cam, seconds):
        self.cam, self.seconds = cam, seconds
        self.width, self.height, self.fps = cam.width, cam.height, cam.fps

    def frames(self):
        start = None
        for f in self.cam.frames():
            if start is None:
                start = time.monotonic()
            if time.monotonic() - start > self.seconds:
                return
            yield f

    def close(self):
        self.cam.close()


def main():
    from exam_cv.capture.camera import Camera
    from exam_cv.capture.recorder import FfmpegRecorder
    from exam_cv.capture.audio import AudioCapture, EnergyVAD, SileroVAD
    from exam_cv.faces.mesh import MeshPipeline
    from exam_cv.faces.identity import EnrollmentGallery, IdentityResolver

    manifest = standalone_manifest(["Test"])
    out = OUT / manifest.session_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    print("opening camera...", flush=True)
    cam = Camera(device_index=0, t0_monotonic=t0)
    print(f"camera {cam.width}x{cam.height} @ target {cam.fps}fps", flush=True)

    print("loading FaceLandmarker (downloads model on first run)...", flush=True)
    mesh = MeshPipeline(max_faces=3)

    recorder = FfmpegRecorder(manifest.session_id, out, cam.width, cam.height, cam.fps)
    wav = out / f"session_{manifest.session_id}_audio.wav"
    audio = AudioCapture(wav_path=wav, t0_monotonic=t0)
    audio.wav_path = wav
    try:
        vad = SileroVAD(); print("VAD: silero", flush=True)
    except Exception as e:
        vad = EnergyVAD(); print(f"VAD: energy fallback ({type(e).__name__})", flush=True)

    state = {"active": False}
    def voice_active(t_ms):
        while True:
            b = audio.get_block(timeout=0.0)
            if b is None:
                break
            state["active"] = vad.is_speech(b)
        return state["active"]

    resolver = IdentityResolver(EnrollmentGallery(), _UnusedEmbedder())
    runner = SessionRunner(
        manifest=manifest,
        frames=TimedCamera(cam, DURATION_S),
        mesh=mesh,
        identity_resolver=resolver,
        output_dir=out,
        voice_active_fn=voice_active,
        recorder=recorder,
        audio=audio,
        config=RunnerConfig(),
    )
    print(f"recording ~{DURATION_S}s — look at the camera and say something...", flush=True)
    summary = runner.run()
    print("\n=== DONE ===", flush=True)
    print("artifact dir:", out, flush=True)
    for s in summary.per_student:
        print(f"  {s.display_name}: {s.turn_count} turns, "
              f"{s.speaking_time_ms/1000:.1f}s speaking, "
              f"attention={s.attention_pct}, flags={len(s.integrity_flags)}", flush=True)
    if summary.recording:
        r = summary.recording
        print(f"  recording: {r.path} (offset={r.video_offset_ms}ms, "
              f"dur={r.duration_ms}ms, drift={r.drift_ms_at_end}ms)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
