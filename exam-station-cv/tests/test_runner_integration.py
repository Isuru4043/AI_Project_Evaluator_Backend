"""Full SessionRunner loop with a scripted scene — no camera, no mediapipe.

Scenario (group of 2, 15 FPS, ~24s):
  0–8s   : Alice (s1) speaks, both on camera
  8–10s  : silence
  10–18s : Bob (s2) speaks
  18–24s : silence; Bob's gaze goes off-camera the whole time (glance);
           an unknown face appears at 20s (integrity flag)

Asserts attribution lands on the right students, advisory events exist, and
the fake embedder's call count proves embeds stayed out of the frame loop.
"""

import numpy as np

from exam_cv.capture.camera import FakeCamera
from exam_cv.contracts.schemas import BehavioralKind, IntegrityKind
from exam_cv.faces.identity import EnrollmentGallery, IdentityResolver
from exam_cv.faces.mesh import FaceObservation
from exam_cv.service import RunnerConfig, SessionRunner

from test_identity_and_tracker import FakeEmbedder, synthetic_landmarks

FPS = 15.0
STEP_MS = 1000.0 / FPS
DURATION_S = 24


class FakeMesh:
    """Scripted scene: decodes the frame index from the synthetic image and
    returns FaceObservations per the scenario above."""

    BBOX_S1 = (0.05, 0.2, 0.35, 0.7)
    BBOX_S2 = (0.55, 0.2, 0.85, 0.7)
    BBOX_UNKNOWN = (0.40, 0.1, 0.55, 0.4)

    def __init__(self):
        self.tracker = type("T", (), {"lost_last_update": []})()

    def _scene(self, image) -> list[FaceObservation]:
        i = int(image[0, 0, 0]) | (int(image[0, 0, 1]) << 8)
        t_s = i * STEP_MS / 1000.0

        # mouth motion: oscillate MAR while "speaking"
        def mouth(speaking: bool) -> float:
            return (0.30 if i % 2 else 0.05) if speaking else 0.05

        s1_speaking = t_s < 8
        s2_speaking = 10 <= t_s < 18
        s2_away = t_s >= 18  # looks away: head turned AND irises off-center

        faces = [
            FaceObservation(1, self.BBOX_S1,
                            synthetic_landmarks(mouth_open=mouth(s1_speaking))),
            FaceObservation(2, self.BBOX_S2,
                            synthetic_landmarks(mouth_open=mouth(s2_speaking),
                                                nose_shift=0.15 if s2_away else 0.0,
                                                iris_shift=0.06 if s2_away else 0.0)),
        ]
        if t_s >= 20:
            faces.append(
                FaceObservation(3, self.BBOX_UNKNOWN, synthetic_landmarks())
            )
        return faces

    def process_frame(self, image):
        return self._scene(image)

    def process_tick(self, image):
        return self._scene(image)

    def crop(self, image, obs, pad=0.2):
        # marker crop the FakeEmbedder maps to a stable identity
        return {1: "s1", 2: "s2", 3: "stranger"}[obs.track_id]


def make_frames() -> list[np.ndarray]:
    frames = []
    for i in range(int(DURATION_S * FPS)):
        img = np.zeros((2, 2, 3), dtype=np.uint16)
        img[0, 0, 0] = i & 0xFF
        img[0, 0, 1] = i >> 8
        frames.append(img)
    return frames


def voice_active(t_ms: int) -> bool:
    t_s = t_ms / 1000.0
    return t_s < 8 or 10 <= t_s < 18


def test_group_session_end_to_end(group_manifest, tmp_path):
    # roster of 3 in the fixture; only s1/s2 appear on camera → s3 absent
    embedder = FakeEmbedder()
    gallery = EnrollmentGallery()
    for sid in ("s1", "s2", "s3"):
        gallery.enroll(sid, embedder.vector_for(sid))
    resolver = IdentityResolver(gallery, embedder, reverify_ms=12_000)

    runner = SessionRunner(
        manifest=group_manifest,
        frames=FakeCamera(synthetic_frames=make_frames(), fps=FPS),
        mesh=FakeMesh(),
        identity_resolver=resolver,
        output_dir=tmp_path,
        voice_active_fn=voice_active,
        config=RunnerConfig(),
    )
    summary = runner.run()

    by_id = {s.student_id: s for s in summary.per_student}

    # --- attribution: right students, right ballpark of speaking time ------
    assert by_id["s1"].turn_count >= 1
    assert by_id["s2"].turn_count >= 1
    assert 6_000 <= by_id["s1"].speaking_time_ms <= 9_500
    assert 6_000 <= by_id["s2"].speaking_time_ms <= 9_500
    assert by_id["s3"].speaking_time_ms == 0
    for turn in summary.timeline:
        if turn.student_id == "s1":
            assert turn.t_end_ms <= 9_000
        if turn.student_id == "s2":
            assert 9_000 <= turn.t_start_ms <= 19_000

    # --- advisory outputs ---------------------------------------------------
    assert by_id["s1"].attention_pct and by_id["s1"].attention_pct > 90
    assert by_id["s2"].attention_pct and by_id["s2"].attention_pct < 90  # looked away 18–24s

    # s3 never showed up → absence integrity flag
    assert any(
        f.kind == IntegrityKind.STUDENT_ABSENT for f in by_id["s3"].integrity_flags
    )
    # unknown face at 20s → session-level flag with a video timecode
    extra = [
        f for f in summary.session_flags
        if f.kind in (IntegrityKind.EXTRA_PERSON, IntegrityKind.UNKNOWN_FACE)
    ]
    assert len(extra) == 1
    assert extra[0].video_timecode.startswith("00:00:2")

    # --- perf rule 2: embeds out of the frame loop --------------------------
    # 360 frames + tick calls; stable tracks re-verify every 12s → a handful
    # of embeds, not hundreds.
    assert embedder.calls <= 12, f"embedder ran {embedder.calls}× — in the frame loop?"

    # artifact file published
    assert (tmp_path / f"session_{group_manifest.session_id}_summary.json").exists()


def test_individual_session_no_embeds(individual_manifest, tmp_path):
    """Individual mode with one face: identity is the roster entry — the
    embedder must never run."""
    embedder = FakeEmbedder()
    gallery = EnrollmentGallery()
    gallery.enroll("s1", embedder.vector_for("s1"))
    resolver = IdentityResolver(gallery, embedder)
    baseline_calls = embedder.calls

    class OneFaceMesh(FakeMesh):
        def _scene(self, image):
            i = int(image[0, 0, 0]) | (int(image[0, 0, 1]) << 8)
            speaking = (i * STEP_MS / 1000.0) < 5
            mar = (0.30 if i % 2 else 0.05) if speaking else 0.05
            return [FaceObservation(1, self.BBOX_S1,
                                    synthetic_landmarks(mouth_open=mar))]

    frames = make_frames()[: int(10 * FPS)]
    runner = SessionRunner(
        manifest=individual_manifest,
        frames=FakeCamera(synthetic_frames=frames, fps=FPS),
        mesh=OneFaceMesh(),
        identity_resolver=resolver,
        output_dir=tmp_path,
        voice_active_fn=lambda t: t < 5000,
    )
    summary = runner.run()

    assert embedder.calls == baseline_calls  # roster-of-one shortcut
    s1 = summary.per_student[0]
    assert s1.turn_count >= 1
    assert 3_500 <= s1.speaking_time_ms <= 6_500
