"""Identity resolution (embed-budget rules) + IoU tracker + landmark features."""

import numpy as np

from exam_cv.faces.identity import EnrollmentGallery, IdentityResolver, cosine
from exam_cv.faces.mesh import (
    IoUTracker,
    coarse_gaze_on_camera,
    iris_gaze_on_camera,
    mouth_aspect_ratio,
)


class FakeEmbedder:
    """Returns a fixed embedding per 'face' (crop is a 1x1 marker array)."""

    def __init__(self):
        self.calls = 0
        self._vectors = {}

    def vector_for(self, key: str) -> np.ndarray:
        if key not in self._vectors:
            rng = np.random.default_rng(abs(hash(key)) % 2**32)
            v = rng.normal(size=64)
            self._vectors[key] = v / np.linalg.norm(v)
        return self._vectors[key]

    def embed(self, crop) -> np.ndarray:
        self.calls += 1
        return self.vector_for(str(crop))


def make_resolver(reverify_ms=10_000):
    emb = FakeEmbedder()
    gallery = EnrollmentGallery()
    for sid in ("s1", "s2"):
        gallery.enroll(sid, emb.vector_for(sid))
    return IdentityResolver(gallery, emb, reverify_ms=reverify_ms), emb


class TestIdentityResolver:
    def test_matches_enrolled_student(self):
        resolver, emb = make_resolver()
        assert resolver.resolve(1, 0, lambda: "s1") == "s1"

    def test_unknown_face_returns_none(self):
        resolver, emb = make_resolver()
        assert resolver.resolve(1, 0, lambda: "stranger") is None

    def test_embeds_stay_out_of_frame_loop(self):
        """Perf rule 2: 100 frames over 5s → exactly 1 embed for a stable track."""
        resolver, emb = make_resolver(reverify_ms=10_000)
        for i in range(100):
            resolver.resolve(1, i * 50, lambda: "s1")  # 50ms frames = 5s total
        assert emb.calls == 1

    def test_reverify_after_interval(self):
        resolver, emb = make_resolver(reverify_ms=1_000)
        resolver.resolve(1, 0, lambda: "s1")
        resolver.resolve(1, 500, lambda: "s1")   # within interval → no embed
        resolver.resolve(1, 1_500, lambda: "s1") # due → embed
        assert emb.calls == 2

    def test_track_loss_forces_reembed(self):
        resolver, emb = make_resolver()
        resolver.resolve(1, 0, lambda: "s1")
        resolver.drop_track(1)
        resolver.resolve(1, 100, lambda: "s1")
        assert emb.calls == 2

    def test_single_disagreement_does_not_flip(self):
        resolver, emb = make_resolver(reverify_ms=0)  # embed every call
        assert resolver.resolve(1, 0, lambda: "s1") == "s1"
        # one noisy sample says s2 — identity must stick (hysteresis)
        assert resolver.resolve(1, 1, lambda: "s2") == "s1"
        # second consecutive disagreement flips
        assert resolver.resolve(1, 2, lambda: "s2") == "s2"


class TestIoUTracker:
    def test_stable_ids(self):
        t = IoUTracker()
        a = (0.1, 0.1, 0.3, 0.3)
        ids1 = t.update([a])
        ids2 = t.update([(0.11, 0.11, 0.31, 0.31)])  # slight drift
        assert ids1 == ids2

    def test_new_face_new_id(self):
        t = IoUTracker()
        [id_a] = t.update([(0.1, 0.1, 0.3, 0.3)])
        ids = t.update([(0.1, 0.1, 0.3, 0.3), (0.6, 0.6, 0.8, 0.8)])
        assert ids[0] == id_a and ids[1] != id_a

    def test_track_lost_after_max_missed(self):
        t = IoUTracker(max_missed=2)
        [tid] = t.update([(0.1, 0.1, 0.3, 0.3)])
        t.update([])
        t.update([])
        t.update([])  # third miss → dropped
        assert tid in t.lost_last_update


def synthetic_landmarks(
    mouth_open: float = 0.0,
    nose_shift: float = 0.0,
    iris_shift: float = 0.0,
    iris_drop: float = 0.0,
    eye_open: float = 0.04,
) -> np.ndarray:
    """478-point array with just the indices our features read set meaningfully.

    iris_shift moves both irises sideways off eye-center and iris_drop moves
    them down (both are the iris-gaze path); nose_shift turns the head (the
    coarse-gaze path). eye_open is the lid separation — drop it toward 0 to
    simulate a blink, which must not read as a downward look.
    """
    pts = np.full((478, 2), 0.5, dtype=np.float32)
    pts[61] = (0.40, 0.60)   # mouth left
    pts[291] = (0.60, 0.60)  # mouth right
    pts[13] = (0.50, 0.60 - mouth_open / 2)  # mouth top
    pts[14] = (0.50, 0.60 + mouth_open / 2)  # mouth bottom
    pts[33] = (0.35, 0.45)   # left eye outer
    pts[133] = (0.45, 0.45)  # left eye inner
    pts[362] = (0.55, 0.45)  # right eye inner
    pts[263] = (0.65, 0.45)  # right eye outer
    pts[159] = (0.40, 0.45 - eye_open / 2)   # left eye upper lid
    pts[145] = (0.40, 0.45 + eye_open / 2)   # left eye lower lid
    pts[386] = (0.60, 0.45 - eye_open / 2)   # right eye upper lid
    pts[374] = (0.60, 0.45 + eye_open / 2)   # right eye lower lid
    pts[1] = (0.50 + nose_shift, 0.55)  # nose tip
    for idx in (468, 469, 470, 471, 472):  # left iris centered in left eye
        pts[idx] = (0.40 + iris_shift, 0.45 + iris_drop)
    for idx in (473, 474, 475, 476, 477):  # right iris centered in right eye
        pts[idx] = (0.60 + iris_shift, 0.45 + iris_drop)
    return pts


class TestLandmarkFeatures:
    def test_mar_closed_vs_open(self):
        closed = mouth_aspect_ratio(synthetic_landmarks(mouth_open=0.01))
        opened = mouth_aspect_ratio(synthetic_landmarks(mouth_open=0.12))
        assert opened > closed * 5

    def test_coarse_gaze_frontal(self):
        assert coarse_gaze_on_camera(synthetic_landmarks(nose_shift=0.0))

    def test_coarse_gaze_turned_head(self):
        assert not coarse_gaze_on_camera(synthetic_landmarks(nose_shift=0.15))

    def test_coarse_gaze_catches_a_modest_head_turn(self):
        """0.08 of eye width used to pass as attentive; a head turned that far
        is a student reading something beside the screen."""
        assert not coarse_gaze_on_camera(synthetic_landmarks(nose_shift=0.08))

    def test_iris_gaze_frontal(self):
        assert iris_gaze_on_camera(synthetic_landmarks())

    def test_iris_gaze_sideways_glance(self):
        # 0.02 of the 0.10 eye span ⇒ ratio 0.70, past the 0.18 band.
        assert not iris_gaze_on_camera(synthetic_landmarks(iris_shift=0.02))

    def test_iris_gaze_looking_down_at_notes(self):
        """The posture horizontal-only gaze missed entirely: head straight,
        eyes down on notes or a phone."""
        assert not iris_gaze_on_camera(synthetic_landmarks(iris_drop=0.010))

    def test_blink_is_not_a_look_away(self):
        """Closing lids collapse onto the iris; without the eye-open guard
        every blink would read as a downward glance."""
        assert iris_gaze_on_camera(
            synthetic_landmarks(iris_drop=0.010, eye_open=0.004)
        )

    def test_iris_gaze_falls_back_to_coarse_without_iris_landmarks(self):
        unrefined = synthetic_landmarks(nose_shift=0.15)[:468]
        assert not iris_gaze_on_camera(unrefined)
