"""Five-point ArcFace alignment: the fix for group-station recognition.

Unaligned bbox crops scored the same person below the 0.42 match gate on
real enrollment photos; these tests pin the alignment contract so it cannot
silently regress to the loose crop.
"""

import numpy as np
import pytest

from exam_cv.faces.identity import face_chip
from exam_cv.faces.mesh import (
    _ARCFACE_TEMPLATE_112,
    FaceObservation,
    MeshPipeline,
    aligned_crop,
    five_point_landmarks,
)


def _mesh_with(left_eye, right_eye, nose, mouth_l, mouth_r):
    """A 478-point mesh (normalized) with only the alignment points populated."""
    landmarks = np.zeros((478, 2), dtype=np.float32)
    landmarks[468:473] = left_eye
    landmarks[473:478] = right_eye
    landmarks[1] = nose
    landmarks[61] = mouth_l
    landmarks[291] = mouth_r
    return landmarks


class TestFivePoints:
    def test_needs_iris_block(self):
        assert five_point_landmarks(np.zeros((468, 2)), 640, 480) is None
        assert five_point_landmarks(None, 640, 480) is None

    def test_orders_eyes_and_mouth_by_image_x(self):
        # Landmarks handed over "swapped" (mirrored webcam) still map to the
        # template's left/right slots by image position.
        lm = _mesh_with((0.6, 0.4), (0.4, 0.4), (0.5, 0.5), (0.6, 0.6), (0.4, 0.6))
        pts = five_point_landmarks(lm, 100, 100)
        assert pts.shape == (5, 2)
        assert pts[0][0] < pts[1][0]  # left eye is left of right eye
        assert pts[3][0] < pts[4][0]  # left mouth corner is left of right


class TestAlignedCrop:
    def test_returns_112_chip_with_eyes_on_template(self):
        # Put a bright dot at the left iris; after warping it must land on
        # the template's left-eye coordinate.
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        lm = _mesh_with((0.40, 0.40), (0.55, 0.40), (0.475, 0.50), (0.42, 0.60), (0.53, 0.60))
        ex, ey = int(0.40 * 640), int(0.40 * 480)
        image[ey - 2:ey + 3, ex - 2:ex + 3] = 255
        obs = FaceObservation(track_id=0, bbox=(0.3, 0.3, 0.7, 0.7), landmarks=lm)
        chip = aligned_crop(image, obs)
        assert chip.shape == (112, 112, 3)
        ys, xs = np.nonzero(chip[:, :, 0])
        assert abs(xs.mean() - _ARCFACE_TEMPLATE_112[0][0]) < 3
        assert abs(ys.mean() - _ARCFACE_TEMPLATE_112[0][1]) < 3

    def test_falls_back_without_landmarks(self):
        class Obs:
            track_id = 0
            bbox = (0.1, 0.1, 0.5, 0.5)

        image = np.zeros((40, 40, 3), dtype=np.uint8)
        assert aligned_crop(image, Obs()) is None


class TestFaceChip:
    def test_uses_bbox_crop_when_mesh_is_a_stub(self):
        class StubMesh:
            def crop(self, image, obs, pad=0.2):
                return np.full((4, 4), 7.0, dtype=np.float32)

        class Obs:
            bbox = (0.1, 0.1, 0.5, 0.5)

        chip = face_chip(StubMesh(), np.zeros((8, 8, 3), dtype=np.float32), Obs())
        assert chip.shape == (4, 4) and float(chip[0, 0]) == 7.0

    def test_prefers_alignment_when_the_mesh_offers_it(self):
        class StubMesh:
            def aligned_crop(self, image, obs, size=112):
                return aligned_crop(image, obs, size)

            def crop(self, image, obs, pad=0.2):  # pragma: no cover - must not run
                raise AssertionError("aligned path should have been taken")

        lm = _mesh_with((0.40, 0.40), (0.55, 0.40), (0.475, 0.50), (0.42, 0.60), (0.53, 0.60))
        obs = FaceObservation(track_id=0, bbox=(0.3, 0.3, 0.7, 0.7), landmarks=lm)
        chip = face_chip(StubMesh(), np.zeros((480, 640, 3), dtype=np.uint8), obs)
        assert chip.shape == (112, 112, 3)


    def test_real_detector_always_offers_alignment(self):
        # The production mesh must expose the method face_chip looks for,
        # otherwise every path would silently regress to the bbox crop.
        assert callable(getattr(MeshPipeline, "aligned_crop", None))
