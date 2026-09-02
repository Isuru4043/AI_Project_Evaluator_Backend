"""Face detection/tracking + landmark features.

Performance rules enforced here:
- MediaPipe Face Mesh is the SOLE detect+track stack (rule 1).
- The frame-rate mesh runs UNREFINED; the refined (iris) mesh exists as a
  separate instance the runner calls only on the low-rate tick, and only on
  speaker candidates (rules 3–5).

Landmark-derived features (mouth aspect ratio, coarse/iris gaze) are pure
numpy over the landmark array, so they unit-test with synthetic landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# MediaPipe FaceMesh canonical landmark indices
_MOUTH_TOP, _MOUTH_BOTTOM = 13, 14
_MOUTH_LEFT, _MOUTH_RIGHT = 61, 291
_LEFT_EYE_OUTER, _LEFT_EYE_INNER = 33, 133
_RIGHT_EYE_INNER, _RIGHT_EYE_OUTER = 362, 263
_LEFT_EYE_TOP, _LEFT_EYE_BOTTOM = 159, 145
_RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM = 386, 374
_LEFT_IRIS = [468, 469, 470, 471, 472]   # refined mesh only
_RIGHT_IRIS = [473, 474, 475, 476, 477]  # refined mesh only
_NOSE_TIP = 1

# Gaze tuning. These decide how far off-axis counts as "not on the task", and
# they are the sensitivity dial for every look-away flag downstream — the
# analyzers only time spells the geometry has already called off-camera.
#
# Horizontal offsets are fractions of the eye/iris span where 0 = dead centre
# and 0.5 = pinned to a corner, so 0.30 (the original value) meant a student
# could stare at the far edge of the desk and still read as on-camera.
GAZE_COARSE_MAX_OFFSET = 0.22    # nose lateral offset / eye width
GAZE_IRIS_MAX_OFFSET = 0.18      # |iris ratio − 0.5|, horizontal
GAZE_IRIS_MAX_VERTICAL = 0.22    # |iris ratio − 0.5|, vertical (notes/phone)
# Below this lid separation the eye is blinking or squinting and the vertical
# iris position is meaningless, so only the horizontal test is trusted.
GAZE_MIN_EYE_OPEN = 0.15         # lid separation / eye width


@dataclass
class FaceObservation:
    track_id: int
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 (normalized)
    landmarks: np.ndarray  # (N, 2) normalized xy


def mouth_aspect_ratio(landmarks: np.ndarray) -> float:
    """Vertical mouth opening / horizontal mouth width. Higher = more open."""
    v = np.linalg.norm(landmarks[_MOUTH_TOP] - landmarks[_MOUTH_BOTTOM])
    h = np.linalg.norm(landmarks[_MOUTH_LEFT] - landmarks[_MOUTH_RIGHT])
    return float(v / (h + 1e-9))


def coarse_gaze_on_camera(
    landmarks: np.ndarray, max_offset: float = GAZE_COARSE_MAX_OFFSET
) -> bool:
    """Cheap gaze proxy for NON-speaker faces (no iris needed): nose position
    relative to the eye line — a turned head reads as off-camera."""
    eye_l = landmarks[_LEFT_EYE_OUTER]
    eye_r = landmarks[_RIGHT_EYE_OUTER]
    nose = landmarks[_NOSE_TIP]
    eye_width = np.linalg.norm(eye_r - eye_l) + 1e-9
    mid = (eye_l + eye_r) / 2.0
    lateral = abs(nose[0] - mid[0]) / eye_width
    return lateral <= max_offset


def iris_gaze_on_camera(
    landmarks: np.ndarray,
    max_offset: float = GAZE_IRIS_MAX_OFFSET,
    max_vertical_offset: float = GAZE_IRIS_MAX_VERTICAL,
) -> bool:
    """Iris-based gaze (refined mesh only): iris center inside the central
    band of the eye ⇒ looking toward the camera/task.

    Both axes are checked. Horizontal alone missed the most common off-task
    posture in an oral exam — head still, eyes down on notes or a phone —
    because reading downward barely moves the iris sideways at all.

    The vertical test is skipped for an eye that is closing: while blinking the
    lids collapse onto the iris and the ratio is noise, which would otherwise
    make every blink read as a look-away.
    """
    if landmarks.shape[0] <= _RIGHT_IRIS[-1]:
        # Unrefined mesh — fall back to the coarse proxy.
        return coarse_gaze_on_camera(landmarks)

    def _axis_ratio(iris_idx, a_idx, b_idx) -> float:
        """Iris centre projected onto the a→b axis. 0.5 = centred."""
        iris = landmarks[iris_idx].mean(axis=0)
        a, b = landmarks[a_idx], landmarks[b_idx]
        span = np.linalg.norm(b - a) + 1e-9
        return float(np.dot(iris - a, b - a) / (span * span))

    def _eye_open(top_idx, bottom_idx, outer_idx, inner_idx) -> float:
        lids = np.linalg.norm(landmarks[bottom_idx] - landmarks[top_idx])
        width = np.linalg.norm(landmarks[outer_idx] - landmarks[inner_idx]) + 1e-9
        return float(lids / width)

    horizontal = (
        _axis_ratio(_LEFT_IRIS, _LEFT_EYE_OUTER, _LEFT_EYE_INNER),
        _axis_ratio(_RIGHT_IRIS, _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER),
    )
    if any(abs(ratio - 0.5) > max_offset for ratio in horizontal):
        return False

    for iris, top, bottom, outer, inner in (
        (_LEFT_IRIS, _LEFT_EYE_TOP, _LEFT_EYE_BOTTOM, _LEFT_EYE_OUTER, _LEFT_EYE_INNER),
        (_RIGHT_IRIS, _RIGHT_EYE_TOP, _RIGHT_EYE_BOTTOM, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER),
    ):
        if _eye_open(top, bottom, outer, inner) < GAZE_MIN_EYE_OPEN:
            continue  # blinking — vertical position tells us nothing
        if abs(_axis_ratio(iris, top, bottom) - 0.5) > max_vertical_offset:
            return False

    return True


class IoUTracker:
    """Assigns stable track_ids to per-frame detections by bbox overlap."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, tuple[tuple, int]] = {}  # id -> (bbox, missed)

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter + 1e-9)

    def update(self, bboxes: list[tuple]) -> list[int]:
        """Match detections to tracks; returns a track_id per bbox (in order)."""
        assigned: list[int] = []
        used: set[int] = set()
        for bbox in bboxes:
            best_id, best_iou = None, self.iou_threshold
            for tid, (tb, _) in self._tracks.items():
                if tid in used:
                    continue
                iou = self._iou(bbox, tb)
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            self._tracks[best_id] = (bbox, 0)
            used.add(best_id)
            assigned.append(best_id)

        lost = []
        for tid in list(self._tracks):
            if tid not in used:
                bbox, missed = self._tracks[tid]
                if missed + 1 > self.max_missed:
                    del self._tracks[tid]
                    lost.append(tid)
                else:
                    self._tracks[tid] = (bbox, missed + 1)
        self.lost_last_update = lost
        return assigned


class MeshPipeline:
    """MediaPipe Tasks FaceLandmarker wrapper (cv2/mediapipe lazy).

    The legacy ``mp.solutions.face_mesh`` API was removed in MediaPipe 0.10.x;
    the Tasks FaceLandmarker replaces it. Its single 478-landmark model already
    includes the iris points, so the old "unrefined at frame rate / refined
    iris on tick" split collapses into ONE model. We still expose
    process_frame / process_tick so the two-rate loop is unchanged, and cache
    the per-frame result so calling both on the same frame runs inference once
    (rule 1: one detector).
    """

    def __init__(
        self,
        max_faces: int = 5,
        model_path=None,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
    ):
        import mediapipe as mp  # lazy
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        from .model_assets import face_landmarker_path

        self._mp = mp
        path = str(model_path or face_landmarker_path())
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            num_faces=max_faces,
            running_mode=mp_vision.RunningMode.IMAGE,  # stateless; we track ourselves
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self.tracker = IoUTracker()
        self._cache_key = None
        self._cache_val: list[FaceObservation] = []

    def _run(self, image_bgr: np.ndarray) -> list[FaceObservation]:
        # One-frame memoization: if the same array object is passed for both
        # the attribution and behavioral paths, detect only once.
        if id(image_bgr) == self._cache_key:
            return self._cache_val

        import cv2  # lazy

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        bboxes = []
        landmark_sets = []
        for fl in result.face_landmarks:
            pts = np.array([(lm.x, lm.y) for lm in fl], dtype=np.float32)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            bboxes.append((float(x0), float(y0), float(x1), float(y1)))
            landmark_sets.append(pts)

        track_ids = self.tracker.update(bboxes)
        obs = [
            FaceObservation(track_id=tid, bbox=bbox, landmarks=pts)
            for tid, bbox, pts in zip(track_ids, bboxes, landmark_sets)
        ]
        self._cache_key, self._cache_val = id(image_bgr), obs
        return obs

    def process_frame(self, image_bgr: np.ndarray) -> list[FaceObservation]:
        """Frame-rate pass (attribution path)."""
        return self._run(image_bgr)

    def process_tick(self, image_bgr: np.ndarray) -> list[FaceObservation]:
        """Low-rate tick (behavioral path). Iris landmarks come from the same
        478-point model, so no second inference is needed."""
        return self._run(image_bgr)

    def crop(self, image_bgr: np.ndarray, obs: FaceObservation, pad: float = 0.2):
        """Face crop for ArcFace embedding — the ONLY detection feed it gets."""
        h, w = image_bgr.shape[:2]
        x0, y0, x1, y1 = obs.bbox
        bw, bh = x1 - x0, y1 - y0
        x0 = max(0, int((x0 - pad * bw) * w))
        y0 = max(0, int((y0 - pad * bh) * h))
        x1 = min(w, int((x1 + pad * bw) * w))
        y1 = min(h, int((y1 + pad * bh) * h))
        return image_bgr[y0:y1, x0:x1]

    def aligned_crop(self, image_bgr: np.ndarray, obs: FaceObservation, size: int = 112):
        """ArcFace-aligned 112x112 chip (see module-level ``aligned_crop``).

        Exposed as a method so ``identity.face_chip`` aligns for every real
        detector while numpy-only test fakes, which have no landmark
        geometry to align, keep their plain ``crop``.
        """
        return aligned_crop(image_bgr, obs, size)

    def close(self) -> None:
        self._landmarker.close()


# ArcFace's canonical 112x112 five-point template (insightface ``arcface_dst``):
# image-left eye, image-right eye, nose tip, image-left mouth corner,
# image-right mouth corner.
_ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)
_ALIGN_LEFT_IRIS = slice(468, 473)
_ALIGN_RIGHT_IRIS = slice(473, 478)
_ALIGN_NOSE_TIP = 1
_ALIGN_MOUTH_CORNERS = (61, 291)


def five_point_landmarks(
    landmarks: Optional[np.ndarray], width: int, height: int
) -> Optional[np.ndarray]:
    """ArcFace's five alignment points, in pixels, from a 478-point mesh.

    Returns None when the mesh lacks the iris block (unrefined 468-point
    output, or a detector stub) so the caller can fall back to a bbox crop.
    Eyes and mouth corners are ordered by image x, so a mirrored webcam feed
    still maps onto the template correctly.
    """
    if landmarks is None or len(landmarks) < 478:
        return None
    eyes = sorted(
        (landmarks[_ALIGN_LEFT_IRIS].mean(axis=0), landmarks[_ALIGN_RIGHT_IRIS].mean(axis=0)),
        key=lambda p: float(p[0]),
    )
    mouth = sorted(
        (landmarks[_ALIGN_MOUTH_CORNERS[0]], landmarks[_ALIGN_MOUTH_CORNERS[1]]),
        key=lambda p: float(p[0]),
    )
    points = np.array(
        [eyes[0], eyes[1], landmarks[_ALIGN_NOSE_TIP], mouth[0], mouth[1]],
        dtype=np.float32,
    )
    points[:, 0] *= width
    points[:, 1] *= height
    return points


def aligned_crop(
    image_bgr: np.ndarray, obs: FaceObservation, size: int = 112
) -> Optional[np.ndarray]:
    """Eye/nose/mouth-aligned face chip - the input ArcFace was trained on.

    A loose bbox crop leaves head roll, scale and framing in the image, which
    ArcFace does not tolerate. Measured on real enrollment photos, the same
    person's unaligned crops scored 0.13-0.34 cosine (below the 0.42 match
    gate) while aligned chips scored 0.49-0.85, and two different people fell
    from 0.45 (a false near-match) to 0.20. This is the standard InsightFace
    ``norm_crop`` warp, done with cv2 alone so it needs no scikit-image.

    Returns None when the observation carries no iris landmarks; callers then
    use ``MeshPipeline.crop``.
    """
    h, w = image_bgr.shape[:2]
    source = five_point_landmarks(getattr(obs, "landmarks", None), w, h)
    if source is None:
        return None
    import cv2

    template = _ARCFACE_TEMPLATE_112 * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(source, template, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(image_bgr, matrix, (size, size), borderValue=0)


def _box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    area = lambda r: max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])  # noqa: E731
    return inter / (area(a) + area(b) - inter + 1e-9)


def _tile_offsets(length: int, tile: int, stride: int) -> list[int]:
    """Window starts covering [0, length) with the last one flush to the end."""
    if tile >= length:
        return [0]
    offsets = list(range(0, length - tile, stride))
    offsets.append(length - tile)
    return sorted(set(offsets))


def detect_faces_multiscale(
    mesh,
    image_bgr: np.ndarray,
    expected_faces: Optional[int] = None,
    max_levels: int = 3,
    min_iou_same_face: float = 0.3,
) -> list[FaceObservation]:
    """Every face in a still frame, including the small ones one pass misses.

    FaceLandmarker's detector shrinks the whole frame to 128px before looking
    for faces, so a face narrower than roughly 10% of the frame is invisible
    to it. Measured on a 1280x720 station frame with three people: 160px
    faces were all found, at 120px one of three was lost, at 90px none were
    found. That is the "not detected although everyone is in view" case of a
    shared exam-station camera.

    This runs the SAME detector (rule 1: one detector) on the full frame and
    then, only while fewer than ``expected_faces`` have been found, on
    progressively smaller overlapping tiles where each face fills a larger
    share of the input. Landmarks come back normalized to the tile and are
    mapped into frame coordinates, so callers crop from the original pixels.
    Duplicates across overlapping tiles are merged by box overlap, keeping
    the largest (least edge-clipped) observation.

    For still-frame binding only: the tracker inside ``mesh`` sees tile
    coordinates, so never call this from the video loop.
    """
    h, w = image_bgr.shape[:2]
    found: list[FaceObservation] = list(mesh.process_frame(image_bgr))
    keep_alive = []  # tile arrays must outlive the loop: the mesh memoizes on id()

    def enough() -> bool:
        return expected_faces is not None and len(found) >= int(expected_faces)

    for level in range(1, max(1, int(max_levels))):
        if enough():
            break
        cols = level + 1
        side = int(np.ceil(w / cols))
        if h <= side * 1.25:
            tile_w, tile_h = side, h
        else:
            tile_w = tile_h = side
        stride_x = max(1, int(tile_w * 0.75))
        stride_y = max(1, int(tile_h * 0.75))
        for y0 in _tile_offsets(h, tile_h, stride_y):
            for x0 in _tile_offsets(w, tile_w, stride_x):
                tile = np.ascontiguousarray(image_bgr[y0:y0 + tile_h, x0:x0 + tile_w])
                keep_alive.append(tile)
                th, tw = tile.shape[:2]
                for obs in mesh.process_frame(tile):
                    pts = np.asarray(obs.landmarks, dtype=np.float32).copy()
                    pts[:, 0] = (x0 + pts[:, 0] * tw) / w
                    pts[:, 1] = (y0 + pts[:, 1] * th) / h
                    bx0, by0 = pts.min(axis=0)
                    bx1, by1 = pts.max(axis=0)
                    found.append(FaceObservation(
                        track_id=-1,
                        bbox=(float(bx0), float(by0), float(bx1), float(by1)),
                        landmarks=pts,
                    ))
    return _merge_observations(found, min_iou_same_face)


def _merge_observations(
    observations: list[FaceObservation], min_iou: float
) -> list[FaceObservation]:
    """Drop duplicates of the same face; largest box wins; ids left-to-right."""
    area = lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1])  # noqa: E731
    merged: list[FaceObservation] = []
    for obs in sorted(observations, key=area, reverse=True):
        if any(_box_iou(obs.bbox, kept.bbox) >= min_iou for kept in merged):
            continue
        merged.append(obs)
    merged.sort(key=lambda o: o.bbox[0])
    return [
        FaceObservation(track_id=index, bbox=o.bbox, landmarks=o.landmarks)
        for index, o in enumerate(merged)
    ]
