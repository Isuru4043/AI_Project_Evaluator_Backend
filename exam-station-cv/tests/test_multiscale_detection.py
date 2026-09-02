"""Tiled detection for small faces on a shared station camera."""

import numpy as np

from exam_cv.faces.mesh import (
    FaceObservation,
    _merge_observations,
    _tile_offsets,
    detect_faces_multiscale,
)


class SmallFaceMesh:
    """Sees a face only when it fills >= 10% of the input's longer side, as
    the real detector does. ``faces`` are (cx, cy, width) in FRAME pixels;
    boxes are sized by the face, not the tile, like a real detector."""

    def __init__(self, faces, frame_w, frame_h):
        self.faces, self.frame_w, self.frame_h = faces, frame_w, frame_h
        self.calls = 0

    def process_frame(self, image):
        self.calls += 1
        h, w = image.shape[:2]
        x0, y0 = int(image[0, 0, 0]), int(image[0, 0, 1])  # exact tile offset stamp
        out = []
        for index, (cx, cy, fw) in enumerate(self.faces):
            inside = x0 <= cx < x0 + w and y0 <= cy < y0 + h
            if inside and fw / max(w, h) >= 0.10:
                nx, ny = (cx - x0) / w, (cy - y0) / h
                hx, hy = (fw / 2) / w, (fw / 2) / h
                out.append(FaceObservation(
                    track_id=index,
                    bbox=(nx - hx, ny - hy, nx + hx, ny + hy),
                    landmarks=_landmarks_box(nx - hx, ny - hy, nx + hx, ny + hy),
                ))
        return out


def _landmarks_box(x0, y0, x1, y1):
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[:, 0] = np.linspace(x0, x1, 478)
    pts[:, 1] = np.linspace(y0, y1, 478)
    return pts


def _stamped_frame(w, h):
    """Frame whose pixel (0,0) of any crop encodes that crop's exact offset."""
    frame = np.zeros((h, w, 3), dtype=np.float32)
    frame[:, :, 0] = np.arange(w)[None, :]
    frame[:, :, 1] = np.arange(h)[:, None]
    return frame


class TestTiling:
    def test_offsets_cover_the_range_and_end_flush(self):
        assert _tile_offsets(1280, 640, 480) == [0, 480, 640]
        assert _tile_offsets(500, 640, 480) == [0]

    def test_full_frame_pass_is_enough_for_large_faces(self):
        mesh = SmallFaceMesh([(200, 360, 200), (640, 360, 200), (1080, 360, 200)], 1280, 720)
        found = detect_faces_multiscale(mesh, _stamped_frame(1280, 720), expected_faces=3)
        assert len(found) == 3
        assert mesh.calls == 1  # no tiles were needed

    def test_small_faces_are_found_on_tiles_and_mapped_back(self):
        # 90px faces: 7% of the frame, invisible to the full-frame pass.
        faces = [(200, 360, 90), (640, 360, 90), (1080, 360, 90)]
        mesh = SmallFaceMesh(faces, 1280, 720)
        found = detect_faces_multiscale(mesh, _stamped_frame(1280, 720), expected_faces=3)
        assert len(found) == 3
        centres = sorted(((o.bbox[0] + o.bbox[2]) / 2 * 1280, (o.bbox[1] + o.bbox[3]) / 2 * 720) for o in found)
        for (cx, cy), (fx, fy, _) in zip(centres, faces):
            assert abs(cx - fx) < 12 and abs(cy - fy) < 12
        # landmarks are in frame coordinates too
        assert all(0.0 <= o.landmarks[:, 0].min() and o.landmarks[:, 0].max() <= 1.0 for o in found)
        assert [o.track_id for o in found] == [0, 1, 2]

    def test_stops_early_without_expected_count_only_after_all_levels(self):
        mesh = SmallFaceMesh([(640, 360, 300)], 1280, 720)
        found = detect_faces_multiscale(mesh, _stamped_frame(1280, 720), expected_faces=None)
        assert len(found) == 1  # duplicates from overlapping tiles were merged
        assert mesh.calls > 1


class TestMerge:
    def test_largest_box_wins_and_ids_run_left_to_right(self):
        big = FaceObservation(track_id=5, bbox=(0.5, 0.5, 0.7, 0.7), landmarks=np.zeros((478, 2)))
        clipped = FaceObservation(track_id=9, bbox=(0.5, 0.5, 0.62, 0.7), landmarks=np.zeros((478, 2)))
        other = FaceObservation(track_id=7, bbox=(0.1, 0.1, 0.2, 0.2), landmarks=np.zeros((478, 2)))
        merged = _merge_observations([clipped, big, other], min_iou=0.3)
        assert [o.bbox for o in merged] == [other.bbox, big.bbox]
        assert [o.track_id for o in merged] == [0, 1]
