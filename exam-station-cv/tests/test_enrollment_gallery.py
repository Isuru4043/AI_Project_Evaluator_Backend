"""Enrollment-photo gallery building (group identity for composite recordings).

The gallery is what lets the examiner see WHO answered in a group viva, so the
rule that matters here is the HITL invariant: an ambiguous reference photo must
leave the student unenrolled (→ unknown at analysis time) rather than enrolling
a face we aren't sure about.
"""

import numpy as np

from exam_cv.faces.identity import (
    EnrollmentGallery,
    assign_embeddings_one_to_one,
    build_gallery_from_photos,
)


class FakeEmbedder:
    """Embedding keyed by the crop's marker value (see FakeMesh.crop)."""

    def __init__(self):
        self.calls = 0

    def embed(self, crop) -> np.ndarray:
        self.calls += 1
        key = float(np.asarray(crop).flat[0])
        rng = np.random.default_rng(int(key) % 2**32)
        v = rng.normal(size=64)
        return v / np.linalg.norm(v)


class FakeObs:
    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox


class FakeMesh:
    """Detector stub: face count per image is taken from the image's marker."""

    def __init__(self, faces_per_image):
        self._faces = faces_per_image
        self.closed = False

    def process_frame(self, image):
        n = self._faces[float(image.flat[0])]
        return [FakeObs(i, (0.1 * i, 0.1, 0.1 * i + 0.05, 0.5)) for i in range(n)]

    def crop(self, image, obs, pad=0.2):
        # Marker survives the crop so the embedder returns a per-photo vector.
        return np.full((4, 4), image.flat[0], dtype=np.float32)

    def close(self):
        self.closed = True


def photo(marker: float) -> np.ndarray:
    return np.full((8, 8, 3), marker, dtype=np.float32)


class TestBuildGalleryFromPhotos:
    def test_enrolls_one_embedding_per_clear_photo(self):
        photos = {"s1": photo(1.0), "s2": photo(2.0)}
        mesh = FakeMesh({1.0: 1, 2.0: 1})
        embedder = FakeEmbedder()

        gallery, skipped = build_gallery_from_photos(photos, mesh, embedder)

        assert gallery.enrolled_ids() == {"s1", "s2"}
        assert skipped == []
        assert embedder.calls == 2  # exactly one embed per student

    def test_photo_with_no_face_is_skipped_not_guessed(self):
        photos = {"s1": photo(1.0), "s2": photo(2.0)}
        mesh = FakeMesh({1.0: 1, 2.0: 0})  # s2's photo has no detectable face

        gallery, skipped = build_gallery_from_photos(photos, mesh, FakeEmbedder())

        assert gallery.enrolled_ids() == {"s1"}
        assert skipped == ["s2"]

    def test_photo_with_multiple_faces_is_skipped(self):
        photos = {"s1": photo(1.0)}
        mesh = FakeMesh({1.0: 2})  # ambiguous: which face is s1?

        gallery, skipped = build_gallery_from_photos(photos, mesh, FakeEmbedder())

        assert gallery.enrolled_ids() == set()
        assert skipped == ["s1"]

    def test_enrolled_photo_matches_the_same_face_later(self):
        """A gallery built from a photo must recognise that face in the video."""
        photos = {"s1": photo(1.0), "s2": photo(2.0)}
        mesh = FakeMesh({1.0: 1, 2.0: 1})
        embedder = FakeEmbedder()

        gallery, _ = build_gallery_from_photos(photos, mesh, embedder)

        # The same crop marker that enrolled s2 must resolve back to s2.
        assert gallery.match(embedder.embed(np.full((4, 4), 2.0))) == "s2"
        # An unenrolled face resolves to None (unknown), never a roster guess.
        assert gallery.match(embedder.embed(np.full((4, 4), 9.0))) is None

    def test_empty_input_yields_empty_gallery(self):
        gallery, skipped = build_gallery_from_photos({}, FakeMesh({}), FakeEmbedder())

        assert gallery.enrolled_ids() == set()
        assert skipped == []

    def test_enrolls_multiple_clear_samples_for_one_student(self):
        samples = [photo(1.0), photo(2.0), photo(3.0)]
        mesh = FakeMesh({1.0: 1, 2.0: 1, 3.0: 1})
        embedder = FakeEmbedder()

        gallery, skipped = build_gallery_from_photos(
            {'s1': samples}, mesh, embedder,
        )

        assert skipped == []
        assert gallery.enrolled_ids() == {'s1'}
        assert embedder.calls == 3
        assert gallery.match(embedder.embed(np.full((4, 4), 3.0))) == 's1'


class TestGlobalGroupAssignment:
    @staticmethod
    def gallery(count):
        gallery = EnrollmentGallery()
        for index in range(count):
            vector = np.zeros(count, dtype=np.float32)
            vector[index] = 1
            gallery.enroll(f's{index + 1}', vector)
        return gallery

    def test_three_people_are_assigned_once_even_when_face_order_changes(self):
        gallery = self.gallery(3)
        faces = [np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])]

        result = assign_embeddings_one_to_one(faces, gallery)

        assert [item['student_id'] for item in result] == ['s3', 's1', 's2']
        assert len({item['student_id'] for item in result}) == 3

    def test_four_people_are_all_assigned_one_to_one(self):
        gallery = self.gallery(4)
        faces = [
            np.array([0, 1, 0, 0]), np.array([0, 0, 0, 1]),
            np.array([1, 0, 0, 0]), np.array([0, 0, 1, 0]),
        ]

        result = assign_embeddings_one_to_one(faces, gallery)

        assert {item['student_id'] for item in result} == {'s1', 's2', 's3', 's4'}
        assert len({item['student_id'] for item in result}) == 4

    def test_close_identity_margin_remains_unknown(self):
        gallery = EnrollmentGallery()
        gallery.enroll('s1', np.array([1.0, 0.0]))
        gallery.enroll('s2', np.array([0.999, 0.045]))

        result = assign_embeddings_one_to_one(
            [np.array([1.0, 0.02])], gallery, threshold=.42, min_margin=.05,
        )

        assert result[0]['student_id'] is None
