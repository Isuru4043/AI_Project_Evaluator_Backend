"""Face identity: enrollment gallery + track→student resolution.

Performance rule 2 is structural here: IdentityResolver only calls the
embedder at (a) enrollment, (b) periodic re-verify per track, (c) track
loss/reacquire. The frame loop never embeds.

The embedder is injected (FaceEmbedder protocol) so tests use a fake;
ArcFaceEmbedder loads only InsightFace's recognition model and embeds
mesh-provided crops — it never runs its own detection (rule 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


class FaceEmbedder(Protocol):
    def embed(self, face_crop_bgr: np.ndarray) -> np.ndarray: ...


class ArcFaceEmbedder:
    """ArcFace recognition model on aligned crops (no internal detection)."""

    def __init__(self, model_name: str = "buffalo_l"):
        import cv2  # lazy
        from insightface.model_zoo import get_model  # lazy
        from insightface.utils import ensure_available  # lazy

        self._cv2 = cv2
        root = ensure_available("models", model_name)
        # recognition model file inside the pack (w600k_r50 for buffalo_l)
        self._model = get_model(f"{root}/w600k_r50.onnx")
        self._model.prepare(ctx_id=-1)  # CPU

    def embed(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        img = self._cv2.resize(face_crop_bgr, (112, 112))
        emb = self._model.get_feat(img).flatten()
        return emb / (np.linalg.norm(emb) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


class EnrollmentGallery:
    """student_id → enrolled embeddings (N snapshots at session start)."""

    def __init__(self):
        self._gallery: dict[str, list[np.ndarray]] = {}

    def enroll(self, student_id: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        self._gallery.setdefault(student_id, []).append(vector)

    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Return the best similarity for every enrolled student.

        Keeping the complete score row lets the room binder solve one global
        assignment. Matching each crop independently can assign the same
        student to two faces and is especially unreliable in 3/4-person
        groups where several candidates may have close scores.
        """
        return {
            sid: max(cosine(embedding, enrolled) for enrolled in embeddings)
            for sid, embeddings in self._gallery.items()
        }

    def match_with_score(
        self,
        embedding: np.ndarray,
        threshold: float = 0.35,
    ) -> tuple[Optional[str], float]:
        """Return the best accepted identity and its cosine similarity.

        Detection confidence and identity confidence are different signals.
        Callers that surface a confidence value must report the ArcFace
        similarity rather than a hard-coded success value.
        """
        best_id, best_sim = None, -1.0
        for sid, sim in self.scores(embedding).items():
            if sim > best_sim:
                best_id, best_sim = sid, sim
        return (best_id if best_sim > threshold else None), best_sim

    def match(self, embedding: np.ndarray, threshold: float = 0.35) -> Optional[str]:
        """Best cosine match above threshold, else None (unknown face)."""
        return self.match_with_score(embedding, threshold)[0]

    def enrolled_ids(self) -> set[str]:
        return set(self._gallery.keys())

    def serializable_embeddings(self) -> dict[str, list[list[float]]]:
        return {
            sid: [embedding.astype(float).tolist() for embedding in embeddings]
            for sid, embeddings in self._gallery.items()
        }


def assign_embeddings_one_to_one(
    embeddings: list[np.ndarray],
    gallery: EnrollmentGallery,
    threshold: float = 0.42,
    min_margin: float = 0.05,
) -> list[dict]:
    """Globally assign room faces to distinct roster identities.

    The small physical-room roster (normally 1-4 people) makes an exhaustive
    assignment both clearer and safer than a greedy nearest-neighbour pass.
    Each accepted match must clear an absolute similarity threshold and a
    margin over that face's next-best identity. Ambiguous faces remain unknown.
    """
    score_rows = [gallery.scores(embedding) for embedding in embeddings]
    best_score = float('-inf')
    best_assignment: list[Optional[str]] = [None] * len(score_rows)

    def search(index: int, used: set[str], assignment: list[Optional[str]], total: float):
        nonlocal best_score, best_assignment
        if index == len(score_rows):
            if total > best_score:
                best_score = total
                best_assignment = assignment.copy()
            return

        # Unknown has a threshold-sized baseline. This means a candidate must
        # actually clear the calibrated threshold to improve the assignment.
        search(index + 1, used, [*assignment, None], total + threshold)
        for student_id, score in score_rows[index].items():
            if student_id in used or score < threshold:
                continue
            search(
                index + 1,
                used | {student_id},
                [*assignment, student_id],
                total + score,
            )

    search(0, set(), [], 0.0)

    result: list[dict] = []
    for row, student_id in zip(score_rows, best_assignment):
        confidence = float(row.get(student_id, -1.0)) if student_id else max(
            row.values(), default=-1.0,
        )
        alternatives = [
            value for candidate, value in row.items() if candidate != student_id
        ]
        margin = confidence - max(alternatives, default=-1.0)
        if student_id and margin < min_margin:
            student_id = None
        result.append({
            'student_id': student_id,
            'confidence': max(0.0, confidence),
            'identity_margin': round(float(margin), 4),
        })
    return result


def face_chip(mesh, image: np.ndarray, obs) -> np.ndarray:
    """The crop ArcFace embeds: landmark-aligned when the mesh allows it.

    Enrollment photos and live room frames MUST go through this same function.
    Comparing an aligned probe against an unaligned gallery (or vice versa) is
    worse than either alone, which is why the alignment lives here rather than
    at one call site.
    """
    align = getattr(mesh, "aligned_crop", None)
    chip = align(image, obs) if align is not None else None
    return chip if chip is not None else mesh.crop(image, obs)


def build_gallery_from_photos(
    photos: dict[str, np.ndarray | list[np.ndarray]],
    mesh,
    embedder: FaceEmbedder,
) -> tuple["EnrollmentGallery", list[str]]:
    """Enroll students from their reference face photos.

    ``photos`` maps student_id to one decoded image or a list of guided
    enrollment samples. ``mesh`` must be a
    throwaway MeshPipeline (its tracker state is mutated here and would
    otherwise pollute the video pass); it is the sole detector, so ArcFace
    still never runs its own detection (rule 1).

    Returns (gallery, skipped_ids). A photo is skipped when it shows no face
    or more than one — an ambiguous reference is left unenrolled so the
    student resolves to unknown rather than being guessed (HITL invariant).
    """
    gallery = EnrollmentGallery()
    skipped: list[str] = []
    for student_id, value in photos.items():
        samples = value if isinstance(value, (list, tuple)) else [value]
        enrolled = 0
        for image in samples:
            observations = mesh.process_frame(image)
            if len(observations) != 1:
                continue
            crop = face_chip(mesh, image, observations[0])
            if crop.size == 0:
                continue
            gallery.enroll(student_id, embedder.embed(crop))
            enrolled += 1
        if not enrolled:
            skipped.append(student_id)
    return gallery, skipped


@dataclass
class _TrackIdentity:
    student_id: Optional[str]  # None = unknown face
    last_verified_ms: int
    hits: int = 2  # hysteresis: two consecutive disagreements flip identity


class IdentityResolver:
    """Sticky track→student mapping with periodic re-verify.

    Identity sticks to a track (hysteresis); embeddings happen only on new
    tracks and every reverify_ms per track. Callers pass a crop provider so
    this class decides WHEN to embed, never the frame loop.
    """

    def __init__(
        self,
        gallery: EnrollmentGallery,
        embedder: FaceEmbedder,
        reverify_ms: int = 12000,
        match_threshold: float = 0.35,
    ):
        self.gallery = gallery
        self.embedder = embedder
        self.reverify_ms = reverify_ms
        self.match_threshold = match_threshold
        self._tracks: dict[int, _TrackIdentity] = {}
        self.embed_calls = 0  # perf tests assert this stays out of the frame loop

    def resolve(
        self,
        track_id: int,
        t_ms: int,
        crop_provider,  # () -> np.ndarray, called only if embedding is due
    ) -> Optional[str]:
        """Return the student_id for a track (None = unknown face)."""
        known = self._tracks.get(track_id)
        if known is not None and t_ms - known.last_verified_ms < self.reverify_ms:
            return known.student_id

        embedding = self.embedder.embed(crop_provider())
        self.embed_calls += 1
        matched = self.gallery.match(embedding, self.match_threshold)

        if known is None:
            self._tracks[track_id] = _TrackIdentity(matched, t_ms)
            return matched

        # Re-verify with hysteresis: one disagreeing sample doesn't flip the
        # identity; two consecutive disagreements do.
        if matched == known.student_id:
            known.last_verified_ms = t_ms
            known.hits = min(known.hits + 1, 3)
        else:
            known.hits -= 1
            known.last_verified_ms = t_ms
            if known.hits <= 0:
                self._tracks[track_id] = _TrackIdentity(matched, t_ms)
        return self._tracks[track_id].student_id

    def drop_track(self, track_id: int) -> None:
        """Track lost — next reacquire re-embeds (rule 2c)."""
        self._tracks.pop(track_id, None)
