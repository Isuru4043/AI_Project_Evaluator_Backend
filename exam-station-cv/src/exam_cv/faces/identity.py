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
        self._gallery.setdefault(student_id, []).append(embedding)

    def match(self, embedding: np.ndarray, threshold: float = 0.35) -> Optional[str]:
        """Best cosine match above threshold, else None (unknown face)."""
        best_id, best_sim = None, threshold
        for sid, embs in self._gallery.items():
            sim = max(cosine(embedding, e) for e in embs)
            if sim > best_sim:
                best_id, best_sim = sid, sim
        return best_id

    def enrolled_ids(self) -> set[str]:
        return set(self._gallery.keys())


def build_gallery_from_photos(
    photos: dict[str, np.ndarray],
    mesh,
    embedder: FaceEmbedder,
) -> tuple["EnrollmentGallery", list[str]]:
    """Enroll students from their reference face photos.

    ``photos`` maps student_id → a decoded BGR image. ``mesh`` must be a
    throwaway MeshPipeline (its tracker state is mutated here and would
    otherwise pollute the video pass); it is the sole detector, so ArcFace
    still never runs its own detection (rule 1).

    Returns (gallery, skipped_ids). A photo is skipped when it shows no face
    or more than one — an ambiguous reference is left unenrolled so the
    student resolves to unknown rather than being guessed (HITL invariant).
    """
    gallery = EnrollmentGallery()
    skipped: list[str] = []
    for student_id, image in photos.items():
        observations = mesh.process_frame(image)
        if len(observations) != 1:
            skipped.append(student_id)
            continue
        crop = mesh.crop(image, observations[0])
        if crop.size == 0:
            skipped.append(student_id)
            continue
        gallery.enroll(student_id, embedder.embed(crop))
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
