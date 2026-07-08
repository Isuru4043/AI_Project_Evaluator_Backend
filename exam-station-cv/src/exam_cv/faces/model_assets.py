"""On-demand model asset fetch (MediaPipe FaceLandmarker task bundle).

The bundle is downloaded once into a local cache dir (``models/`` at the
package root by default, gitignored) and reused thereafter.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Package root: .../src/exam_cv/faces/model_assets.py -> project root is parents[3]
_DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "models"


def face_landmarker_path(cache_dir: Path | None = None) -> Path:
    """Return the local path to face_landmarker.task, downloading if absent.

    Override the cache location with EXAM_CV_MODEL_DIR.
    """
    cache = Path(
        cache_dir
        or os.getenv("EXAM_CV_MODEL_DIR")
        or _DEFAULT_CACHE
    )
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / "face_landmarker.task"
    if not dest.exists() or dest.stat().st_size == 0:
        tmp = dest.with_suffix(".task.part")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
        tmp.replace(dest)
    return dest
