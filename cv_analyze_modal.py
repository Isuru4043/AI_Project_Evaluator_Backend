"""Modal app — post-hoc CV/behavioral analysis of a viva recording.

Runs the exam-station-cv engine on Modal's CPU containers so the heavy CV
toolchain (mediapipe / opencv / insightface) never has to live in the Django
deploy, and no HPC box is needed.

Shape mirrors the project's other Modal apps (canary_transcribe.py,
qwen_vl_analyze.py), with one difference: analysis takes ~5-10 min for a
20-min viva, far longer than an HTTP request should live. So this exposes a
SUBMIT/POLL pair instead of one synchronous endpoint:

    POST /submit  {video_url, manifest, enrollment_photos?, token} -> {call_id}
    GET  /result?call_id=...&token=...  -> 202 running | {status: done, summary}

Django hands over SAS URLs, never bytes — a 20-min recording is ~150-250MB and
Modal fetches it straight from Azure Blob itself.

Deploy:  modal deploy cv_analyze_modal.py
Secret:  modal secret create exam-cv-token CV_ANALYZE_TOKEN=<random>
"""

import modal

app = modal.App("exam-cv-analyze")

# Returned with every identity response so Django can reject a stale rolling
# deployment instead of silently treating a one-frame legacy result as final.
ENGINE_VERSION = "face-binding-v3-aligned"

ENGINE_SRC = "exam-station-cv/src/exam_cv"
MODEL_DIR = "/root/models"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _bake_face_landmarker():
    """Fetch the MediaPipe bundle while building the Modal image.

    Local exam stations lazily cache this file under ``models/``. Requiring
    that untracked cache in the deploy source made a clean checkout unable to
    deploy, so the cloud image fetches and embeds the official bundle itself.
    """
    import urllib.request
    from pathlib import Path

    destination = Path(MODEL_DIR) / "face_landmarker.task"
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, destination)


def _bake_insightface_models():
    """Build step: pull ArcFace (buffalo_l) into the image.

    Group vivas match faces against enrollment photos, so the recognition pack
    is on the critical path; downloading it at request time would add ~170MB
    to every cold start.
    """
    import os

    # Do not emit InsightFace's Unicode tqdm progress bar while Modal streams
    # build logs to a Windows console using a legacy code page.
    os.environ["TQDM_DISABLE"] = "1"
    from insightface.utils import ensure_available

    ensure_available("models", "buffalo_l")


# CPU only — the post-hoc path is mediapipe + opencv + VAD, no GPU anywhere.
engine_image = (
    modal.Image.debian_slim(python_version="3.11")
    # libgl1/libglib2.0-0: opencv runtime; ffmpeg: the engine's audio extraction.
    .apt_install("ffmpeg", "libegl1", "libgles2", "libgl1", "libglib2.0-0")
    .pip_install(
        "numpy>=1.26",
        "pydantic>=2.5",
        "opencv-python-headless>=4.9",  # headless: no GUI libs in a container
        "mediapipe>=0.10",
        "soundfile>=0.12",
        "silero-vad>=5.0",
        "insightface>=0.7",
        "onnxruntime>=1.17",
        "requests",
    )
    .add_local_dir(ENGINE_SRC, remote_path="/root/exam_cv", copy=True)
    .run_function(_bake_face_landmarker)
    # Point the engine's asset loader at the baked-in landmarker so it never
    # downloads one at runtime (see faces/model_assets.py).
    .env({"EXAM_CV_MODEL_DIR": MODEL_DIR})
    .run_function(_bake_insightface_models)
)

# The HTTP endpoints only broker jobs — keep them off the heavy image so they
# cold-start in seconds.
api_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]"
)

token_secret = modal.Secret.from_name("exam-cv-token")


def _check_token(token: str) -> None:
    """This service fetches caller-supplied URLs, so it must not be open."""
    import hmac
    import os

    from fastapi import HTTPException

    expected = os.environ.get("CV_ANALYZE_TOKEN", "")
    if not expected or not hmac.compare_digest(token or "", expected):
        raise HTTPException(status_code=401, detail="bad token")


def _download(url: str, dest) -> None:
    import requests

    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


@app.function(image=engine_image, cpu=4.0, memory=8192, timeout=3600)
def analyze_recording(
    video_url: str,
    manifest: dict,
    enrollment_photos: dict | None = None,
) -> dict:
    """Analyze one recording; returns the SessionSummary artifact as JSON.

    ``enrollment_photos`` maps student_id -> SAS URLs for that student's
    guided reference samples. Supplied for group sessions only; without it the
    engine falls back to seating order.
    """
    import json
    import sys
    from pathlib import Path
    from urllib.parse import urlparse

    sys.path.insert(0, "/root")
    from exam_cv.analyze import analyze  # noqa: E402  (image-only import)

    work = Path("/tmp/job")
    work.mkdir(parents=True, exist_ok=True)

    suffix = Path(urlparse(video_url).path).suffix or ".mp4"
    video_path = work / f"recording{suffix}"
    print(f"downloading recording -> {video_path}", flush=True)
    _download(video_url, video_path)
    print(f"recording is {video_path.stat().st_size} bytes", flush=True)

    manifest_path = work / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    enrollment_dir = None
    if enrollment_photos:
        enrollment_dir = work / "enroll"
        enrollment_dir.mkdir(exist_ok=True)
        for student_id, value in enrollment_photos.items():
            urls = value if isinstance(value, list) else [value]
            for index, photo_url in enumerate(urls):
                try:
                    _download(
                        photo_url,
                        enrollment_dir / f"{student_id}__{index}.jpg",
                    )
                except Exception as e:
                    # One missing sample must not discard the other views.
                    print(
                        f"enrollment sample {index} failed for {student_id}: {e}",
                        flush=True,
                    )

    summary = analyze(
        video_path,
        manifest_path,
        work / "out",
        enrollment_dir=enrollment_dir,
    )
    return json.loads(summary.model_dump_json())


_MODEL_CACHE: dict = {}


def _shared_embedder(factory):
    """One ArcFace session per warm container rather than one per request."""
    embedder = _MODEL_CACHE.get("arcface")
    if embedder is None:
        embedder = _MODEL_CACHE["arcface"] = factory()
    return embedder


@app.function(
    image=engine_image,
    cpu=2.0,
    memory=4096,
    timeout=300,
    # A kiosk scans several times per session; keep the models loaded
    # between scans instead of paying the ~10s cold start each time.
    scaledown_window=600,
)
def bind_faces(
    frame_b64: str,
    enrollment_photos: dict,
    frames_b64: list | None = None,
    match_threshold: float = 0.42,
    match_margin: float = 0.05,
    enrollment_embeddings: dict | None = None,
) -> dict:
    """Match faces across a short camera burst against the enrolment gallery.

    The gallery and models are loaded once for the whole burst. Repeated
    observations tolerate blinking, head turns and different seating
    distances without weakening the ArcFace identity threshold.
    """
    import base64
    import sys
    from pathlib import Path

    import cv2
    import numpy as np

    sys.path.insert(0, "/root")
    from exam_cv.faces.identity import (  # noqa: E402  (image-only import)
        ArcFaceEmbedder,
        EnrollmentGallery,
        assign_embeddings_one_to_one,
        build_gallery_from_photos,
        face_chip,
    )
    from exam_cv.faces.mesh import (  # noqa: E402
        MeshPipeline,
        detect_faces_multiscale,
    )

    encoded_frames = (frames_b64 or [frame_b64])[:10]
    frames = [
        decoded
        for decoded in (
            cv2.imdecode(
                np.frombuffer(base64.b64decode(encoded), np.uint8),
                cv2.IMREAD_COLOR,
            )
            for encoded in encoded_frames
        )
        if decoded is not None
    ]
    if not frames:
        raise ValueError("could not decode any binding frame")

    work = Path("/tmp/bind")
    work.mkdir(parents=True, exist_ok=True)

    cached = enrollment_embeddings or {}
    photos = {}
    download_failed = set()
    for student_id, value in enrollment_photos.items():
        if cached.get(str(student_id)):
            continue
        urls = value if isinstance(value, list) else [value]
        for index, photo_url in enumerate(urls):
            dest = work / f"{student_id}__{index}.jpg"
            try:
                _download(photo_url, dest)
                image = cv2.imread(str(dest))
                if image is not None and image.size:
                    photos.setdefault(student_id, []).append(image)
            except Exception as e:
                download_failed.add(str(student_id))
                print(
                    f"enrollment sample {index} failed for {student_id}: {e}",
                    flush=True,
                )

    if not photos and not cached:
        return {
            "engine_version": ENGINE_VERSION,
            "frame_matches": [],
            "matches": [],
            "frames_processed": len(frames),
            "unusable_enrollment": sorted(set(enrollment_photos) | download_failed),
        }

    embedder = _shared_embedder(ArcFaceEmbedder)

    gallery = EnrollmentGallery()
    for student_id, vectors in cached.items():
        for vector in vectors:
            gallery.enroll(str(student_id), np.asarray(vector, dtype=np.float32))

    skipped = []
    # Vectors built from photos on this call. Django stores them so the next
    # scan sends ``enrollment_embeddings`` and no photo is downloaded again.
    fresh_embeddings = {}
    if photos:
        # Separate detector instance: enrollment advances the tracker, which
        # must not leak into the room-frame pass.
        enroll_mesh = MeshPipeline(max_faces=2)
        try:
            photo_gallery, skipped = build_gallery_from_photos(
                photos, enroll_mesh, embedder,
            )
            fresh_embeddings = photo_gallery.serializable_embeddings()
            for student_id, vectors in fresh_embeddings.items():
                for vector in vectors:
                    gallery.enroll(student_id, np.asarray(vector, dtype=np.float32))
        finally:
            enroll_mesh.close()

    if skipped:
        print(
            f"no single clear face in photos for {sorted(skipped)} - "
            "those students resolve as unknown",
            flush=True,
        )

    mesh = MeshPipeline(
        max_faces=max(5, len(photos) + 1),
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    try:
        frame_matches = []
        for frame in frames:
            matches = []
            observations = []
            embeddings = []
            # Tiled pass finds the small faces of a shared station camera
            # that a single full-frame detection misses.
            for obs in detect_faces_multiscale(
                mesh, frame, expected_faces=len(gallery.enrolled_ids()),
            ):
                crop = face_chip(mesh, frame, obs)
                if crop.size == 0:
                    continue
                observations.append(obs)
                embeddings.append(embedder.embed(crop))
            assigned = assign_embeddings_one_to_one(
                embeddings,
                gallery,
                threshold=float(match_threshold),
                min_margin=float(match_margin),
            )
            for obs, identity in zip(observations, assigned):
                matches.append({
                    "student_id": identity["student_id"],
                    "track_ref": str(obs.track_id),
                    "bbox": [float(v) for v in obs.bbox],
                    "confidence": identity["confidence"],
                    "identity_margin": identity["identity_margin"],
                })
            frame_matches.append(matches)
    finally:
        mesh.close()

    recognised = {
        match["student_id"]
        for matches in frame_matches
        for match in matches
        if match["student_id"]
    }
    print(
        f"burst bound {len(recognised)} students across {len(frames)} frames",
        flush=True,
    )

    # Keep the old response contract during rolling deployments. Older API
    # workers read only ``matches`` while current workers consume every item
    # in ``frame_matches`` and perform their own temporal aggregation. Return
    # the strongest observation for each identity so an older worker does not
    # silently turn a successful multi-frame recognition pass into zero saved
    # SpeakerBinding rows.
    strongest_by_student = {}
    best_unknowns = []
    max_detected = 0
    for matches in frame_matches:
        max_detected = max(max_detected, len(matches))
        unknowns = [match for match in matches if not match.get("student_id")]
        if len(unknowns) > len(best_unknowns):
            best_unknowns = unknowns
        for match in matches:
            student_id = match.get("student_id")
            if not student_id:
                continue
            current = strongest_by_student.get(student_id)
            if current is None or float(match.get("confidence", 0) or 0) > float(
                current.get("confidence", 0) or 0
            ):
                strongest_by_student[student_id] = match

    legacy_matches = list(strongest_by_student.values())
    unknown_slots = max(0, max_detected - len(legacy_matches))
    legacy_matches.extend(best_unknowns[:unknown_slots])
    return {
        "engine_version": ENGINE_VERSION,
        "frame_matches": frame_matches,
        "matches": legacy_matches,
        "frames_processed": len(frames),
        "enrollment_embeddings": fresh_embeddings,
        # This diagnostic used to exist only in Modal logs, leaving the kiosk
        # unable to distinguish a missing person from a bad enrollment image.
        "unusable_enrollment": sorted(
            set(enrollment_photos) - gallery.enrolled_ids()
        ),
    }


@app.function(image=engine_image, cpu=2.0, memory=4096, timeout=300)
def precompute_enrollment(enrollment_photos: dict) -> dict:
    """Turn guided enrollment photos into reusable private ArcFace vectors."""
    import sys
    from pathlib import Path

    import cv2

    sys.path.insert(0, "/root")
    from exam_cv.faces.identity import ArcFaceEmbedder, build_gallery_from_photos
    from exam_cv.faces.mesh import MeshPipeline

    work = Path("/tmp/enrollment")
    work.mkdir(parents=True, exist_ok=True)
    photos = {}
    for student_id, value in enrollment_photos.items():
        urls = value if isinstance(value, list) else [value]
        for index, photo_url in enumerate(urls):
            try:
                destination = work / f"{student_id}__{index}.jpg"
                _download(photo_url, destination)
                image = cv2.imread(str(destination))
                if image is not None and image.size:
                    photos.setdefault(str(student_id), []).append(image)
            except Exception:
                # The final unusable list is returned to Django; no diagnostic
                # is hidden only in container logs.
                pass

    if not photos:
        return {
            "engine_version": ENGINE_VERSION,
            "embeddings": {},
            "unusable_enrollment": sorted(enrollment_photos),
        }
    embedder = ArcFaceEmbedder()
    mesh = MeshPipeline(max_faces=2)
    try:
        gallery, skipped = build_gallery_from_photos(photos, mesh, embedder)
    finally:
        mesh.close()
    return {
        "engine_version": ENGINE_VERSION,
        "embeddings": gallery.serializable_embeddings(),
        "unusable_enrollment": sorted(
            set(enrollment_photos) - gallery.enrolled_ids()
        ),
    }


@app.function(image=api_image, secrets=[token_secret], timeout=300)
@modal.fastapi_endpoint(method="POST")
def bind(payload: dict):
    """Bind faces from a short camera burst to roster students."""
    _check_token(payload.get("token", ""))

    from fastapi import HTTPException

    frame_b64 = payload.get("frame_b64")
    photos = payload.get("enrollment_photos") or {}
    if not frame_b64:
        raise HTTPException(status_code=400, detail="frame_b64 required")
    if not photos:
        raise HTTPException(
            status_code=400, detail="enrollment_photos required to match against"
        )

    try:
        result = bind_faces.remote(
            frame_b64,
            photos,
            payload.get("frames_b64") or [frame_b64],
            float(payload.get("match_threshold", 0.42)),
            float(payload.get("match_margin", 0.05)),
            payload.get("enrollment_embeddings") or {},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"binding failed: {str(e)[:300]}")
    return result


@app.function(image=api_image, secrets=[token_secret], timeout=300)
@modal.fastapi_endpoint(method="POST")
def enroll(payload: dict):
    """Authenticated precomputation endpoint called after face registration."""
    _check_token(payload.get("token", ""))
    from fastapi import HTTPException

    photos = payload.get("enrollment_photos") or {}
    if not photos:
        raise HTTPException(status_code=400, detail="enrollment_photos required")
    try:
        return precompute_enrollment.remote(photos)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"enrollment precomputation failed: {str(exc)[:300]}",
        )


@app.function(image=api_image, secrets=[token_secret], timeout=30)
@modal.fastapi_endpoint(method="GET")
def version(token: str):
    """Deployment probe used to confirm the expected face-binding engine."""
    _check_token(token)
    return {"engine_version": ENGINE_VERSION}


@app.function(image=api_image, secrets=[token_secret], timeout=60)
@modal.fastapi_endpoint(method="POST")
def submit(payload: dict):
    """Queue an analysis; returns the call id to poll with."""
    _check_token(payload.get("token", ""))

    from fastapi import HTTPException

    video_url = payload.get("video_url")
    manifest = payload.get("manifest")
    if not video_url or not manifest:
        raise HTTPException(status_code=400, detail="video_url and manifest required")

    call = analyze_recording.spawn(
        video_url, manifest, payload.get("enrollment_photos") or None,
    )
    return {"call_id": call.object_id}


@app.function(image=api_image, secrets=[token_secret], timeout=60)
@modal.fastapi_endpoint(method="GET")
def result(call_id: str, token: str):
    """Poll a submitted job. 202 while it is still running."""
    from fastapi.responses import JSONResponse

    _check_token(token)

    function_call = modal.FunctionCall.from_id(call_id)
    try:
        summary = function_call.get(timeout=0)
    except TimeoutError:
        return JSONResponse({"status": "running"}, status_code=202)
    except Exception as e:
        # The job itself raised (or its result expired) — terminal, so the
        # caller stops polling and surfaces the reason.
        return JSONResponse({"status": "failed", "error": str(e)[:800]})
    return {"status": "done", "summary": summary}
