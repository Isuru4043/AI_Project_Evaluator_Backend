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

    ``enrollment_photos`` maps student_id -> SAS URL of that student's
    reference face photo. Supplied for group sessions only; without it the
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
        for student_id, photo_url in enrollment_photos.items():
            try:
                _download(photo_url, enrollment_dir / f"{student_id}.jpg")
            except Exception as e:
                # A missing photo costs that student face-identification (they
                # resolve to unknown); it must not fail the whole analysis.
                print(f"enrollment photo failed for {student_id}: {e}", flush=True)

    summary = analyze(
        video_path,
        manifest_path,
        work / "out",
        enrollment_dir=enrollment_dir,
    )
    return json.loads(summary.model_dump_json())


@app.function(image=engine_image, cpu=2.0, memory=4096, timeout=300)
def bind_faces(frame_b64: str, enrollment_photos: dict) -> list:
    """Match every face in one still frame against the enrolment gallery.

    This is the physical exam-station path: one camera, the whole group in
    frame, so before lip activity can name a speaker each face has to be tied
    to a roster student. Recognition is expensive but only changes when
    someone moves seats, so it runs here — rarely, on one frame — while the
    kiosk does the continuous per-frame work.

    Synchronous rather than submit/poll: this is a handful of embeddings on a
    single image (a second or two), not a 20-minute video, and the kiosk is
    waiting on it before questioning starts.

    Returns [{student_id, track_ref, bbox, confidence}]. A face matching no
    enrolled student comes back with student_id=None — that is the
    extra-person case the integrity flags exist for, never a nearest guess.
    """
    import base64
    import sys
    from pathlib import Path

    import cv2
    import numpy as np

    sys.path.insert(0, "/root")
    from exam_cv.faces.identity import (  # noqa: E402  (image-only import)
        ArcFaceEmbedder,
        build_gallery_from_photos,
    )
    from exam_cv.faces.mesh import MeshPipeline  # noqa: E402

    frame = cv2.imdecode(
        np.frombuffer(base64.b64decode(frame_b64), np.uint8), cv2.IMREAD_COLOR
    )
    if frame is None:
        raise ValueError("could not decode the binding frame")

    work = Path("/tmp/bind")
    work.mkdir(parents=True, exist_ok=True)

    photos = {}
    for student_id, photo_url in enrollment_photos.items():
        dest = work / f"{student_id}.jpg"
        try:
            _download(photo_url, dest)
            image = cv2.imread(str(dest))
            if image is not None and image.size:
                photos[student_id] = image
        except Exception as e:
            # An unreadable photo costs that student recognition; everyone
            # else must still be bound.
            print(f"enrollment photo failed for {student_id}: {e}", flush=True)

    if not photos:
        print("no usable enrollment photos - nothing to match against", flush=True)
        return []

    embedder = ArcFaceEmbedder()

    # Separate detector instance: build_gallery_from_photos advances the
    # tracker, which must not leak into the frame pass.
    enroll_mesh = MeshPipeline(max_faces=2)
    try:
        gallery, skipped = build_gallery_from_photos(photos, enroll_mesh, embedder)
    finally:
        enroll_mesh.close()

    if skipped:
        print(
            f"no single clear face in photos for {sorted(skipped)} - "
            "those students resolve as unknown",
            flush=True,
        )

    mesh = MeshPipeline(max_faces=max(5, len(photos) + 1))
    try:
        matches = []
        for obs in mesh.process_frame(frame):
            crop = mesh.crop(frame, obs)
            if crop.size == 0:
                continue
            student_id = gallery.match(embedder.embed(crop))
            matches.append({
                "student_id": student_id,
                "track_ref": str(obs.track_id),
                "bbox": [float(v) for v in obs.bbox],
                "confidence": 1.0 if student_id else 0.0,
            })
    finally:
        mesh.close()

    recognised = sum(1 for m in matches if m["student_id"])
    print(f"bound {recognised}/{len(matches)} faces to students", flush=True)
    return matches


@app.function(image=api_image, secrets=[token_secret], timeout=300)
@modal.fastapi_endpoint(method="POST")
def bind(payload: dict):
    """Bind the faces in one frame to roster students. Synchronous."""
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
        matches = bind_faces.remote(frame_b64, photos)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"binding failed: {str(e)[:300]}")
    return {"matches": matches}


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
