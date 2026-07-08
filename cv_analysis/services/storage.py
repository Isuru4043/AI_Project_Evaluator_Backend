"""Recording storage — local disk (default) or Azure blob.

Local storage keeps viva recordings on the machine running the backend,
avoiding Azure egress/storage cost. Selected by CV_RECORDING_STORAGE
('local' | 'azure'). References are stored in SessionRecording.video_file_url:
an ``http(s)://`` URL for Azure, or an absolute filesystem path for local.
"""

import re
import uuid
from pathlib import Path

from django.conf import settings
from django.core.signing import BadSignature, SignatureExpired, TimestampSigner

_SIGNER_SALT = 'cv_analysis.recording.playback'


def storage_backend() -> str:
    return getattr(settings, 'CV_RECORDING_STORAGE', 'local').lower()


def recordings_root() -> Path:
    root = Path(getattr(settings, 'CV_RECORDINGS_DIR', 'cv_recordings'))
    root.mkdir(parents=True, exist_ok=True)
    return root


def save_recording_locally(uploaded_file, session_id) -> str:
    """Stream an uploaded file to <CV_RECORDINGS_DIR>/<session_id>/… .
    Returns the absolute path (stored as the recording reference)."""
    session_dir = recordings_root() / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower() or '.webm'
    dest = session_dir / f"recording_{uuid.uuid4().hex[:8]}{ext}"
    with open(dest, 'wb') as f:
        for chunk in uploaded_file.chunks():
            f.write(chunk)
    return str(dest)


def is_local_recording(ref: str) -> bool:
    return bool(ref) and not ref.lower().startswith(('http://', 'https://'))


def content_type_for(path: Path) -> str:
    return 'video/mp4' if path.suffix.lower() == '.mp4' else 'video/webm'


# --- Signed playback tokens ------------------------------------------------
# The <video> element can't send an Authorization header, so local playback
# uses a short-lived signed token in the query string (the SAS-URL analogue).

def make_playback_token(session_id) -> str:
    return TimestampSigner(salt=_SIGNER_SALT).sign(str(session_id))


def check_playback_token(token: str, session_id, max_age: int = 7200) -> bool:
    try:
        value = TimestampSigner(salt=_SIGNER_SALT).unsign(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return False
    return value == str(session_id)


# --- Range-aware file serving ----------------------------------------------

_RANGE_RE = re.compile(r'bytes=(\d+)-(\d*)')


def _range_stream(path: Path, start: int, length: int, block: int = 65536):
    with open(path, 'rb') as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            data = f.read(min(block, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def serve_file_with_range(request, path: Path):
    """Serve a local file honoring HTTP Range requests (206) so the browser
    <video> can seek — required for the flag-timecode jump feature."""
    from django.http import FileResponse, StreamingHttpResponse

    content_type = content_type_for(path)
    file_size = path.stat().st_size
    range_header = request.headers.get('Range', '')
    match = _RANGE_RE.match(range_header)

    if match:
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
        end = min(end, file_size - 1)
        if start > end:
            start = 0
        length = end - start + 1
        resp = StreamingHttpResponse(
            _range_stream(path, start, length),
            status=206,
            content_type=content_type,
        )
        resp['Content-Length'] = str(length)
        resp['Content-Range'] = f'bytes {start}-{end}/{file_size}'
    else:
        resp = FileResponse(open(path, 'rb'), content_type=content_type)
        resp['Content-Length'] = str(file_size)

    resp['Accept-Ranges'] = 'bytes'
    return resp
