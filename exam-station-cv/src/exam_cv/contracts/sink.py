"""ArtifactSink — where the end-of-session artifact goes.

Files-always for v1. The platform backend does not exist yet; when it does,
implement a sink that uploads (artifact JSON + recording) to it. Until then
the HTTP path stays a stub — no web dependencies in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from .schemas import SessionSummary


class ArtifactSink(Protocol):
    def publish(
        self,
        summary: SessionSummary,
        events_path: Path | None = None,
        recording_path: Path | None = None,
    ) -> None: ...


class FileSink:
    """Writes summary JSON next to the session's events/recording."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def publish(
        self,
        summary: SessionSummary,
        events_path: Path | None = None,
        recording_path: Path | None = None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / f"session_{summary.session_id}_summary.json"
        out.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


class BackendSink:
    """Uploads the end-of-session artifact to the platform (seam 3).

    The consumer now exists: the Django `attribution` app ingests the
    summary's speaking timeline as evidence and files each answer against the
    student who spoke it. This is the live exam-station counterpart of the
    post-hoc path, where `cv_analysis` feeds the same artifact in directly.

    Writes locally as well as posting, always. A network failure at the end of
    a viva must not destroy the session's only record — the artifact stays on
    disk and can be replayed.
    """

    def __init__(self, endpoint: str, token: str = "", timeout: int = 60):
        self.endpoint = endpoint.rstrip("/")
        self.token = token
        self.timeout = timeout

    def publish(
        self,
        summary: SessionSummary,
        events_path: Path | None = None,
        recording_path: Path | None = None,
    ) -> None:
        import requests  # lazy: keeps contracts importable without web deps

        out_dir = events_path.parent if events_path else Path(".")
        FileSink(out_dir).publish(summary, events_path, recording_path)

        payload = {
            "summary": summary.model_dump(mode="json"),
            "recording_path": str(recording_path) if recording_path else None,
        }
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Station-Token"] = self.token

        try:
            response = requests.post(
                f"{self.endpoint}/artifact/",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                print(
                    f"artifact upload rejected ({response.status_code}): "
                    f"{response.text[:300]}",
                    flush=True,
                )
            else:
                print("artifact uploaded", flush=True)
        except Exception as e:
            # ASCII only: read back through a pipe on Windows consoles.
            print(f"artifact upload failed ({e}) - kept on disk", flush=True)


class LiveEvidenceSink:
    """Streams speaking turns to the platform DURING the session.

    The end-of-session artifact is enough to reconcile a report, but the viva
    itself adapts per student as it runs: the questioner picks the next
    question from the answering student's ability estimate. That needs to know
    who spoke before the answer is submitted, not after the session ends.

    Turns are batched rather than posted individually — a turn closes every
    second or so, and one request each would add latency to the frame loop for
    no benefit. Failures are swallowed and the batch dropped: attribution is
    decision-support, and the exam must not stall on it. Whatever is lost live
    is recovered from the artifact at the end.
    """

    def __init__(
        self,
        endpoint: str,
        session_id: str,
        t0_utc,
        token: str = "",
        batch_size: int = 8,
        timeout: int = 10,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.session_id = session_id
        self.t0_utc = t0_utc
        self.token = token
        self.batch_size = batch_size
        self.timeout = timeout
        self._pending: list[dict] = []
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="speaker-evidence",
        )

    def push(self, event) -> None:
        """Queue one AttributionEvent. Non-attribution events are ignored."""
        if getattr(event, "type", None) != "attribution":
            return
        from datetime import timedelta

        student_id = getattr(event, "student_id", None)
        # The uncertainty sentinel means "speech we could not attribute" — send
        # it as an unattributed span so the window reads as contested rather
        # than silent, exactly as the resolver expects.
        if student_id == "uncertain":
            student_id = None

        # A face we can track but cannot name travels as a track reference
        # rather than a student. The platform gives it a stable "Unknown
        # Speaker" identity that holds its marks until an examiner says who it
        # was — so an unenrolled student loses nothing by not having a photo.
        track_ref = None
        if student_id and str(student_id).startswith("unknown_track:"):
            track_ref = str(student_id).split(":", 1)[1]
            student_id = None

        self._pending.append({
            "student_id": student_id,
            "track_ref": track_ref,
            "t_start": (
                self.t0_utc + timedelta(milliseconds=event.t_start_ms)
            ).isoformat(),
            "t_end": (
                self.t0_utc + timedelta(milliseconds=event.t_end_ms)
            ).isoformat(),
            "confidence": event.confidence,
        })
        if len(self._pending) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        batch, self._pending = self._pending, []
        self._executor.submit(self._send_batch, batch)

    def close(self) -> None:
        """Queue the tail and wait for pending network writes at shutdown."""
        self.flush()
        self._executor.shutdown(wait=True)

    def _send_batch(self, batch: list[dict]) -> None:
        import requests  # lazy

        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Station-Token"] = self.token

        try:
            response = requests.post(
                f"{self.endpoint}/evidence/",
                json={"source": "live_cv", "events": batch},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as e:
            print(f"live evidence batch dropped ({e})", flush=True)
