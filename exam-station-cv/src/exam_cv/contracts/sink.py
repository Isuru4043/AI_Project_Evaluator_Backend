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
    """Stub for the future platform backend (seam 3). Intentionally
    unimplemented until a consumer exists."""

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def publish(
        self,
        summary: SessionSummary,
        events_path: Path | None = None,
        recording_path: Path | None = None,
    ) -> None:
        raise NotImplementedError(
            "Backend upload is a stub until the platform exists. "
            "Use FileSink; the report generator ingests the JSON artifact."
        )
