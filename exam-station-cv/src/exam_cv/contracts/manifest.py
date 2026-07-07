"""Session manifest loading + standalone/dev generation (seam 1).

The platform hands us a manifest; in standalone mode we generate one with the
identical shape so nothing downstream knows the difference.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schemas import RosterEntry, SessionManifest, SessionMode


def load_manifest(path: Path | str) -> SessionManifest:
    return SessionManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def standalone_manifest(
    student_names: list[str],
    mode: SessionMode | None = None,
) -> SessionManifest:
    """Dev/standalone manifest. student_ids are local-only placeholders;
    a platform-issued manifest replaces this wholesale at integration."""
    if not student_names:
        raise ValueError("at least one student required")
    if mode is None:
        mode = SessionMode.INDIVIDUAL if len(student_names) == 1 else SessionMode.GROUP
    return SessionManifest(
        session_id=str(uuid.uuid4()),
        mode=mode,
        roster=[
            RosterEntry(student_id=f"local-{i+1}", display_name=name)
            for i, name in enumerate(student_names)
        ],
        t0_utc=datetime.now(timezone.utc),
        notes="standalone/dev manifest — not platform-issued",
    )
