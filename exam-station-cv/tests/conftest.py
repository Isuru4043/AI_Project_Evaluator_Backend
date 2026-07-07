import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exam_cv.contracts.schemas import RosterEntry, SessionManifest, SessionMode


@pytest.fixture
def group_manifest() -> SessionManifest:
    return SessionManifest(
        session_id="test-session-1",
        mode=SessionMode.GROUP,
        roster=[
            RosterEntry(student_id="s1", display_name="Alice"),
            RosterEntry(student_id="s2", display_name="Bob"),
            RosterEntry(student_id="s3", display_name="Cara"),
        ],
        t0_utc=datetime(2026, 7, 6, 9, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def individual_manifest() -> SessionManifest:
    return SessionManifest(
        session_id="test-session-2",
        mode=SessionMode.INDIVIDUAL,
        roster=[RosterEntry(student_id="s1", display_name="Alice")],
        t0_utc=datetime(2026, 7, 6, 9, 0, tzinfo=timezone.utc),
    )
