"""Append-only JSONL event store. No DB, no bus — one local process,
tiny event volume per session; everything replayable for the summary pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Union

from pydantic import BaseModel

from ..contracts.schemas import AttributionEvent, BehavioralEvent, IntegrityFlag

Event = Union[AttributionEvent, BehavioralEvent, IntegrityFlag]

_EVENT_TYPES = {
    "attribution": AttributionEvent,
    "behavioral": BehavioralEvent,
    "integrity_flag": IntegrityFlag,
}


def append_event(path: Path, event: BaseModel) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(event.model_dump_json())
        f.write("\n")


def read_events(path: Path) -> Iterator[Event]:
    # A session that produced no events never creates the file — that's a
    # valid (if empty) session, so treat a missing file as zero events.
    if not Path(path).exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            cls = _EVENT_TYPES.get(raw.get("type"))
            if cls is None:
                raise ValueError(f"unknown event type in {path}: {raw.get('type')!r}")
            yield cls.model_validate(raw)
