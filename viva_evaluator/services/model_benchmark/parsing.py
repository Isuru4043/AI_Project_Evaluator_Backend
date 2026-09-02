from __future__ import annotations

import json
import re
from typing import Any, Optional


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def parse_json_response(text: str) -> tuple[Optional[Any], bool]:
    """Return parsed JSON and whether the provider emitted strict raw JSON.

    Benchmark scoring accepts recoverable fenced/surrounded JSON so model
    quality is not confused with presentation formatting. Strict adherence is
    returned separately and remains available as an evaluation metric.
    """
    raw = str(text or "").strip()
    if not raw:
        return None, False
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        pass

    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        pass

    # Explanatory prose before a complete JSON payload is recoverable, but a
    # truncated top-level object must not be "rescued" by parsing one of its
    # nested objects. Only the earliest JSON opening character can represent
    # the provider's intended top-level response.
    openings = [index for index in (raw.find("{"), raw.find("[")) if index >= 0]
    if openings:
        try:
            value, _ = json.JSONDecoder().raw_decode(raw[min(openings):])
            return value, False
        except json.JSONDecodeError:
            pass
    return None, False
