"""Camera capture — the webcam is opened exactly ONCE per session.

Frames are tee'd by the session runner to (a) the analysis loop and (b) the
ffmpeg recorder; nothing else may open the device. FakeCamera provides the
same interface from a video file or synthetic frames for tests/eval clips.

cv2 is imported lazily so contract/analyzer tests run without OpenCV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Frame:
    t_ms: int  # session time (ms since t0)
    image: np.ndarray  # BGR


class FrameSource(Protocol):
    width: int
    height: int
    fps: float

    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class Camera:
    """Live webcam. target_fps caps the processed rate (frames are grabbed
    at device rate but yielded at most target_fps to protect the budget)."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: float = 15.0,
        t0_monotonic: float | None = None,
    ):
        import sys

        import cv2  # lazy

        self._cv2 = cv2
        # On Windows the default MSMF backend drops frames / returns transient
        # read failures during continuous capture; DirectShow is far steadier.
        if sys.platform == "win32":
            self._cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(device_index)  # fall back
        else:
            self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera {device_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = target_fps
        self._t0 = t0_monotonic if t0_monotonic is not None else time.monotonic()

    def frames(self, max_consecutive_failures: int = 30) -> Iterator[Frame]:
        min_interval = 1.0 / self.fps
        last_yield = 0.0
        failures = 0
        while True:
            ok, image = self._cap.read()
            if not ok:
                # Tolerate transient read failures (common on Windows during
                # warmup / USB hiccups); only give up on a sustained stall.
                failures += 1
                if failures >= max_consecutive_failures:
                    return
                time.sleep(0.01)
                continue
            failures = 0
            now = time.monotonic()
            if now - last_yield < min_interval:
                continue  # grabbed (keeps driver buffer fresh) but not processed
            last_yield = now
            yield Frame(t_ms=int((now - self._t0) * 1000), image=image)

    def close(self) -> None:
        self._cap.release()


class FakeCamera:
    """Deterministic frame source for tests and the evaluation harness.

    Either wraps a recorded clip (video_path) or yields synthetic frames.
    Session time advances at exactly 1000/fps ms per frame — no wall clock.
    """

    def __init__(
        self,
        video_path: Path | str | None = None,
        synthetic_frames: list[np.ndarray] | None = None,
        fps: float = 15.0,
    ):
        self.fps = fps
        self._video_path = Path(video_path) if video_path else None
        self._synthetic = synthetic_frames
        if self._video_path is None and self._synthetic is None:
            raise ValueError("need video_path or synthetic_frames")
        if self._synthetic is not None:
            self.height, self.width = self._synthetic[0].shape[:2]
            self._cap = None
        else:
            import cv2  # lazy

            self._cap = cv2.VideoCapture(str(self._video_path))
            if not self._cap.isOpened():
                raise RuntimeError(f"cannot open clip {self._video_path}")
            self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[Frame]:
        step_ms = 1000.0 / self.fps
        i = 0
        if self._synthetic is not None:
            for image in self._synthetic:
                yield Frame(t_ms=int(i * step_ms), image=image)
                i += 1
            return
        while True:
            ok, image = self._cap.read()
            if not ok:
                return
            yield Frame(t_ms=int(i * step_ms), image=image)
            i += 1

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
