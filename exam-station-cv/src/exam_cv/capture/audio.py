"""Audio capture + voice activity detection.

The mic is opened once; blocks are tee'd to (a) the VAD and (b) a WAV file
that the recorder muxes with video at session end (mono mic — Razer Seiren
Mini class — so speaker identity comes from vision, not audio direction).

Silero VAD is the runtime detector; EnergyVAD is the dependency-free fallback
used in tests and as a safety net if torch/silero is unavailable.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

SAMPLE_RATE = 16000  # silero's native rate; fine for VAD + speech review


@dataclass
class AudioBlock:
    t_ms: int
    samples: np.ndarray  # float32 mono, SAMPLE_RATE


class VoiceActivityDetector(Protocol):
    def is_speech(self, block: AudioBlock) -> bool: ...


class EnergyVAD:
    """RMS-energy VAD with an adaptive noise floor. Test/fallback detector."""

    def __init__(self, ratio: float = 3.0, floor_alpha: float = 0.02):
        self.ratio = ratio
        self.floor_alpha = floor_alpha
        self._noise_floor: Optional[float] = None

    def is_speech(self, block: AudioBlock) -> bool:
        rms = float(np.sqrt(np.mean(block.samples**2)) + 1e-9)
        if self._noise_floor is None:
            self._noise_floor = rms
            return False
        active = rms > self._noise_floor * self.ratio
        if not active:  # only adapt the floor on non-speech
            self._noise_floor = (
                (1 - self.floor_alpha) * self._noise_floor + self.floor_alpha * rms
            )
        return active


class SileroVAD:
    """Silero VAD (lazy import; falls back is caller's decision)."""

    def __init__(self, threshold: float = 0.5):
        from silero_vad import load_silero_vad  # lazy

        import torch  # lazy

        self._torch = torch
        self._model = load_silero_vad()
        self.threshold = threshold

    def is_speech(self, block: AudioBlock) -> bool:
        # silero expects chunks of 512 samples @16k; use the block mean prob
        t = self._torch
        samples = block.samples
        probs = []
        for start in range(0, len(samples) - 511, 512):
            chunk = t.from_numpy(samples[start : start + 512])
            probs.append(float(self._model(chunk, SAMPLE_RATE).item()))
        return bool(probs) and max(probs) >= self.threshold


class AudioCapture:
    """sounddevice input stream → timestamped blocks on a queue + WAV tee."""

    def __init__(
        self,
        wav_path: Path,
        block_ms: int = 96,  # 3 × 512 samples @ 16 kHz
        device: Optional[int] = None,
        t0_monotonic: Optional[float] = None,
    ):
        import sounddevice as sd  # lazy
        import soundfile as sf  # lazy
        import time

        self._sd = sd
        self._blocksize = SAMPLE_RATE * block_ms // 1000
        self._queue: "queue.Queue[AudioBlock]" = queue.Queue(maxsize=256)
        self._wav = sf.SoundFile(
            str(wav_path), mode="w", samplerate=SAMPLE_RATE, channels=1
        )
        self._wav_lock = threading.Lock()
        self._t0 = t0_monotonic if t0_monotonic is not None else time.monotonic()
        self._time = time

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            device=device,
            callback=self._on_block,
        )

    def _on_block(self, indata, frames, time_info, status) -> None:
        samples = indata[:, 0].copy()
        t_ms = int((self._time.monotonic() - self._t0) * 1000)
        with self._wav_lock:
            if not self._wav.closed:
                self._wav.write(samples)
        try:
            self._queue.put_nowait(AudioBlock(t_ms=t_ms, samples=samples))
        except queue.Full:
            pass  # VAD can afford to drop; the WAV tee above never drops

    def start(self) -> None:
        self._stream.start()

    def get_block(self, timeout: float = 0.5) -> Optional[AudioBlock]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
        with self._wav_lock:
            self._wav.close()
