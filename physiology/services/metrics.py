"""Heart-rate variability maths. Pure functions, no Django, no database.

This is where the actual claim about a student gets made, so it is kept
separate and testable with scripted interval traces - the same discipline the
CV analyzers follow.

WHY INTERVALS, NOT BPM
    Arousal shows up in the VARIABILITY between beats, not in the average
    rate. Under sympathetic arousal successive beats become more uniform, so
    RMSSD falls. A rate that has been median-filtered and smoothed - which is
    what a display wants - has had that signal removed. So everything here
    works from inter-beat intervals in milliseconds.

WHY RMSSD
    Of the standard time-domain HRV measures, RMSSD is the one that stays
    meaningful over short windows (30 s is accepted; SDNN really wants 5 min).
    A viva answer lasts under a minute, so RMSSD is the only established
    metric that fits the question being asked.

WHY BOTH DIRECTIONS MUST AGREE
    Speaking raises heart rate on its own, so a rate rise by itself says
    little about how a student felt. Requiring RMSSD to fall at the same time
    removes most of that confound: talking does not collapse variability the
    way genuine arousal does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

# Physiologically plausible interval bounds: 40-180 bpm.
IBI_MIN_MS = 333
IBI_MAX_MS = 1500

# An interval differing from its neighbour by more than this is a detection
# artifact (a missed or doubled beat), not a real rhythm change. 25% follows
# the usual short-window HRV cleaning convention. It is applied to SUCCESSIVE
# differences rather than to a running median, because a median filter would
# also remove the genuine variability being measured.
ARTIFACT_TOLERANCE = 0.25

# Below this many clean beats the numbers are noise dressed as data. This
# governs the rolling 30 s analysis windows, which at any plausible rate hold
# far more beats than this.
MIN_BEATS = 20

# The calm baseline is deliberately short: it is dead time in front of an
# examiner, and asking a student to sit still for the best part of a minute
# before their viva costs more than the precision it buys. A 10 s window holds
# roughly 8 beats at a slow resting rate and about 16 at a fast one, so the
# usability floor has to match the window rather than the 30 s one.
#
# This is ultra-short HRV. RMSSD from that few intervals is coarser than a
# textbook recording, and it is used only as each student's own reference
# point for their own later windows - never compared between people, and
# never scored. The comparison stays valid; its resolution is lower.
MIN_BASELINE_BEATS = 8

# How far above baseline the rate must sit, in the student's own SDs.
HR_Z_THRESHOLD = 1.0
# ...and how far in plain beats per minute. Both are required, because a
# student whose resting rate is very steady has a tiny SD, and then a rise of
# two or three beats scores several SDs and would be flagged as arousal. A
# z-score alone measures "unusual for this person" but not "enough to matter".
MIN_HR_DELTA_BPM = 5.0
# How far RMSSD must fall, as a fraction of baseline.
RMSSD_RATIO_THRESHOLD = 0.80


@dataclass(frozen=True)
class HrvMetrics:
    mean_hr: Optional[float]
    rmssd: Optional[float]
    sdnn: Optional[float]
    beat_count: int
    quality: float          # share of supplied intervals that survived cleaning

    @property
    def is_usable(self) -> bool:
        return self.beat_count >= MIN_BEATS and self.quality >= 0.5


def clean_intervals(ibis: Iterable[float]) -> tuple[list[float], float]:
    """Drop implausible and artifact intervals. Returns (kept, quality).

    Two passes, deliberately gentle. Anything stricter starts deleting the
    variability that is the whole measurement.
    """
    raw = [float(v) for v in ibis if v is not None]
    if not raw:
        return [], 0.0

    plausible = [v for v in raw if IBI_MIN_MS <= v <= IBI_MAX_MS]
    if not plausible:
        return [], 0.0

    kept = [plausible[0]]
    for value in plausible[1:]:
        previous = kept[-1]
        if abs(value - previous) / previous <= ARTIFACT_TOLERANCE:
            kept.append(value)
        # else: a dropped or doubled beat - skip it, keep the reference beat

    return kept, len(kept) / len(raw)


def rmssd(ibis: Sequence[float]) -> Optional[float]:
    """Root mean square of successive differences, in ms."""
    if len(ibis) < 2:
        return None
    diffs = [ibis[i + 1] - ibis[i] for i in range(len(ibis) - 1)]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def sdnn(ibis: Sequence[float]) -> Optional[float]:
    """Standard deviation of intervals, in ms. Reported for completeness;
    over a sub-minute window it is not trustworthy on its own."""
    if len(ibis) < 2:
        return None
    mean = sum(ibis) / len(ibis)
    variance = sum((v - mean) ** 2 for v in ibis) / (len(ibis) - 1)
    return math.sqrt(variance)


def mean_hr(ibis: Sequence[float]) -> Optional[float]:
    """Mean heart rate in bpm, derived from the intervals.

    Taken as 60000 / mean(interval) rather than the mean of per-beat rates:
    averaging rates over-weights short intervals and biases the result upward.
    """
    if not ibis:
        return None
    average = sum(ibis) / len(ibis)
    return 60000.0 / average if average > 0 else None


def hr_sd(ibis: Sequence[float]) -> Optional[float]:
    """Spread of instantaneous heart rate, in bpm. This is the unit the
    baseline z-score is expressed in, so it must come from the same beats."""
    if len(ibis) < 2:
        return None
    rates = [60000.0 / v for v in ibis if v > 0]
    if len(rates) < 2:
        return None
    mean = sum(rates) / len(rates)
    variance = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
    return math.sqrt(variance)


def compute(ibis: Iterable[float]) -> HrvMetrics:
    """Clean a set of intervals and derive every metric from what survives."""
    kept, quality = clean_intervals(ibis)
    return HrvMetrics(
        mean_hr=mean_hr(kept),
        rmssd=rmssd(kept),
        sdnn=sdnn(kept),
        beat_count=len(kept),
        quality=round(quality, 3),
    )


@dataclass(frozen=True)
class Arousal:
    """One window's standing relative to the student's own resting state."""

    hr_z: Optional[float]           # SDs above baseline rate
    hr_delta: Optional[float]       # bpm above baseline - the readable number
    rmssd_ratio: Optional[float]    # window RMSSD / baseline RMSSD
    elevated: bool                  # both signals agree
    usable: bool                    # enough clean beats to say anything
    reason: str


def arousal(
    window: HrvMetrics,
    baseline_hr_mean: Optional[float],
    baseline_hr_sd: Optional[float],
    baseline_rmssd: Optional[float],
    hr_z_threshold: float = HR_Z_THRESHOLD,
    rmssd_ratio_threshold: float = RMSSD_RATIO_THRESHOLD,
    min_hr_delta: float = MIN_HR_DELTA_BPM,
) -> Arousal:
    """Compare one window against the student's baseline.

    Returns `elevated` only when the rate is up AND variability is down. Either
    alone is too easily produced by talking, moving, or a noisy clip.

    `usable=False` is a first-class outcome and must be shown as "no reading",
    never rendered as a calm one.
    """
    if not window.is_usable:
        return Arousal(None, None, None, False, False, 'too few clean beats')
    if baseline_hr_mean is None or baseline_rmssd is None:
        return Arousal(None, None, None, False, False, 'no baseline')

    # A student whose resting rate barely moves would otherwise divide by ~0
    # and produce enormous z-scores from trivial changes.
    sd = baseline_hr_sd if (baseline_hr_sd and baseline_hr_sd >= 1.0) else 1.0

    delta = window.mean_hr - baseline_hr_mean
    hr_z = delta / sd
    ratio = (window.rmssd / baseline_rmssd) if baseline_rmssd > 0 else None

    if ratio is None:
        return Arousal(round(hr_z, 2), round(delta, 1), None, False, False,
                       'baseline rmssd zero')

    rate_up = hr_z >= hr_z_threshold and delta >= min_hr_delta
    variability_down = ratio <= rmssd_ratio_threshold
    elevated = rate_up and variability_down

    if elevated:
        reason = 'rate up and variability down'
    elif rate_up:
        reason = 'rate up only - not distinguishable from speaking'
    elif variability_down and hr_z >= hr_z_threshold:
        # Unusual for this student, but only by a couple of beats.
        reason = 'rise too small to matter'
    elif variability_down:
        reason = 'variability down only'
    else:
        reason = 'within resting range'

    return Arousal(round(hr_z, 2), round(delta, 1), round(ratio, 3),
                   elevated, True, reason)
