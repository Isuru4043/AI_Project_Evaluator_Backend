"""Tests for the HRV maths.

These carry the correctness burden of the whole app: everything downstream is
plumbing, but a wrong RMSSD becomes a wrong claim about how a student felt.
No database, no hardware - scripted interval traces only.
"""

import math

from django.test import SimpleTestCase

from physiology.services.metrics import (
    ARTIFACT_TOLERANCE,
    MIN_BEATS,
    Arousal,
    arousal,
    clean_intervals,
    compute,
    hr_sd,
    mean_hr,
    rmssd,
    sdnn,
)


def steady(count, ibi=1000.0):
    """A metronome: identical intervals, i.e. zero variability."""
    return [ibi] * count


def varied(count, ibi=1000.0, swing=40.0):
    """Alternating intervals - healthy beat-to-beat variation."""
    return [ibi + (swing if i % 2 else -swing) for i in range(count)]


class CleaningTests(SimpleTestCase):
    def test_implausible_intervals_are_dropped(self):
        kept, quality = clean_intervals([1000, 50, 1000, 9000, 1000])
        self.assertEqual(kept, [1000, 1000, 1000])
        self.assertAlmostEqual(quality, 0.6)

    def test_a_missed_beat_is_rejected(self):
        """A skipped detection doubles one interval; keeping it would invent
        variability that never happened."""
        kept, _ = clean_intervals([800, 800, 1600, 800])
        self.assertNotIn(1600, kept)

    def test_genuine_variation_survives(self):
        """Cleaning must not flatten the signal being measured."""
        trace = varied(20, swing=40)      # 8% swing, well inside tolerance
        kept, quality = clean_intervals(trace)
        self.assertEqual(len(kept), len(trace))
        self.assertEqual(quality, 1.0)

    def test_empty_input_is_safe(self):
        self.assertEqual(clean_intervals([]), ([], 0.0))


class MetricTests(SimpleTestCase):
    def test_rmssd_of_a_metronome_is_zero(self):
        self.assertEqual(rmssd(steady(10)), 0.0)

    def test_rmssd_rises_with_variability(self):
        self.assertGreater(rmssd(varied(20, swing=50)), rmssd(varied(20, swing=10)))

    def test_rmssd_matches_hand_calculation(self):
        # diffs: +100, -100, +100 -> rms = 100
        self.assertAlmostEqual(rmssd([800, 900, 800, 900]), 100.0)

    def test_rmssd_needs_two_beats(self):
        self.assertIsNone(rmssd([1000]))

    def test_mean_hr_from_intervals(self):
        self.assertAlmostEqual(mean_hr(steady(10, 1000.0)), 60.0)
        self.assertAlmostEqual(mean_hr(steady(10, 750.0)), 80.0)

    def test_mean_hr_uses_mean_interval_not_mean_rate(self):
        """Averaging per-beat rates biases upward; the two differ measurably."""
        ibis = [500.0, 1500.0]                      # 120 bpm and 40 bpm
        naive = sum(60000.0 / v for v in ibis) / 2  # = 80
        self.assertAlmostEqual(mean_hr(ibis), 60.0)
        self.assertNotAlmostEqual(mean_hr(ibis), naive)

    def test_sdnn_of_a_metronome_is_zero(self):
        self.assertEqual(sdnn(steady(10)), 0.0)

    def test_hr_sd_is_zero_for_a_metronome(self):
        self.assertEqual(hr_sd(steady(10)), 0.0)


class ComputeTests(SimpleTestCase):
    def test_usable_requires_enough_beats(self):
        self.assertFalse(compute(steady(5)).is_usable)
        self.assertTrue(compute(varied(MIN_BEATS + 5)).is_usable)

    def test_quality_reflects_rejected_beats(self):
        trace = varied(30) + [50, 9000, 60]      # three impossible values
        self.assertLess(compute(trace).quality, 1.0)

    def test_metrics_are_none_when_there_is_nothing_to_measure(self):
        m = compute([])
        self.assertIsNone(m.mean_hr)
        self.assertIsNone(m.rmssd)
        self.assertFalse(m.is_usable)


class ArousalTests(SimpleTestCase):
    """The decision rule. Every branch here becomes something an examiner
    reads about a student, so each is pinned."""

    BASE_HR = 70.0
    BASE_SD = 4.0
    BASE_RMSSD = 45.0

    def _arousal(self, window_ibis):
        return arousal(
            compute(window_ibis), self.BASE_HR, self.BASE_SD, self.BASE_RMSSD,
        )

    def test_rate_up_and_variability_down_is_elevated(self):
        # ~86 bpm (700ms) with almost no swing -> rmssd far below baseline
        result = self._arousal(varied(30, ibi=700.0, swing=3.0))
        self.assertTrue(result.usable)
        self.assertTrue(result.elevated)
        self.assertGreater(result.hr_z, 1.0)
        self.assertLess(result.rmssd_ratio, 0.8)

    def test_rate_up_alone_is_not_elevated(self):
        """Talking raises heart rate. Without a variability drop that is not
        evidence of anything, and must not be reported as though it were."""
        result = self._arousal(varied(30, ibi=700.0, swing=45.0))
        self.assertTrue(result.usable)
        self.assertFalse(result.elevated)
        self.assertIn('rate up only', result.reason)

    def test_resting_window_is_not_elevated(self):
        result = self._arousal(varied(30, ibi=857.0, swing=40.0))  # ~70 bpm
        self.assertTrue(result.usable)
        self.assertFalse(result.elevated)
        self.assertEqual(result.reason, 'within resting range')

    def test_too_few_beats_is_unusable_not_calm(self):
        """The critical distinction: no reading must never render as a calm
        reading."""
        result = self._arousal(steady(4))
        self.assertFalse(result.usable)
        self.assertFalse(result.elevated)
        self.assertEqual(result.reason, 'too few clean beats')

    def test_missing_baseline_yields_no_verdict(self):
        result = arousal(compute(varied(30, ibi=700.0, swing=3.0)),
                         None, None, None)
        self.assertFalse(result.usable)
        self.assertEqual(result.reason, 'no baseline')

    def test_flat_baseline_does_not_explode_the_z_score(self):
        """A student whose resting rate barely moves would otherwise divide by
        near-zero and be flagged by trivial changes."""
        result = arousal(
            compute(varied(30, ibi=850.0, swing=20.0)),
            baseline_hr_mean=70.0, baseline_hr_sd=0.0, baseline_rmssd=45.0,
        )
        self.assertTrue(result.usable)
        self.assertLess(abs(result.hr_z), 10.0)

    def test_baseline_is_personal(self):
        """The same window is elevated for a slow-hearted student and
        unremarkable for a fast-hearted one - the reason absolute thresholds
        are not used anywhere."""
        window = compute(varied(30, ibi=700.0, swing=3.0))   # ~86 bpm, flat
        slow = arousal(window, 65.0, 4.0, 45.0)
        fast = arousal(window, 88.0, 4.0, 20.0)
        self.assertTrue(slow.elevated)
        self.assertFalse(fast.elevated)


class HeartRateMeasurementParsingTests(SimpleTestCase):
    """Decoding the standard 0x2A37 characteristic.

    Byte-level and easy to get subtly wrong - a misread flags bit silently
    turns RR-intervals into garbage, and every HRV number downstream inherits
    it. Layouts here follow the Bluetooth SIG spec.
    """

    def _parse(self, data):
        from physiology.station_sidecar import parse_hrm
        return parse_hrm(bytes(data))

    def test_uint8_rate_no_rr(self):
        # flags=0x00 -> 8-bit rate, no contact reporting, no RR
        result = self._parse([0x00, 72])
        self.assertEqual(result['bpm'], 72)
        self.assertEqual(result['ibi_ms'], [])

    def test_uint16_rate(self):
        # flags bit0 set -> rate is little-endian uint16
        result = self._parse([0x01, 0x2C, 0x01])      # 300
        self.assertEqual(result['bpm'], 300)

    def test_rr_intervals_are_converted_from_1024ths(self):
        """RR comes in 1/1024 s units, not milliseconds. Treating it as ms
        would inflate every interval by ~2.4% and quietly bias HRV."""
        # flags 0x10 -> RR present. 1024 units == exactly 1000 ms
        result = self._parse([0x10, 60, 0x00, 0x04])
        self.assertEqual(result['ibi_ms'], [1000.0])

    def test_multiple_rr_intervals_in_one_notification(self):
        # two beats since the last notify: 1024 and 512 units
        result = self._parse([0x10, 60, 0x00, 0x04, 0x00, 0x02])
        self.assertEqual(result['ibi_ms'], [1000.0, 500.0])

    def test_energy_expended_field_is_skipped(self):
        """Bit 3 inserts two bytes before the RR array; not skipping them
        would parse energy as an interval."""
        # flags 0x18 -> energy present + RR present
        result = self._parse([0x18, 60, 0xFF, 0x00, 0x00, 0x04])
        self.assertEqual(result['ibi_ms'], [1000.0])

    def test_sensor_contact_reported_when_supported(self):
        # bit2 supported, bit1 clear -> clip is off the finger
        self.assertFalse(self._parse([0x04, 60])['contact'])
        # bit2 supported, bit1 set -> contact detected
        self.assertTrue(self._parse([0x06, 60])['contact'])

    def test_contact_assumed_when_band_does_not_report_it(self):
        """Refusing every sample from a band without contact reporting would
        be worse than trusting the clip."""
        self.assertTrue(self._parse([0x00, 60])['contact'])

    def test_empty_payload_is_safe(self):
        result = self._parse([])
        self.assertIsNone(result['bpm'])
        self.assertEqual(result['ibi_ms'], [])

    def test_full_frame_uint16_contact_energy_and_rr(self):
        """Every optional field at once - the layout most likely to be
        mis-parsed."""
        data = [0x1F,                # uint16 + contact detected/supported + energy + RR
                0x50, 0x00,          # 80 bpm
                0x10, 0x00,          # energy expended
                0x00, 0x04,          # 1000 ms
                0x66, 0x03]          # 870 units -> ~849.6 ms
        result = self._parse(data)
        self.assertEqual(result['bpm'], 80)
        self.assertTrue(result['contact'])
        self.assertEqual(result['ibi_ms'][0], 1000.0)
        self.assertAlmostEqual(result['ibi_ms'][1], 849.6, places=1)


class SmallRiseTests(SimpleTestCase):
    """A steady resting heart makes the SD tiny, so a rise of two or three
    beats scores several SDs. Without an absolute floor those would be flagged
    as arousal, which is how a metric starts making claims it cannot support."""

    def test_tiny_rise_on_a_very_steady_baseline_is_not_flagged(self):
        # baseline 70 bpm, SD 0.5 -> a +2 bpm rise is 4 SDs
        window = compute(varied(30, ibi=833.0, swing=8.0))   # ~72 bpm, low HRV
        result = arousal(window, baseline_hr_mean=70.0, baseline_hr_sd=0.5,
                         baseline_rmssd=45.0)
        self.assertTrue(result.usable)
        self.assertGreater(result.hr_z, 1.0)          # unusual for this person
        self.assertLess(result.hr_delta, 5.0)         # but only by a couple of beats
        self.assertFalse(result.elevated)
        self.assertEqual(result.reason, 'rise too small to matter')

    def test_large_rise_is_still_flagged(self):
        window = compute(varied(30, ibi=700.0, swing=3.0))   # ~86 bpm
        result = arousal(window, 70.0, 0.5, 45.0)
        self.assertGreater(result.hr_delta, 5.0)
        self.assertTrue(result.elevated)

    def test_hr_delta_is_reported_in_bpm(self):
        """The readable number: z-scores mean little to an examiner."""
        window = compute(varied(30, ibi=750.0, swing=10.0))  # 80 bpm
        result = arousal(window, 70.0, 4.0, 45.0)
        self.assertAlmostEqual(result.hr_delta, 10.0, delta=1.0)
