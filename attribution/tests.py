"""Tests for the pure attribution logic.

The resolver and the VTT speaker parser carry the correctness burden of this
app, and neither needs a database, a camera, or a network — so they are tested
directly with scripted traces, the same way exam_cv tests decide_speaker.
"""

from datetime import datetime, timedelta, timezone

from django.test import SimpleTestCase

from agora_service.transcript_parser import parse_vtt_to_speaker_turns
from attribution.services.resolver import (
    Evidence,
    overlap_ms,
    resolve,
)

T0 = datetime(2026, 8, 20, 10, 0, 0, tzinfo=timezone.utc)
ALICE, BOB, CARA = 'alice-id', 'bob-id', 'cara-id'


def at(start_s, end_s, student, source='agora_stt', confidence=1.0):
    return Evidence(
        t_start=T0 + timedelta(seconds=start_s),
        t_end=T0 + timedelta(seconds=end_s),
        student_id=student,
        confidence=confidence,
        source=source,
    )


def window(start_s=0, end_s=30):
    return T0 + timedelta(seconds=start_s), T0 + timedelta(seconds=end_s)


class OverlapTests(SimpleTestCase):
    def test_full_containment(self):
        ws, we = window(0, 30)
        self.assertEqual(overlap_ms(at(5, 10, ALICE), ws, we), 5000)

    def test_partial_overlap_is_clipped(self):
        ws, we = window(0, 10)
        self.assertEqual(overlap_ms(at(8, 20, ALICE), ws, we), 2000)

    def test_disjoint_contributes_nothing(self):
        ws, we = window(0, 10)
        self.assertEqual(overlap_ms(at(20, 30, ALICE), ws, we), 0)


class ResolveTests(SimpleTestCase):
    def test_no_evidence_is_not_a_guess(self):
        ws, we = window()
        d = resolve(ws, we, [])
        self.assertIsNone(d.student_id)
        self.assertEqual(d.outcome, 'no_evidence')

    def test_clear_single_speaker(self):
        ws, we = window()
        d = resolve(ws, we, [at(2, 20, ALICE)])
        self.assertEqual(d.student_id, ALICE)
        self.assertEqual(d.outcome, 'attributed')
        self.assertEqual(d.share, 1.0)

    def test_dominant_speaker_wins_over_brief_interjection(self):
        ws, we = window()
        d = resolve(ws, we, [at(0, 25, ALICE), at(25, 27, BOB)])
        self.assertEqual(d.student_id, ALICE)
        self.assertEqual(d.outcome, 'attributed')

    def test_evenly_split_window_is_uncertain(self):
        """Two students speaking equally must never resolve to either."""
        ws, we = window()
        d = resolve(ws, we, [at(0, 15, ALICE), at(15, 30, BOB)])
        self.assertIsNone(d.student_id)
        self.assertEqual(d.outcome, 'uncertain')

    def test_narrow_margin_is_uncertain(self):
        ws, we = window()
        d = resolve(ws, we, [at(0, 16, ALICE), at(16, 30, BOB)])
        self.assertIsNone(d.student_id)
        self.assertEqual(d.outcome, 'uncertain')

    def test_co_speakers_recorded_on_a_collaborative_answer(self):
        ws, we = window()
        d = resolve(ws, we, [at(0, 24, ALICE), at(24, 30, BOB)])
        self.assertEqual(d.student_id, ALICE)
        self.assertIn(BOB, d.co_speakers)

    def test_manual_source_overrides_all_model_evidence(self):
        """The examiner's word beats any amount of confident machine evidence."""
        ws, we = window()
        d = resolve(ws, we, [
            at(0, 30, ALICE, source='agora_stt', confidence=1.0),
            at(0, 30, BOB, source='manual', confidence=1.0),
        ])
        self.assertEqual(d.student_id, BOB)
        self.assertEqual(d.outcome, 'manual')

    def test_unattributable_evidence_names_nobody(self):
        """Speech from an unrecognised face must not elect the only candidate."""
        ws, we = window()
        d = resolve(ws, we, [at(0, 28, None), at(0, 2, ALICE)])
        self.assertEqual(d.student_id, ALICE)
        self.assertIn('unattributable_ms', d.breakdown)
        self.assertEqual(d.breakdown['unattributable_ms'], 28000.0)

    def test_source_weight_decides_a_conflict(self):
        """Equal spans, unequal trust: the stronger source wins."""
        ws, we = window()
        d = resolve(ws, we, [
            at(0, 30, ALICE, source='agora_stt'),     # 0.90
            at(0, 30, BOB, source='submitter'),       # 0.50
        ])
        self.assertEqual(d.student_id, ALICE)

    def test_low_confidence_evidence_counts_for_less(self):
        ws, we = window()
        d = resolve(ws, we, [
            at(0, 30, ALICE, confidence=1.0),
            at(0, 30, BOB, confidence=0.1),
        ])
        self.assertEqual(d.student_id, ALICE)

    def test_evidence_outside_the_window_is_ignored(self):
        ws, we = window(0, 10)
        d = resolve(ws, we, [at(50, 60, ALICE)])
        self.assertEqual(d.outcome, 'no_evidence')

    def test_three_way_split_is_uncertain(self):
        ws, we = window()
        d = resolve(ws, we, [at(0, 10, ALICE), at(10, 20, BOB), at(20, 30, CARA)])
        self.assertIsNone(d.student_id)


class ContributionShareTests(SimpleTestCase):
    """Shares decide how marks are divided, so they carry real grade weight."""

    def test_shares_sum_to_one(self):
        ws, we = window()
        d = resolve(ws, we, [at(0, 21, ALICE), at(21, 30, BOB)])
        self.assertAlmostEqual(sum(d.shares.values()), 1.0, places=3)

    def test_dominant_speaker_gets_the_larger_share(self):
        ws, we = window()
        d = resolve(ws, we, [at(0, 24, ALICE), at(24, 30, BOB)])
        self.assertEqual(d.student_id, ALICE)
        self.assertGreater(d.shares[ALICE], d.shares[BOB])
        self.assertAlmostEqual(d.shares[ALICE], 0.8, places=1)

    def test_shares_exist_even_when_no_winner_is_named(self):
        """An answer two students genuinely shared still divides its marks,
        even though it is too close to name a dominant speaker."""
        ws, we = window()
        d = resolve(ws, we, [at(0, 15, ALICE), at(15, 30, BOB)])
        self.assertIsNone(d.student_id)
        self.assertEqual(d.outcome, 'uncertain')
        self.assertAlmostEqual(d.shares[ALICE], 0.5, places=1)
        self.assertAlmostEqual(d.shares[BOB], 0.5, places=1)

    def test_trivial_contributors_are_excluded(self):
        """A two-word interjection should not claim a slice of the marks."""
        ws, we = window()
        d = resolve(ws, we, [at(0, 29, ALICE), at(29, 30, BOB)])
        self.assertIn(ALICE, d.shares)
        self.assertNotIn(BOB, d.shares)

    def test_manual_override_takes_the_whole_answer(self):
        ws, we = window()
        d = resolve(ws, we, [
            at(0, 20, ALICE),
            at(0, 30, BOB, source='manual'),
        ])
        self.assertEqual(d.shares, {BOB: 1.0})

    def test_unknown_speaker_competes_as_a_normal_candidate(self):
        """An unenrolled student's turns must win marks like anyone else's —
        the resolver treats their pseudo-identity as just another key."""
        ws, we = window()
        unknown = 'unknown:abc-123'
        d = resolve(ws, we, [at(0, 25, unknown), at(25, 30, ALICE)])
        self.assertEqual(d.student_id, unknown)
        self.assertGreater(d.shares[unknown], 0.6)


class CandidateKeyTests(SimpleTestCase):
    """Unknown speakers ride through the resolver as namespaced keys; the
    engine has to split them back apart without confusing the two."""

    def test_student_key_round_trips(self):
        from attribution.services.engine import split_candidate

        student_id, unknown_id = split_candidate(ALICE)
        self.assertEqual(student_id, ALICE)
        self.assertIsNone(unknown_id)

    def test_unknown_key_round_trips(self):
        from attribution.services.engine import UNKNOWN_PREFIX, split_candidate

        student_id, unknown_id = split_candidate(f'{UNKNOWN_PREFIX}abc-123')
        self.assertIsNone(student_id)
        self.assertEqual(unknown_id, 'abc-123')


class UnknownTrackIdTests(SimpleTestCase):
    """The CV engine's pseudo-identity must never collide with a real id."""

    def setUp(self):
        import sys
        from pathlib import Path

        engine_src = (
            Path(__file__).resolve().parent.parent / 'exam-station-cv' / 'src'
        )
        if str(engine_src) not in sys.path:
            sys.path.insert(0, str(engine_src))

    def test_pseudo_id_is_recognisable(self):
        from exam_cv.service import is_unknown_id, unknown_track_id

        self.assertTrue(is_unknown_id(unknown_track_id(7)))

    def test_real_student_id_is_not_mistaken_for_unknown(self):
        from exam_cv.service import is_unknown_id

        self.assertFalse(is_unknown_id(ALICE))
        self.assertFalse(is_unknown_id(None))
        self.assertFalse(is_unknown_id('uncertain'))

    def test_track_id_survives_the_round_trip(self):
        """The platform parses the track back out to key an UnknownSpeaker."""
        from exam_cv.service import unknown_track_id

        pseudo = unknown_track_id(42)
        self.assertEqual(pseudo.split(':', 1)[1], '42')


class VttSpeakerTurnTests(SimpleTestCase):
    VTT = """WEBVTT

1
00:00:04.120 --> 00:00:09.480
<v 1274918203>The architecture uses a message queue.</v>

2
00:00:10.000 --> 00:00:14.500
<v 887766554>I handled the database layer.</v>

3
00:00:20.000 --> 00:00:22.000
No voice tag on this cue.
"""

    def test_extracts_speaker_and_timing(self):
        turns = parse_vtt_to_speaker_turns(self.VTT)
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[0]['speaker'], '1274918203')
        self.assertEqual(turns[0]['t_start_ms'], 4120)
        self.assertEqual(turns[0]['t_end_ms'], 9480)
        self.assertIn('message queue', turns[0]['text'])

    def test_second_speaker_is_distinct(self):
        turns = parse_vtt_to_speaker_turns(self.VTT)
        self.assertEqual(turns[1]['speaker'], '887766554')
        self.assertEqual(turns[1]['t_start_ms'], 10000)

    def test_untagged_cue_is_kept_with_no_speaker(self):
        """An untagged cue means the window was contested, not silent."""
        turns = parse_vtt_to_speaker_turns(self.VTT)
        self.assertIsNone(turns[2]['speaker'])
        self.assertIn('No voice tag', turns[2]['text'])

    def test_handles_short_mm_ss_timestamps(self):
        turns = parse_vtt_to_speaker_turns(
            "WEBVTT\n\n00:04.120 --> 00:09.480\n<v 42>Hello.</v>\n"
        )
        self.assertEqual(turns[0]['t_start_ms'], 4120)
        self.assertEqual(turns[0]['speaker'], '42')

    def test_empty_input_is_safe(self):
        self.assertEqual(parse_vtt_to_speaker_turns(''), [])

    def test_rag_chunker_still_strips_tags(self):
        """The RAG path wants prose; it must be unaffected by this change."""
        from agora_service.transcript_parser import parse_vtt_to_text

        text = parse_vtt_to_text(self.VTT)
        self.assertNotIn('<v', text)
        self.assertIn('message queue', text)


class StationAuthTests(SimpleTestCase):
    """The station secret is the only credential a headless exam station has,
    so its failure modes matter as much as its success one."""

    def _request(self, token=None):
        from django.test import RequestFactory

        headers = {'HTTP_X_STATION_TOKEN': token} if token is not None else {}
        return RequestFactory().post('/', **headers)

    def test_no_header_defers_to_the_next_authenticator(self):
        from attribution.authentication import ExamStationAuthentication

        self.assertIsNone(ExamStationAuthentication().authenticate(self._request()))

    def test_correct_token_authenticates_a_station(self):
        from attribution.authentication import ExamStationAuthentication

        with self.settings(EXAM_STATION_TOKEN='s3cret'):
            user, _ = ExamStationAuthentication().authenticate(
                self._request('s3cret')
            )
        self.assertTrue(user.is_authenticated)
        self.assertTrue(user.is_station)
        self.assertIsNone(user.id)

    def test_wrong_token_is_rejected(self):
        from rest_framework import exceptions

        from attribution.authentication import ExamStationAuthentication

        with self.settings(EXAM_STATION_TOKEN='s3cret'):
            with self.assertRaises(exceptions.AuthenticationFailed):
                ExamStationAuthentication().authenticate(self._request('wrong'))

    def test_unconfigured_deployment_rejects_every_token(self):
        """An empty setting must not become an empty-token backdoor."""
        from rest_framework import exceptions

        from attribution.authentication import ExamStationAuthentication

        with self.settings(EXAM_STATION_TOKEN=''):
            with self.assertRaises(exceptions.AuthenticationFailed):
                ExamStationAuthentication().authenticate(self._request('anything'))


class LiveEvidenceSinkTests(SimpleTestCase):
    """The sink converts engine offsets to wall-clock spans; if that mapping
    is wrong every live attribution lands in the wrong answer window."""

    def setUp(self):
        """The CV engine lives in its own source tree (and normally its own
        virtualenv), so put it on the path before importing from it."""
        import sys
        from pathlib import Path

        engine_src = (
            Path(__file__).resolve().parent.parent
            / 'exam-station-cv' / 'src'
        )
        if str(engine_src) not in sys.path:
            sys.path.insert(0, str(engine_src))

    def _sink(self):
        from exam_cv.contracts.sink import LiveEvidenceSink

        return LiveEvidenceSink(
            'http://localhost/api/sessions/x/attribution',
            session_id='x',
            t0_utc=T0,
            batch_size=99,  # never auto-flush; we inspect the queue directly
        )

    def _turn(self, start_ms, end_ms, student, confidence=0.8):
        from exam_cv.contracts.schemas import AttributionEvent

        return AttributionEvent(
            t_start_ms=start_ms,
            t_end_ms=end_ms,
            student_id=student,
            confidence=confidence,
        )

    def test_offsets_become_absolute_wall_clock_spans(self):
        sink = self._sink()
        sink.push(self._turn(5000, 9000, ALICE))
        span = sink._pending[0]
        self.assertEqual(span['student_id'], ALICE)
        self.assertEqual(
            span['t_start'], (T0 + timedelta(seconds=5)).isoformat(),
        )
        self.assertEqual(
            span['t_end'], (T0 + timedelta(seconds=9)).isoformat(),
        )

    def test_uncertain_sentinel_becomes_an_unattributed_span(self):
        """'uncertain' is the engine's way of saying it will not guess; it
        must not be persisted as a student named 'uncertain'."""
        sink = self._sink()
        sink.push(self._turn(0, 1000, 'uncertain'))
        self.assertIsNone(sink._pending[0]['student_id'])

    def test_non_attribution_events_are_ignored(self):
        """Gaze and integrity flags are advisory and never enter scoring."""
        from exam_cv.contracts.schemas import BehavioralEvent, BehavioralKind

        sink = self._sink()
        sink.push(BehavioralEvent(
            t_ms=1000, kind=BehavioralKind.GAZE_SAMPLE, student_id=ALICE,
        ))
        self.assertEqual(sink._pending, [])
