import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from django.test import override_settings

from viva_evaluator.services.stt import elevenlabs


_STT_SETTINGS = {
    "ELEVENLABS_STT_ENABLED": True,
    "ELEVENLABS_API_KEY": "server-secret",
    "ELEVENLABS_STT_MODEL_ID": "scribe_v1",
    "ELEVENLABS_STT_LANGUAGE_CODE": "",
    "ELEVENLABS_STT_TIMEOUT_SECONDS": 5,
    "ELEVENLABS_STT_MAX_AUDIO_BYTES": 1_000_000,
    "ELEVENLABS_STT_MIN_AUDIO_BYTES": 10,
}


def _ok_response(payload):
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: payload,
    )


class ElevenLabsSTTTests(TestCase):
    @override_settings(**_STT_SETTINGS)
    def test_transcription_uses_official_endpoint_and_server_side_key(self):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return _ok_response({
                "text": "  The system uses a message queue.  ",
                "language_code": "eng",
                "language_probability": 0.97,
            })

        with patch.object(elevenlabs.requests, "post", side_effect=fake_post):
            result = elevenlabs.transcribe_answer_audio(
                b"x" * 4_096, content_type="audio/webm;codecs=opus",
            )

        self.assertEqual(captured["url"], "https://api.elevenlabs.io/v1/speech-to-text")
        self.assertEqual(captured["headers"]["xi-api-key"], "server-secret")
        self.assertEqual(captured["data"]["model_id"], "scribe_v1")
        # Audio-event tags would otherwise land inside the graded answer.
        self.assertEqual(captured["data"]["tag_audio_events"], "false")
        self.assertNotIn("language_code", captured["data"])
        self.assertEqual(captured["files"]["file"][0], "answer.webm")

        self.assertEqual(result.status, "ready")
        self.assertEqual(result.text, "The system uses a message queue.")
        self.assertEqual(result.language_code, "eng")

    @override_settings(**dict(_STT_SETTINGS, ELEVENLABS_STT_LANGUAGE_CODE="eng"))
    def test_configured_language_is_pinned(self):
        with patch.object(
            elevenlabs.requests, "post", return_value=_ok_response({"text": "hello"}),
        ) as post:
            elevenlabs.transcribe_answer_audio(b"x" * 4_096, content_type="audio/mp4")

        self.assertEqual(post.call_args.kwargs["data"]["language_code"], "eng")
        self.assertEqual(post.call_args.kwargs["files"]["file"][0], "answer.mp4")

    @override_settings(**_STT_SETTINGS)
    def test_silent_clip_is_not_sent_to_the_provider(self):
        with patch.object(elevenlabs.requests, "post") as post:
            result = elevenlabs.transcribe_answer_audio(b"x" * 4, content_type="audio/webm")

        post.assert_not_called()
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.text, "")

    @override_settings(**_STT_SETTINGS)
    def test_oversized_clip_is_rejected_before_upload(self):
        with patch.object(elevenlabs.requests, "post") as post:
            result = elevenlabs.transcribe_answer_audio(
                b"x" * 1_000_001, content_type="audio/webm",
            )

        post.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error, "audio_too_large")

    @override_settings(**dict(_STT_SETTINGS, ELEVENLABS_STT_ENABLED=False))
    def test_disabled_provider_reports_status_without_calling_out(self):
        with patch.object(elevenlabs.requests, "post") as post:
            result = elevenlabs.transcribe_answer_audio(b"x" * 4_096)

        post.assert_not_called()
        self.assertEqual(result.status, "disabled")
        self.assertFalse(elevenlabs.is_enabled())

    @override_settings(**_STT_SETTINGS)
    def test_provider_failure_degrades_instead_of_raising(self):
        with patch.object(
            elevenlabs.requests, "post", side_effect=RuntimeError("boom"),
        ):
            result = elevenlabs.transcribe_answer_audio(b"x" * 4_096)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.text, "")
        self.assertEqual(result.error, "RuntimeError")
