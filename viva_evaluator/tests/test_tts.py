import hashlib
import os
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AI_Evaluator_Backend.settings")

from django.test import override_settings

from viva_evaluator.services.tts import elevenlabs


class _MemoryStorage:
    def __init__(self):
        self.files = {}
        self.deleted = []

    def exists(self, path):
        return path in self.files

    def save(self, path, content):
        self.files[path] = content.read()
        return path

    def delete(self, path):
        self.deleted.append(path)
        self.files.pop(path, None)

    def open(self, path, _mode):
        from io import BytesIO
        return BytesIO(self.files[path])


_TTS_SETTINGS = {
    "ELEVENLABS_TTS_ENABLED": True,
    "ELEVENLABS_API_KEY": "server-secret",
    "ELEVENLABS_VOICE_ID": "voice-1",
    "ELEVENLABS_MODEL_ID": "eleven_flash_v2_5",
    "ELEVENLABS_OUTPUT_FORMAT": "mp3_44100_128",
    "ELEVENLABS_TIMEOUT_SECONDS": 5,
    "ELEVENLABS_TTS_CACHE_MAX_JOBS": 32,
}


class ElevenLabsTTSTests(TestCase):
    def setUp(self):
        with elevenlabs._LOCK:
            elevenlabs._JOBS.clear()

    def tearDown(self):
        with elevenlabs._LOCK:
            elevenlabs._JOBS.clear()

    @override_settings(**_TTS_SETTINGS)
    def test_generation_uses_official_endpoint_and_server_side_key(self):
        storage = _MemoryStorage()
        response = SimpleNamespace(
            content=b"mp3-bytes",
            raise_for_status=lambda: None,
        )
        candidate_hash = hashlib.sha256(b"question").hexdigest()
        config = elevenlabs._config()
        ticket = elevenlabs.TTSTicket(
            elevenlabs._cache_key(candidate_hash, config),
            candidate_hash,
            8,
        )

        with (
            patch.object(elevenlabs, "default_storage", storage),
            patch.object(elevenlabs.requests, "post", return_value=response) as post,
        ):
            result = elevenlabs._generate_audio(ticket, "question", config)

        self.assertEqual(result.status, "ready")
        self.assertFalse(result.cache_hit)
        self.assertEqual(storage.files[result.storage_path], b"mp3-bytes")
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs["headers"]["xi-api-key"], "server-secret")
        self.assertEqual(kwargs["json"]["model_id"], "eleven_flash_v2_5")
        self.assertTrue(post.call_args.args[0].endswith("/voice-1"))

    @override_settings(**_TTS_SETTINGS)
    def test_final_candidate_hash_discards_stale_speculation(self):
        stale_hash = hashlib.sha256(b"stale").hexdigest()
        final_hash = hashlib.sha256(b"final").hexdigest()
        stale = elevenlabs.TTSTicket("a" * 64, stale_hash, 5)
        stale_future = Future()
        stale_future.set_result(
            elevenlabs.TTSResult(
                "ready", "viva-tts/stale.mp3", "audio/mpeg", 10, False, 5
            )
        )
        with elevenlabs._LOCK:
            elevenlabs._JOBS[stale.cache_key] = elevenlabs._TTSJob(
                stale,
                stale_future,
            )

        final_ticket = elevenlabs.TTSTicket("b" * 64, final_hash, 5)
        final_future = Future()
        final_future.set_result(
            elevenlabs.TTSResult(
                "ready", "viva-tts/final.mp3", "audio/mpeg", 12, False, 5
            )
        )
        storage = _MemoryStorage()
        storage.files["viva-tts/stale.mp3"] = b"stale"

        with (
            patch.object(elevenlabs, "default_storage", storage),
            patch.object(
                elevenlabs,
                "start_speculative_tts",
                return_value=final_ticket,
            ),
        ):
            with elevenlabs._LOCK:
                elevenlabs._JOBS[final_ticket.cache_key] = elevenlabs._TTSJob(
                    final_ticket,
                    final_future,
                )
            descriptor = elevenlabs.finalize_question_tts(
                "final",
                final_hash,
                stale,
            )

        self.assertTrue(descriptor["speculative_wasted"])
        self.assertEqual(descriptor["candidate_hash"], final_hash)
        self.assertIn("viva-tts/stale.mp3", storage.deleted)
        self.assertNotIn(stale.cache_key, elevenlabs._JOBS)

    @override_settings(**_TTS_SETTINGS)
    def test_persisted_question_binding_schedules_async_audit_update(self):
        candidate_hash = hashlib.sha256(b"accepted").hexdigest()
        ticket = elevenlabs.TTSTicket("c" * 64, candidate_hash, 8)
        future = Future()
        future.set_result(
            elevenlabs.TTSResult(
                "ready", "viva-tts/accepted.mp3", "audio/mpeg", 25, False, 8
            )
        )
        with elevenlabs._LOCK:
            elevenlabs._JOBS[ticket.cache_key] = elevenlabs._TTSJob(
                ticket,
                future,
                keep=True,
            )
        question = SimpleNamespace(
            id="question-1",
            extension=SimpleNamespace(
                generation_audit={
                    "candidate_hash": candidate_hash,
                    "tts": {
                        "candidate_hash": candidate_hash,
                        "cache_key": ticket.cache_key,
                    },
                }
            ),
        )

        with patch.object(elevenlabs._EXECUTOR, "submit") as submit:
            elevenlabs.bind_question_tts_audit(question)

        submit.assert_called_once()
        self.assertIs(submit.call_args.args[0], elevenlabs._persist_tts_result)
