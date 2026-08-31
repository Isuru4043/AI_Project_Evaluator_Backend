from datetime import timedelta
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings
from django.utils import timezone

from core.models import (
    EvaluationSession,
    Project,
    SessionRecording,
    StudentGroup,
)
from cv_analysis.models import CVSessionReport
from sessions_app.services.recording_finalizer import (
    RecordingFinalizationError,
    finalize_completed_online_session,
    finalize_online_recording,
)


class OnlineRecordingFinalizerTests(TestCase):
    def setUp(self):
        self.project = Project.objects.create(
            project_name="Online recording finalization",
            evaluation_mode=Project.EvaluationMode.REMOTE,
        )
        now = timezone.now()
        self.session = EvaluationSession.objects.create(
            project=self.project,
            scheduled_start=now - timedelta(minutes=20),
            scheduled_end=now + timedelta(minutes=40),
            actual_start=now - timedelta(minutes=10),
            demo_completed_at=now - timedelta(minutes=9),
            status=EvaluationSession.Status.IN_PROGRESS,
            agora_channel_name="individual-channel",
            agora_recording_resource_id="resource-1",
            agora_recording_sid="sid-1",
        )

    @patch("sessions_app.services.recording_finalizer._enqueue_analysis")
    @patch("agora_service.cloud_recording.stop_recording")
    @patch("agora_service.cloud_recording.is_enabled", return_value=True)
    def test_agora_recording_is_saved_and_analysis_is_queued(
        self, _enabled, stop_recording, enqueue_analysis,
    ):
        started_at = timezone.now() - timedelta(minutes=9)
        stop_recording.return_value = {
            "url": "https://storage.example/recordings/session.mp4",
            "started_at": started_at,
        }

        result = finalize_online_recording(self.session.id)

        self.session.refresh_from_db()
        self.assertEqual(
            self.session.status,
            EvaluationSession.Status.COMPLETED,
        )
        self.assertFalse(result.already_finalized)
        self.assertEqual(
            result.recording.video_file_url,
            "https://storage.example/recordings/session.mp4",
        )
        self.assertEqual(result.recording.recording_started_at, started_at)
        stop_recording.assert_called_once()
        enqueue_analysis.assert_called_once_with(self.session.id)

    @patch("sessions_app.services.recording_finalizer._enqueue_analysis")
    @patch("agora_service.cloud_recording.stop_recording")
    @patch("agora_service.cloud_recording.is_enabled", return_value=True)
    def test_retry_reuses_recording_without_stopping_or_creating_twice(
        self, _enabled, stop_recording, enqueue_analysis,
    ):
        cloud_result = {
            "url": "https://storage.example/recordings/session.mp4",
            "started_at": timezone.now(),
        }

        def clear_handles(owner):
            owner.agora_recording_resource_id = ""
            owner.agora_recording_sid = ""
            owner.save(update_fields=[
                "agora_recording_resource_id", "agora_recording_sid",
            ])
            return cloud_result

        stop_recording.side_effect = clear_handles

        first = finalize_online_recording(self.session.id)
        second = finalize_online_recording(self.session.id)

        self.assertFalse(first.already_finalized)
        self.assertTrue(second.already_finalized)
        self.assertEqual(
            SessionRecording.objects.filter(session=self.session).count(),
            1,
        )
        self.assertEqual(stop_recording.call_count, 1)
        self.assertEqual(enqueue_analysis.call_count, 1)

    @patch("sessions_app.services.recording_finalizer._enqueue_analysis")
    @patch("agora_service.cloud_recording.stop_recording")
    @patch("agora_service.cloud_recording.is_enabled", return_value=True)
    def test_group_sibling_uses_channel_owner_and_completes_every_member(
        self, _enabled, stop_recording, enqueue_analysis,
    ):
        group = StudentGroup.objects.create(
            project=self.project,
            group_name="Group One",
        )
        self.session.group = group
        self.session.agora_channel_name = str(self.session.id)
        self.session.save(update_fields=["group", "agora_channel_name"])
        sibling = EvaluationSession.objects.create(
            project=self.project,
            group=group,
            scheduled_start=self.session.scheduled_start,
            scheduled_end=self.session.scheduled_end,
            actual_start=self.session.actual_start,
            demo_completed_at=self.session.demo_completed_at,
            status=EvaluationSession.Status.IN_PROGRESS,
            agora_channel_name=str(self.session.id),
        )
        stop_recording.return_value = {
            "url": "https://storage.example/recordings/group.mp4",
            "started_at": timezone.now(),
        }

        result = finalize_online_recording(sibling.id)

        owner_arg = stop_recording.call_args.args[0]
        self.assertEqual(owner_arg.id, self.session.id)
        self.session.refresh_from_db()
        sibling.refresh_from_db()
        self.assertEqual(self.session.status, EvaluationSession.Status.COMPLETED)
        self.assertEqual(sibling.status, EvaluationSession.Status.COMPLETED)
        self.assertEqual(result.recording.session_id, sibling.id)
        enqueue_analysis.assert_called_once_with(sibling.id)

    @patch("sessions_app.services.recording_finalizer._enqueue_analysis")
    @patch("agora_service.cloud_recording.is_enabled", return_value=False)
    def test_uploaded_video_is_valid_fallback_when_cloud_recording_is_off(
        self, _enabled, enqueue_analysis,
    ):
        self.session.agora_recording_sid = ""
        self.session.agora_recording_resource_id = ""
        self.session.save(update_fields=[
            "agora_recording_sid", "agora_recording_resource_id",
        ])

        result = finalize_online_recording(
            self.session.id,
            fallback_video_url="https://storage.example/videos/client.webm",
        )

        self.assertEqual(
            result.recording.video_file_url,
            "https://storage.example/videos/client.webm",
        )
        enqueue_analysis.assert_called_once_with(self.session.id)

    @patch("agora_service.cloud_recording.is_enabled", return_value=True)
    def test_missing_agora_recording_creates_no_empty_recording(self, _enabled):
        self.session.agora_recording_sid = ""
        self.session.agora_recording_resource_id = ""
        self.session.save(update_fields=[
            "agora_recording_sid", "agora_recording_resource_id",
        ])

        with self.assertRaises(RecordingFinalizationError):
            finalize_online_recording(self.session.id)

        self.assertFalse(
            SessionRecording.objects.filter(session=self.session).exists()
        )

    @patch("agora_service.cloud_recording.is_enabled", return_value=True)
    def test_missing_recording_preserves_detailed_start_failure(self, _enabled):
        self.session.agora_recording_sid = ""
        self.session.agora_recording_resource_id = ""
        self.session.save(update_fields=[
            "agora_recording_sid", "agora_recording_resource_id",
        ])
        CVSessionReport.objects.create(
            session=self.session,
            status=CVSessionReport.Status.FAILED,
            error_message=(
                "Agora Cloud Recording failed to start: acquire returned HTTP "
                "400: invalid_appid. Enable Cloud Recording for the Agora project."
            ),
        )

        with self.assertRaisesRegex(
            RecordingFinalizationError,
            "invalid_appid.*Enable Cloud Recording",
        ):
            finalize_online_recording(self.session.id)

    @patch("sessions_app.services.recording_finalizer.finalize_online_recording")
    def test_automatic_failure_is_reported_without_losing_completion_payload(
        self, finalize_recording,
    ):
        finalize_recording.side_effect = RecordingFinalizationError(
            "Agora did not produce a video."
        )
        payload = {"answer_saved": True, "session_complete": True}

        returned = finalize_completed_online_session(self.session, payload)

        self.assertTrue(returned["answer_saved"])
        self.assertTrue(returned["session_complete"])
        self.assertEqual(
            returned["recording_finalization"]["status"],
            "failed",
        )
        report = CVSessionReport.objects.get(session=self.session)
        self.assertEqual(report.status, CVSessionReport.Status.FAILED)
        self.assertIn("Agora did not produce", report.error_message)

    def test_physical_session_is_never_finalized_by_agora_service(self):
        self.project.evaluation_mode = Project.EvaluationMode.PHYSICAL
        self.project.save(update_fields=["evaluation_mode"])

        with self.assertRaises(RecordingFinalizationError):
            finalize_online_recording(self.session.id)

        payload = {"answer_saved": True, "session_complete": True}
        returned = finalize_completed_online_session(self.session, payload)
        self.assertNotIn("recording_finalization", returned)

    @override_settings(
        AGORA_CLOUD_RECORDING_ENABLED=True,
        AGORA_APP_ID="test-app",
        AGORA_CUSTOMER_KEY="test-customer",
        AGORA_CUSTOMER_SECRET="test-secret",
    )
    @patch("agora_service.cloud_recording.requests.post")
    def test_failed_agora_stop_preserves_handles_for_retry(self, post):
        from agora_service.cloud_recording import stop_recording

        post.return_value.status_code = 503
        post.return_value.text = "temporarily unavailable"

        self.assertIsNone(stop_recording(self.session))

        self.session.refresh_from_db()
        self.assertEqual(self.session.agora_recording_resource_id, "resource-1")
        self.assertEqual(self.session.agora_recording_sid, "sid-1")

    @patch("AI_Evaluator_Backend.azure_storage._ensure_container")
    def test_cloud_recording_uses_microsoft_azure_vendor_code(self, ensure):
        from agora_service.cloud_recording import _storage_config

        config = _storage_config(self.session)

        self.assertEqual(config["vendor"], 5)
        ensure.assert_called_once()

    @override_settings(
        AGORA_CLOUD_RECORDING_ENABLED=True,
        AGORA_REST_BASE_URL="https://api-ap-southeast-1.agora.io",
        AGORA_APP_ID="test-app",
        AGORA_CUSTOMER_KEY="test-customer",
        AGORA_CUSTOMER_SECRET="test-secret",
    )
    @patch("agora_service.token_builder.build_rtc_token", return_value="token")
    @patch("agora_service.cloud_recording._storage_config", return_value={})
    @patch("agora_service.cloud_recording.requests.post")
    def test_cloud_recording_uses_configured_regional_endpoint(
        self, post, _storage, _token,
    ):
        from agora_service.cloud_recording import start_recording

        acquired = MagicMock(status_code=200)
        acquired.json.return_value = {"resourceId": "regional-resource"}
        started = MagicMock(status_code=200)
        started.json.return_value = {"sid": "regional-sid"}
        post.side_effect = [acquired, started]

        result = start_recording(self.session)

        self.assertEqual(result["sid"], "regional-sid")
        self.assertTrue(
            all(
                call.args[0].startswith(
                    "https://api-ap-southeast-1.agora.io/v1/apps/"
                )
                for call in post.call_args_list
            )
        )

    @override_settings(
        AGORA_CLOUD_RECORDING_ENABLED=True,
        AGORA_APP_ID="test-app",
        AGORA_CUSTOMER_KEY="test-customer",
        AGORA_CUSTOMER_SECRET="test-secret",
    )
    @patch("agora_service.cloud_recording.requests.post")
    def test_cloud_recording_start_failure_is_persisted(self, post):
        from agora_service.cloud_recording import start_recording

        post.side_effect = RuntimeError("regional endpoint timed out")

        self.assertIsNone(start_recording(self.session))
        report = CVSessionReport.objects.get(session=self.session)
        self.assertEqual(report.status, CVSessionReport.Status.FAILED)
        self.assertIn("regional endpoint timed out", report.error_message)
