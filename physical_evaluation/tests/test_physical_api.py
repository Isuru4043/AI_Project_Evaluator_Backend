from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.utils import timezone
from rest_framework.test import APIClient

from core.models import (
    EvaluationSession,
    ExaminerProfile,
    Project,
    ProjectExaminer,
    ProjectSubmission,
    RubricCategory,
    RubricCriteria,
    SessionRecording,
    StudentProfile,
    User,
    VivaAnswer,
    VivaQuestion,
)
from physical_evaluation.models import (
    PhysicalEvaluationRun,
    PhysicalKioskAccess,
    PhysicalProjectConfig,
    PhysicalRecordingUpload,
)
from viva_evaluator.models import SubmissionIndexStatus, VivaQuestionExtension


class PhysicalEvaluationAPITests(TestCase):
    pin = 'Room-42!'

    def setUp(self):
        self.examiner_user = User.objects.create_user(
            email='examiner@example.com',
            password='password',
            full_name='Examiner One',
            role=User.Role.EXAMINER,
        )
        self.examiner = ExaminerProfile.objects.create(
            user=self.examiner_user,
            employee_id='EMP-1',
        )
        self.student_user = User.objects.create_user(
            email='student@example.com',
            password='password',
            full_name='Student One',
            role=User.Role.STUDENT,
        )
        self.student = StudentProfile.objects.create(
            user=self.student_user,
            registration_number='REG-1',
        )
        self.project = Project.objects.create(
            project_name='Physical FYP',
            evaluation_mode=Project.EvaluationMode.PHYSICAL,
            status=Project.Status.ACTIVE,
        )
        ProjectExaminer.objects.create(
            project=self.project,
            examiner=self.examiner,
            role_in_project=ProjectExaminer.RoleInProject.LEAD,
        )
        self.config = PhysicalProjectConfig(
            project=self.project,
            location='Engineering Room 42',
            created_by=self.examiner,
        )
        self.config.set_panel_pin(self.pin)
        self.config.save()
        self.submission = ProjectSubmission.objects.create(
            project=self.project,
            student=self.student,
            report_file_url='https://files.example/report.pdf',
        )
        SubmissionIndexStatus.objects.create(
            submission=self.submission,
            status=SubmissionIndexStatus.IndexStatus.READY,
            extracted_text='A ready project report.',
        )
        category = RubricCategory.objects.create(
            project=self.project,
            category_name='Technical',
            weight_percentage=100,
        )
        self.criterion = RubricCriteria.objects.create(
            category=category,
            criteria_name='Architecture',
            max_score=10,
            questions_to_ask=1,
        )
        now = timezone.now()
        self.session = EvaluationSession.objects.create(
            project=self.project,
            student=self.student,
            submission=self.submission,
            scheduled_start=now - timedelta(minutes=5),
            scheduled_end=now + timedelta(minutes=55),
            demo_enabled=True,
            location_room='Engineering Room 42',
        )
        self.examiner_client = APIClient()
        self.examiner_client.force_authenticate(self.examiner_user)

    def open_kiosk(self):
        response = self.examiner_client.post(
            f'/api/physical/projects/{self.project.id}/kiosk/open/',
            {'pin': self.pin},
            format='json',
        )
        self.assertEqual(response.status_code, 200, response.data)
        raw_token = response.data['data']['kiosk_token']
        client = APIClient()
        client.credentials(HTTP_X_PHYSICAL_KIOSK_TOKEN=raw_token)
        return client, raw_token

    def start_recording(self, client, session=None):
        target = session or self.session
        response = client.post(
            f'/api/physical/kiosk/sessions/{target.id}/recording/start/',
            {'started_at': timezone.now().isoformat()},
            format='json',
        )
        self.assertEqual(response.status_code, 200, response.data)
        return response

    def test_project_creation_initializes_physical_configuration(self):
        response = self.examiner_client.post(
            '/api/projects/create/',
            {
                'project_name': 'New Physical Project',
                'evaluation_mode': 'physical',
                'physical_location': 'Lab A',
                'physical_panel_pin': 'Secret-123',
            },
            format='json',
        )
        self.assertEqual(response.status_code, 201, response.data)
        project = Project.objects.get(project_name='New Physical Project')
        self.assertEqual(project.evaluation_mode, Project.EvaluationMode.PHYSICAL)
        self.assertEqual(project.physical_config.location, 'Lab A')
        self.assertTrue(project.physical_config.check_panel_pin('Secret-123'))
        self.assertNotIn('physical_panel_pin', response.data['data'])

    def test_partial_identity_status_authorizes_present_group_members(self):
        run = SimpleNamespace(
            session=SimpleNamespace(group_id='group-id'),
            identity_status=PhysicalEvaluationRun.IdentityStatus.PARTIAL,
            IdentityStatus=PhysicalEvaluationRun.IdentityStatus,
        )

        self.assertTrue(PhysicalEvaluationRun.identity_authorized.fget(run))

    def test_existing_project_creation_contract_still_defaults_to_remote(self):
        response = self.examiner_client.post(
            '/api/projects/create/',
            {'project_name': 'Existing Remote Flow'},
            format='json',
        )
        self.assertEqual(response.status_code, 201, response.data)
        project = Project.objects.get(project_name='Existing Remote Flow')
        self.assertEqual(project.evaluation_mode, Project.EvaluationMode.REMOTE)
        self.assertFalse(PhysicalProjectConfig.objects.filter(project=project).exists())

    def test_kiosk_token_is_hashed_and_invalid_pin_is_rejected(self):
        denied = self.examiner_client.post(
            f'/api/physical/projects/{self.project.id}/kiosk/open/',
            {'pin': 'wrong-pin'},
            format='json',
        )
        self.assertEqual(denied.status_code, 403)

        _, raw_token = self.open_kiosk()
        access = PhysicalKioskAccess.objects.get(config=self.config, closed_at__isnull=True)
        self.assertNotEqual(access.token_digest, raw_token)
        self.assertEqual(access.token_digest, PhysicalKioskAccess.digest_token(raw_token))

    def test_student_can_self_start_demo_then_use_only_its_shared_viva(self):
        client, _ = self.open_kiosk()
        listed = client.get('/api/physical/kiosk/sessions/')
        self.assertEqual(listed.status_code, 200, listed.data)
        self.assertEqual(len(listed.data['data']['sessions']), 1)
        self.assertTrue(listed.data['data']['sessions'][0]['submission_ready'])

        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)
        self.assertIsNone(started.data['data']['recording_started_at'])
        self.assertEqual(started.data['data']['next_action'], 'start_demo')
        self.session.refresh_from_db()
        self.assertEqual(self.session.status, EvaluationSession.Status.IN_PROGRESS)
        self.assertEqual(self.session.agora_channel_name, '')

        demo_done = client.post(
            f'/api/physical/kiosk/sessions/{self.session.id}/demo/complete/'
        )
        self.assertEqual(demo_done.status_code, 200, demo_done.data)
        self.assertEqual(
            self.session.physical_run.status,
            PhysicalEvaluationRun.Status.VIVA_IN_PROGRESS,
        )

        allowed = client.get(f'/api/viva/sessions/{self.session.id}/status/')
        self.assertEqual(allowed.status_code, 200, allowed.data)

        other = EvaluationSession.objects.create(
            project=self.project,
            student=self.student,
            submission=self.submission,
            scheduled_start=timezone.now(),
            scheduled_end=timezone.now() + timedelta(hours=1),
        )
        denied = client.get(f'/api/viva/sessions/{other.id}/status/')
        self.assertEqual(denied.status_code, 403)

        # Even an ordinary logged-in user cannot bypass the physical kiosk for
        # start/answer operations on this project.
        bypass = self.examiner_client.post(
            '/api/viva/sessions/start/', {'session_id': str(self.session.id)}, format='json'
        )
        self.assertEqual(bypass.status_code, 403)

    def test_recording_upload_does_not_block_the_next_active_run(self):
        client, _ = self.open_kiosk()
        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)

        run = self.session.physical_run
        run.status = PhysicalEvaluationRun.Status.RECORDING_UPLOADING
        run.save(update_fields=['status', 'updated_at'])
        self.session.status = EvaluationSession.Status.COMPLETED
        self.session.save(update_fields=['status'])
        PhysicalRecordingUpload.objects.create(
            run=run,
            status=PhysicalRecordingUpload.Status.UPLOADING,
            uploaded_chunk_indices=[0, 1, 2],
            expected_chunks=4,
            duration_seconds=31,
        )

        project_sessions = self.examiner_client.get(
            f'/api/projects/{self.project.id}/sessions/',
        )
        self.assertEqual(project_sessions.status_code, 200, project_sessions.data)
        session_card = next(
            item for item in project_sessions.data['data']
            if str(item['id']) == str(self.session.id)
        )
        self.assertEqual(session_card['status'], EvaluationSession.Status.COMPLETED)
        self.assertEqual(session_card['recording_status'], 'uploading')

        behavior = self.examiner_client.get(
            f'/api/sessions/{self.session.id}/cv/summary/',
        )
        self.assertEqual(behavior.status_code, 200, behavior.data)
        self.assertEqual(behavior.data['data']['status'], 'recording_uploading')

        active = client.get('/api/physical/kiosk/active/')
        self.assertEqual(active.status_code, 200, active.data)
        self.assertIsNone(active.data['data'])

        next_session = EvaluationSession.objects.create(
            project=self.project,
            student=self.student,
            submission=self.submission,
            scheduled_start=timezone.now() - timedelta(minutes=1),
            scheduled_end=timezone.now() + timedelta(minutes=59),
            demo_enabled=False,
            location_room='Engineering Room 42',
        )
        next_started = client.post(
            f'/api/physical/kiosk/sessions/{next_session.id}/start/',
        )
        self.assertEqual(next_started.status_code, 200, next_started.data)
        live = client.get('/api/physical/kiosk/active/')
        self.assertEqual(live.status_code, 200, live.data)
        self.assertEqual(
            live.data['data']['session']['session_id'],
            str(next_session.id),
        )

    @patch('viva_evaluator.services.pipeline.process_answer_and_pick_next')
    def test_physical_answer_uses_existing_adaptive_scoring_pipeline(self, process_turn):
        self.session.demo_enabled = False
        self.session.save(update_fields=['demo_enabled'])
        client, _ = self.open_kiosk()
        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)

        question = VivaQuestion.objects.create(
            session=self.session,
            question_text='Why did you choose this architecture?',
            blooms_level='Analyze',
            question_order=1,
            question_source='ai',
        )
        VivaQuestionExtension.objects.create(
            question=question,
            criteria=self.criterion,
            difficulty_level='medium',
        )
        process_turn.return_value = {
            'analysis': {
                'correctness': {'score': 0.8},
                'depth': {'score': 0.7},
                'consistency': {'score': 0.9},
                'reasoning': 'Grounded answer.',
            },
            'soft_score': 0.8,
            'speech_confidence': {},
            'session_complete': True,
            'termination_reason': 'test complete',
        }

        response = client.post(
            f'/api/viva/sessions/{self.session.id}/answer/',
            {
                'question_id': str(question.id),
                'answer_text': 'Because it separates the application responsibilities.',
                'speaker_id': str(self.student.id),
            },
            format='json',
        )
        self.assertEqual(response.status_code, 200, response.data)
        self.assertTrue(response.data['session_complete'])
        process_turn.assert_called_once()
        answer = VivaAnswer.objects.get(question=question)
        self.assertEqual(float(answer.ai_answer_score), 8.0)
        self.assertEqual(answer.student, self.student)

    @patch('cv_analysis.services.runner.enqueue_cv_analysis')
    @patch('AI_Evaluator_Backend.azure_storage.upload_video_to_blob')
    def test_completed_viva_uploads_full_recording_and_unlocks_close(
        self, upload_video, enqueue_cv,
    ):
        upload_video.return_value = 'https://blob.example/physical.webm'
        self.session.demo_enabled = False
        self.session.save(update_fields=['demo_enabled'])
        client, _ = self.open_kiosk()
        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)
        self.start_recording(client)

        self.session.status = EvaluationSession.Status.COMPLETED
        self.session.save(update_fields=['status'])
        video = SimpleUploadedFile(
            'physical-session.webm', b'fake-webm-data', content_type='video/webm',
        )
        completed = client.post(
            f'/api/physical/kiosk/sessions/{self.session.id}/complete/',
            {'video_file': video},
            format='multipart',
        )
        self.assertEqual(completed.status_code, 200, completed.data)
        self.session.physical_run.refresh_from_db()
        self.assertEqual(
            self.session.physical_run.status,
            PhysicalEvaluationRun.Status.COMPLETED,
        )
        recording = SessionRecording.objects.get(session=self.session)
        self.assertEqual(recording.video_file_url, upload_video.return_value)
        self.assertIsNotNone(recording.recording_started_at)
        enqueue_cv.assert_called_once_with(self.session.id)

        closed = client.post('/api/physical/kiosk/close/', {'pin': self.pin}, format='json')
        self.assertEqual(closed.status_code, 200, closed.data)

    def test_camera_only_finish_needs_no_recording_and_unlocks_close(self):
        self.session.demo_enabled = False
        self.session.save(update_fields=['demo_enabled'])
        client, _ = self.open_kiosk()

        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)
        self.assertFalse(
            PhysicalRecordingUpload.objects.filter(run=self.session.physical_run).exists()
        )
        # Runs started before camera-only mode may still have upload metadata.
        # Finishing must clear that stale state without requiring a video.
        PhysicalRecordingUpload.objects.create(run=self.session.physical_run)

        self.session.status = EvaluationSession.Status.COMPLETED
        self.session.save(update_fields=['status'])
        finished = client.post(
            f'/api/physical/kiosk/sessions/{self.session.id}/finish/',
        )

        self.assertEqual(finished.status_code, 200, finished.data)
        self.session.physical_run.refresh_from_db()
        self.assertEqual(
            self.session.physical_run.status,
            PhysicalEvaluationRun.Status.COMPLETED,
        )
        self.assertIsNone(self.session.physical_run.recording_id)
        self.assertFalse(SessionRecording.objects.filter(session=self.session).exists())
        self.assertFalse(
            PhysicalRecordingUpload.objects.filter(run=self.session.physical_run).exists()
        )

        closed = client.post('/api/physical/kiosk/close/', {'pin': self.pin}, format='json')
        self.assertEqual(closed.status_code, 200, closed.data)

    @patch('viva_evaluator.services.pipeline.process_answer_and_pick_next')
    def test_retrying_saved_answer_replays_next_unanswered_question(
        self, process_turn,
    ):
        self.session.demo_enabled = False
        self.session.save(update_fields=['demo_enabled'])
        client, _ = self.open_kiosk()
        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)

        first_question = VivaQuestion.objects.create(
            session=self.session,
            question_text='How is the file encrypted?',
            blooms_level='Understand',
            question_order=1,
            question_source='ai',
        )
        VivaQuestionExtension.objects.create(
            question=first_question,
            criteria=self.criterion,
            difficulty_level='easy',
        )
        VivaAnswer.objects.create(
            question=first_question,
            student=self.student,
            transcribed_answer='The browser encrypts it before upload.',
            ai_answer_score=8,
        )
        next_question = VivaQuestion.objects.create(
            session=self.session,
            question_text='Where are the keys stored?',
            blooms_level='Analyze',
            question_order=2,
            question_source='ai',
        )
        VivaQuestionExtension.objects.create(
            question=next_question,
            criteria=self.criterion,
            difficulty_level='medium',
        )

        retried = client.post(
            f'/api/viva/sessions/{self.session.id}/answer/',
            {
                'question_id': str(first_question.id),
                'answer_text': 'The browser encrypts it before upload.',
                'speaker_id': str(self.student.id),
            },
            format='json',
        )

        self.assertEqual(retried.status_code, 200, retried.data)
        self.assertTrue(retried.data['duplicate_ignored'])
        self.assertEqual(
            retried.data['next_question']['question_id'],
            str(next_question.id),
        )
        self.assertEqual(VivaAnswer.objects.filter(question=first_question).count(), 1)
        process_turn.assert_not_called()

    @patch('cv_analysis.services.runner.enqueue_cv_analysis')
    @patch('AI_Evaluator_Backend.azure_storage.commit_physical_video_blocks')
    @patch('AI_Evaluator_Backend.azure_storage.stage_physical_video_block')
    def test_chunked_recording_releases_kiosk_before_azure_finalize(
        self, stage_block, commit_blocks, enqueue_cv,
    ):
        commit_blocks.return_value = 'https://blob.example/chunked-physical.webm'
        self.session.demo_enabled = False
        self.session.save(update_fields=['demo_enabled'])
        client, _ = self.open_kiosk()
        started = client.post(f'/api/physical/kiosk/sessions/{self.session.id}/start/')
        self.assertEqual(started.status_code, 200, started.data)
        self.start_recording(client)

        self.session.status = EvaluationSession.Status.COMPLETED
        self.session.save(update_fields=['status'])

        # Finalizing the evaluation is a small request and immediately changes
        # the run out of an active state, even while video blocks are pending.
        finish = client.post(
            f'/api/physical/kiosk/sessions/{self.session.id}/recording/finalize/',
            {
                'total_chunks': 2,
                'duration_seconds': 42,
                'mime_type': 'video/webm',
                'extension': 'webm',
                'defer_commit': True,
            },
            format='json',
        )
        self.assertEqual(finish.status_code, 202, finish.data)
        recording_token = finish.data['data']['upload_token']
        recording_client = APIClient()
        recording_client.credentials(
            HTTP_X_PHYSICAL_RECORDING_TOKEN=recording_token,
        )
        self.session.physical_run.refresh_from_db()
        self.assertEqual(
            self.session.physical_run.status,
            PhysicalEvaluationRun.Status.RECORDING_UPLOADING,
        )
        commit_blocks.assert_not_called()

        # Closing the kiosk no longer cancels the detached recording upload.
        close_while_uploading = client.post(
            '/api/physical/kiosk/close/', {'pin': self.pin}, format='json',
        )
        self.assertEqual(close_while_uploading.status_code, 200)
        reopen_while_uploading = self.examiner_client.post(
            f'/api/physical/projects/{self.project.id}/kiosk/open/',
            {'pin': self.pin},
            format='json',
        )
        self.assertEqual(reopen_while_uploading.status_code, 200)
        next_client = APIClient()
        next_client.credentials(
            HTTP_X_PHYSICAL_KIOSK_TOKEN=reopen_while_uploading.data['data']['kiosk_token'],
        )

        # A second student can start while the first recording continues in
        # the background; upload state no longer locks the physical kiosk.
        next_session = EvaluationSession.objects.create(
            project=self.project,
            student=self.student,
            submission=self.submission,
            scheduled_start=timezone.now() - timedelta(minutes=1),
            scheduled_end=timezone.now() + timedelta(minutes=59),
            demo_enabled=False,
            location_room='Engineering Room 42',
        )
        next_started = next_client.post(
            f'/api/physical/kiosk/sessions/{next_session.id}/start/',
        )
        self.assertEqual(next_started.status_code, 200, next_started.data)

        for index in range(2):
            chunk = SimpleUploadedFile(
                f'chunk-{index}.webm',
                f'chunk-{index}'.encode(),
                content_type='video/webm',
            )
            uploaded = recording_client.post(
                f'/api/physical/kiosk/sessions/{self.session.id}/recording/chunks/{index}/',
                {
                    'chunk': chunk,
                    'mime_type': 'video/webm',
                    'extension': 'webm',
                },
                format='multipart',
            )
            self.assertEqual(uploaded.status_code, 201, uploaded.data)

        self.session.physical_run.refresh_from_db()
        self.assertEqual(
            self.session.physical_run.status,
            PhysicalEvaluationRun.Status.COMPLETED,
        )
        upload = PhysicalRecordingUpload.objects.get(run=self.session.physical_run)
        self.assertEqual(upload.status, PhysicalRecordingUpload.Status.READY)
        self.assertEqual(upload.uploaded_chunks, 2)
        recording = SessionRecording.objects.get(id=self.session.physical_run.recording_id)
        self.assertEqual(recording.video_file_url, commit_blocks.return_value)
        self.assertEqual(recording.duration_seconds, 42)
        self.assertEqual(stage_block.call_count, 2)
        commit_blocks.assert_called_once()
        enqueue_cv.assert_called_once_with(self.session.id)
