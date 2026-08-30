from io import BytesIO
from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient

from core.models import StudentProfile, User


def sample_image(name):
    buffer = BytesIO()
    Image.new('RGB', (640, 480), color=(120, 140, 160)).save(
        buffer, format='JPEG',
    )
    return SimpleUploadedFile(name, buffer.getvalue(), content_type='image/jpeg')


class FaceEnrollmentTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email='face@example.com',
            password='password',
            full_name='Face Student',
            role=User.Role.STUDENT,
        )
        self.student = StudentProfile.objects.create(
            user=self.user,
            registration_number='FACE-1',
        )
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    @patch('AI_Evaluator_Backend.azure_storage.upload_face_photo_to_blob')
    def test_rejects_too_few_samples(self, upload):
        response = self.client.post(
            '/api/auth/me/face-photo/',
            {'photos': [sample_image('front.jpg')]},
            format='multipart',
        )
        self.assertEqual(response.status_code, 400)
        upload.assert_not_called()

    @patch('authentication.views_face._sas_or_none', side_effect=lambda url: f'signed:{url}')
    @patch('AI_Evaluator_Backend.azure_storage.upload_face_photo_to_blob')
    def test_three_samples_replace_registration_atomically(self, upload, _sign):
        urls = [f'https://blob/faces/{self.student.id}/face_{i}.jpg' for i in range(3)]
        upload.side_effect = urls
        response = self.client.post(
            '/api/auth/me/face-photo/',
            {'photos': [
                sample_image('front.jpg'),
                sample_image('left.jpg'),
                sample_image('right.jpg'),
            ]},
            format='multipart',
        )

        self.assertEqual(response.status_code, 201, response.data)
        self.student.refresh_from_db()
        self.assertEqual(self.student.face_photo_url, urls[0])
        self.assertEqual(self.student.face_photo_urls, urls)
        self.assertEqual(response.data['data']['sample_count'], 3)
        self.assertEqual(response.data['data']['registration_status'], 'complete')

    def test_legacy_primary_photo_remains_visible_as_one_sample(self):
        self.student.face_photo_url = 'https://blob/faces/legacy.jpg'
        self.student.save(update_fields=['face_photo_url'])
        self.assertEqual(
            self.student.enrollment_face_photos(),
            ['https://blob/faces/legacy.jpg'],
        )
