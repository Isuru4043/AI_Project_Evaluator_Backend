from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from rest_framework.test import APIClient

from core.models import ExaminerProfile, ModuleMaterial, Project, ProjectExaminer, User


class ModuleMaterialUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="module-examiner@example.com",
            password="password",
            full_name="Module Examiner",
            role=User.Role.EXAMINER,
        )
        examiner = ExaminerProfile.objects.create(user=self.user)
        self.project = Project.objects.create(project_name="Module Upload Test")
        ProjectExaminer.objects.create(
            project=self.project,
            examiner=examiner,
            role_in_project=ProjectExaminer.RoleInProject.LEAD,
        )
        self.client = APIClient()
        self.client.force_authenticate(self.user)
        self.url = f"/api/viva/projects/{self.project.id}/module-materials/upload/"

    @patch("django_q.tasks.async_task")
    @patch("django.core.files.storage.default_storage.url")
    @patch("django.core.files.storage.default_storage.save")
    def test_upload_registers_material_and_queues_indexing(
        self,
        save_mock,
        url_mock,
        async_task_mock,
    ):
        saved_path = f"module_materials/{self.project.id}/material.pdf"
        save_mock.return_value = saved_path
        url_mock.return_value = f"https://storage.example/{saved_path}"

        response = self.client.post(
            self.url,
            {"file": SimpleUploadedFile("notes.pdf", b"%PDF-1.4 test")},
            format="multipart",
        )

        self.assertEqual(response.status_code, 201, response.data)
        material = ModuleMaterial.objects.get()
        self.assertEqual(material.original_filename, "notes.pdf")
        self.assertEqual(material.processing_status, ModuleMaterial.ProcessingStatus.PENDING)
        async_task_mock.assert_called_once_with(
            "viva_evaluator.tasks.process_module_material_task",
            material.id,
        )

    @patch("django.core.files.storage.default_storage.save", side_effect=OSError("offline"))
    def test_storage_failure_returns_actionable_json_without_database_record(self, _save_mock):
        response = self.client.post(
            self.url,
            {"file": SimpleUploadedFile("notes.pdf", b"%PDF-1.4 test")},
            format="multipart",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.data["code"], "module_material_storage_failed")
        self.assertEqual(ModuleMaterial.objects.count(), 0)

    def test_unsupported_file_is_rejected_before_storage(self):
        response = self.client.post(
            self.url,
            {"file": SimpleUploadedFile("notes.txt", b"text")},
            format="multipart",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(ModuleMaterial.objects.count(), 0)
