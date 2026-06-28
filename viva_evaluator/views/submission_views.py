from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from urllib.request import urlopen

from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.serializers import (
    SubmissionUploadSerializer,
    SubmissionIndexStatusSerializer,
)

from viva_evaluator.views._helpers import (
    _resolve_session_submission,
    _difficulty_signal_from_score,
    _get_or_create_index_status,
)


class SubmissionUploadView(APIView):
    """
    POST /api/viva/submissions/upload/

    Handles report upload for both students and examiners.
    - Student logged in: auto-attaches their profile, only needs project + file
    - Examiner logged in: requires student UUID in request body
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        from core.utils.document_parser import extract_text_from_bytes

        data = request.data.copy()
        user = request.user

        # Auto-detect student if logged in user is a student
        if user.role == 'student':
            try:
                student_profile = user.student_profile
                data['student'] = str(student_profile.id)
            except Exception:
                return Response(
                    {"error": "Student profile not found for this user."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            # Examiner must supply student UUID
            if not data.get('student'):
                return Response(
                    {"error": "student UUID is required when uploading as examiner."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        serializer = SubmissionUploadSerializer(data=data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            submission = serializer.save()
            index_status = submission.index_status

            index_status.status = SubmissionIndexStatus.IndexStatus.PROCESSING
            index_status.save()

            # Read file content directly so this works with cloud-backed storage.
            with index_status.report_file.open('rb') as f:
                file_content = f.read()
            extracted_text = extract_text_from_bytes(
                file_content, index_status.report_file.name
            )

            index_status.extracted_text = extracted_text

            # =========================================================
            # Build FAISS index from the extracted text (Week 2 RAG).
            # Uses section-aware chunking + multimodal image captioning.
            # On indexing failure we still mark the submission READY so
            # the legacy text-only flow keeps working — but log it loudly.
            # =========================================================
            try:
                from viva_evaluator.services.indexing import index_report
                from viva_evaluator.services.rag.vector_store import save_index_for_submission

                index_result = index_report(file_content, enable_image_captions=True)
                chunks = index_result['chunks']
                num_chunks, _ = save_index_for_submission(submission, chunks)
                indexed_chunks = num_chunks
                images_captioned = index_result['images_captioned']
            except Exception as idx_exc:
                import logging
                logging.getLogger(__name__).exception(
                    'FAISS indexing failed for submission=%s: %s', submission.id, idx_exc,
                )
                indexed_chunks = 0
                images_captioned = 0

            index_status.status = SubmissionIndexStatus.IndexStatus.READY
            index_status.processed_at = timezone.now()
            index_status.save()

            return Response(
                {
                    "message": "Report uploaded and processed successfully.",
                    "submission_id": str(submission.id),
                    "status": "ready",
                    "characters_extracted": len(extracted_text),
                    "chunks_indexed": indexed_chunks,
                    "images_captioned": images_captioned,
                },
                status=status.HTTP_201_CREATED,
            )

        except Exception as e:
            try:
                index_status.status = SubmissionIndexStatus.IndexStatus.FAILED
                index_status.error_message = str(e)
                index_status.save()
            except Exception:
                pass
            return Response(
                {"error": f"Upload failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SubmissionStatusView(APIView):
    """
    GET /api/viva/submissions/<submission_id>/status/

    Returns the current processing status of a submission.
    Frontend checks this before allowing a session to start.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request, submission_id):
        try:
            index_status = SubmissionIndexStatus.objects.get(
                submission__id=submission_id
            )
        except SubmissionIndexStatus.DoesNotExist:
            return Response(
                {"error": "Submission not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = SubmissionIndexStatusSerializer(index_status)
        return Response(serializer.data, status=status.HTTP_200_OK)
