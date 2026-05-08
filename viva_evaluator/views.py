from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated

from core.models import ProjectSubmission
from viva_evaluator.models import SubmissionIndexStatus
from viva_evaluator.serializers import (
    SubmissionUploadSerializer,
    SubmissionIndexStatusSerializer,
)


class SubmissionUploadView(APIView):
    """
    POST /api/viva/submissions/upload/

    Student uploads their report file here.
    Creates a ProjectSubmission and a SubmissionIndexStatus(pending).
    Actual FAISS indexing is triggered separately via /index/ endpoint.
    """
    parser_classes = [MultiPartParser, FormParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = SubmissionUploadSerializer(data=request.data)

        if serializer.is_valid():
            submission = serializer.save()
            return Response(
                {
                    "message": "Report uploaded successfully. Indexing is pending.",
                    "submission_id": str(submission.id),
                    "index_status": "pending",
                },
                status=status.HTTP_201_CREATED,
            )

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class SubmissionIndexView(APIView):
    """
    POST /api/viva/submissions/<submission_id>/index/

    Triggers FAISS indexing for a submission.
    Call this after upload, from the HPC environment where memory allows.
    """
    permission_classes = [IsAuthenticated]

    def post(self, request, submission_id):
        try:
            index_status = SubmissionIndexStatus.objects.get(
                submission__id=submission_id
            )
        except SubmissionIndexStatus.DoesNotExist:
            return Response(
                {"error": "Submission not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if index_status.status == SubmissionIndexStatus.IndexStatus.INDEXED:
            return Response(
                {"message": "Already indexed.", "submission_id": submission_id},
                status=status.HTTP_200_OK,
            )

        # Mark as indexing
        index_status.status = SubmissionIndexStatus.IndexStatus.INDEXING
        index_status.save()

        try:
            from django.conf import settings
            from core.utils.document_parser import extract_text_from_file, chunk_text
            from core.utils.vector_store import VectorStore
            import os
            from django.utils import timezone

            # Get the file path from the FileField
            file_path = index_status.report_file.path

            # Extract text
            extracted_text = extract_text_from_file(file_path)
            chunks = chunk_text(extracted_text)

            # Build and save FAISS index
            store = VectorStore()
            store.build(chunks)

            faiss_dir = os.path.join(settings.MEDIA_ROOT, 'faiss_indexes')
            index_path = store.save(faiss_dir, str(submission_id))

            # Save results
            index_status.extracted_text = extracted_text
            index_status.faiss_index_path = index_path
            index_status.status = SubmissionIndexStatus.IndexStatus.INDEXED
            index_status.indexed_at = timezone.now()
            index_status.save()

            return Response(
                {
                    "message": "Indexing complete.",
                    "submission_id": submission_id,
                    "chunks_indexed": len(chunks),
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            index_status.status = SubmissionIndexStatus.IndexStatus.FAILED
            index_status.error_message = str(e)
            index_status.save()

            return Response(
                {"error": f"Indexing failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class SubmissionStatusView(APIView):
    """
    GET /api/viva/submissions/<submission_id>/status/

    Returns the current indexing status of a submission.
    Frontend can poll this after upload to know when the session can start.
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