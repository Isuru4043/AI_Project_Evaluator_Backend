"""
Azure Blob Storage helper functions for uploading reports,
videos, and audio files, plus SAS URL generation.
"""

import os
from datetime import datetime, timedelta, timezone

from azure.core.exceptions import ResourceExistsError
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions


# =============================================================================
# Azure Blob Storage Credentials
# =============================================================================
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CONNECTION_STRING = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)

# Container names
AZURE_CONTAINER_REPORTS = "reports"
AZURE_CONTAINER_VIDEOS = "videos"
AZURE_CONTAINER_AUDIOS = "audios"
AZURE_CONTAINER_FACES = "faces"


def _get_blob_service_client():
    """Return a BlobServiceClient instance."""
    if not AZURE_ACCOUNT_NAME or not AZURE_ACCOUNT_KEY:
        raise RuntimeError(
            "Azure Blob Storage credentials are not configured. "
            "Set AZURE_ACCOUNT_NAME and AZURE_ACCOUNT_KEY in your .env file."
        )
    return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)


def _ensure_container(container_name):
    """
    Ensure a container exists.  The container is kept private (no anonymous
    blob access) — use ``generate_sas_url()`` to hand out temporary read
    URLs when the frontend needs to open a file.
    """
    client = _get_blob_service_client()
    container_client = client.get_container_client(container_name)

    try:
        container_client.create_container()
    except ResourceExistsError:
        pass

    return container_client


# =============================================================================
# 1. Upload Report
# =============================================================================

def upload_report_to_blob(file, project_id, student_id=None, group_id=None):
    """
    Upload a PDF report to the reports container.

    Blob path:
      individual: <project_id>/individual/<student_id>/report.pdf
      group:      <project_id>/groups/<group_id>/report.pdf

    Returns the blob URL on success.
    Raises Exception with a message on failure.
    """
    try:
        if student_id:
            blob_path = f"{project_id}/individual/{student_id}/{file.name}"
        elif group_id:
            blob_path = f"{project_id}/groups/{group_id}/{file.name}"
        else:
            raise ValueError("Either student_id or group_id must be provided.")

        client = _get_blob_service_client()
        _ensure_container(AZURE_CONTAINER_REPORTS)

        blob_client = client.get_blob_client(
            container=AZURE_CONTAINER_REPORTS, blob=blob_path,
        )
        blob_client.upload_blob(file.read(), overwrite=True)
        url = blob_client.url
        print(f"[AZURE] Report uploaded successfully: {url}")
        return url
    except Exception as e:
        print(f"[AZURE ERROR] Report upload failed: {str(e)}")
        raise Exception(f"Report upload failed: {str(e)}")


# =============================================================================
# 2. Upload Video
# =============================================================================

def upload_video_to_blob(file, project_id, session_id):
    """
    Upload a video file to the videos container.

    Blob path: <project_id>/<session_id>/<filename>

    Returns the blob URL on success.
    """
    try:
        blob_path = f"{project_id}/{session_id}/{file.name}"
        client = _get_blob_service_client()
        _ensure_container(AZURE_CONTAINER_VIDEOS)

        blob_client = client.get_blob_client(
            container=AZURE_CONTAINER_VIDEOS, blob=blob_path,
        )
        blob_client.upload_blob(file.read(), overwrite=True)
        url = blob_client.url
        print(f"[AZURE] Video uploaded successfully: {url}")
        return url
    except Exception as e:
        print(f"[AZURE ERROR] Video upload failed: {str(e)}")
        raise Exception(f"Video upload failed: {str(e)}")


# =============================================================================
# 3. Upload Audio
# =============================================================================

def upload_audio_to_blob(file, project_id, session_id):
    """
    Upload an audio file to the audios container.

    Blob path: <project_id>/<session_id>/<filename>

    Returns the blob URL on success.
    """
    try:
        blob_path = f"{project_id}/{session_id}/{file.name}"
        client = _get_blob_service_client()
        _ensure_container(AZURE_CONTAINER_AUDIOS)

        blob_client = client.get_blob_client(
            container=AZURE_CONTAINER_AUDIOS, blob=blob_path,
        )
        blob_client.upload_blob(file.read(), overwrite=True)
        url = blob_client.url
        print(f"[AZURE] Audio uploaded successfully: {url}")
        return url
    except Exception as e:
        print(f"[AZURE ERROR] Audio upload failed: {str(e)}")
        raise Exception(f"Audio upload failed: {str(e)}")


# =============================================================================
# 3b. Upload Face Enrollment Photo
# =============================================================================

def upload_face_photo_to_blob(file, student_id):
    """
    Upload a student's enrollment face photo to the (private) faces container.

    Blob path: <student_id>/face_<uuid8>.<ext>

    The photo is biometric reference data used only to identify who is speaking
    in a group viva recording. The container is private; hand it out solely via
    short-lived ``generate_sas_url()`` links. A random filename per upload keeps
    a replaced photo from being served from any cached URL.

    Returns the blob URL on success.
    """
    import uuid as _uuid

    try:
        ext = os.path.splitext(file.name)[1].lower() or '.jpg'
        blob_path = f"{student_id}/face_{_uuid.uuid4().hex[:8]}{ext}"
        client = _get_blob_service_client()
        _ensure_container(AZURE_CONTAINER_FACES)

        blob_client = client.get_blob_client(
            container=AZURE_CONTAINER_FACES, blob=blob_path,
        )
        blob_client.upload_blob(file.read(), overwrite=True)
        url = blob_client.url
        print(f"[AZURE] Face photo uploaded successfully: {url}")
        return url
    except Exception as e:
        print(f"[AZURE ERROR] Face photo upload failed: {str(e)}")
        raise Exception(f"Face photo upload failed: {str(e)}")


# =============================================================================
# 4. Delete Blob
# =============================================================================

def delete_blob(container_name, blob_path):
    """
    Delete a blob from the given container.
    Used for cleanup if needed.
    """
    try:
        client = _get_blob_service_client()
        blob_client = client.get_blob_client(
            container=container_name, blob=blob_path,
        )
        blob_client.delete_blob()
        print(f"[AZURE] Blob deleted: {container_name}/{blob_path}")
    except Exception as e:
        print(f"[AZURE ERROR] Blob deletion failed: {str(e)}")
        raise Exception(f"Blob deletion failed: {str(e)}")


# =============================================================================
# 5. Generate SAS URL
# =============================================================================

def generate_sas_url(container_name, blob_path, expiry_hours=2):
    """
    Generate a temporary SAS URL for secure file access.
    Default expiry is 2 hours.
    Returns the full SAS URL.
    """
    try:
        sas_token = generate_blob_sas(
            account_name=AZURE_ACCOUNT_NAME,
            account_key=AZURE_ACCOUNT_KEY,
            container_name=container_name,
            blob_name=blob_path,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=expiry_hours),
        )
        sas_url = (
            f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/"
            f"{container_name}/{blob_path}?{sas_token}"
        )
        print(f"[AZURE] SAS URL generated: {sas_url[:80]}...")
        return sas_url
    except Exception as e:
        print(f"[AZURE ERROR] SAS URL generation failed: {str(e)}")
        raise Exception(f"SAS URL generation failed: {str(e)}")