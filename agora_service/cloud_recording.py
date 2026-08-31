"""Agora Cloud Recording — server-side recording of the viva channel.

Records the whole Agora channel (mix/composite mode) straight into Azure Blob
Storage, so a session recording exists WITHOUT relying on any student's
laptop. Mirrors the REST style of ``stt_manager.py`` (HTTP Basic auth with
AGORA_CUSTOMER_KEY / AGORA_CUSTOMER_SECRET).

Flow: acquire → start (→ resourceId + sid persisted on the session) → stop
(→ Azure blob URL of the mp4). Feature-flagged by
AGORA_CLOUD_RECORDING_ENABLED; everything fails soft so end-viva never breaks.

NOTE: Cloud Recording is a metered Agora add-on and must be enabled on the
Agora project. The Azure region code (storageConfig.region) is provider-
specific — set AGORA_RECORDING_AZURE_REGION to match your storage account's
region per Agora's region enum.
"""

import base64
import logging
from typing import Optional

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# UID the recording client joins as — must not collide with real participants.
RECORDING_UID = 88888

# Agora storageConfig vendor code for Microsoft Azure. Vendor 2 is Alibaba
# Cloud OSS; using it with Azure credentials makes recording start/upload fail.
_VENDOR_AZURE = 5


def is_enabled() -> bool:
    return getattr(settings, 'AGORA_CLOUD_RECORDING_ENABLED', False)


def _base_url() -> str:
    """Return the configured regional Agora REST endpoint."""
    root = getattr(
        settings,
        'AGORA_REST_BASE_URL',
        'https://api-ap-southeast-1.agora.io',
    ).rstrip('/')
    return f'{root}/v1/apps'


def _mark_start_failure(session, detail: str) -> None:
    """Persist the real failure before a later analysis retry obscures it."""
    try:
        from cv_analysis.models import CVSessionReport

        report, _ = CVSessionReport.objects.get_or_create(session=session)
        if report.status == CVSessionReport.Status.COMPLETED:
            return
        report.status = CVSessionReport.Status.FAILED
        report.error_message = (
            f'Agora Cloud Recording failed to start: {detail}'
        )[:2000]
        report.save(update_fields=['status', 'error_message', 'updated_at'])
    except Exception:
        logger.exception(
            'cloud_recording: could not persist start failure for session %s',
            session.id,
        )


def _actionable_agora_error(response, operation: str) -> str:
    """Return a useful operator-facing message for an Agora REST failure."""
    raw = response.text[:400]
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = {}

    reason = str(body.get('reason') or '').strip()
    if reason == 'invalid_appid':
        return (
            f'{operation} returned HTTP {response.status_code}: invalid_appid. '
            'Enable Cloud Recording for the Agora project that owns AGORA_APP_ID, '
            'and ensure AGORA_CUSTOMER_KEY/AGORA_CUSTOMER_SECRET belong to that '
            'same Agora account.'
        )
    return f'{operation} returned HTTP {response.status_code}: {raw}'


def _auth_header() -> dict:
    key = settings.AGORA_CUSTOMER_KEY
    secret = settings.AGORA_CUSTOMER_SECRET
    if not key or not secret:
        raise ValueError(
            'Agora REST credentials missing. Set AGORA_CUSTOMER_KEY and '
            'AGORA_CUSTOMER_SECRET in .env.'
        )
    encoded = base64.b64encode(f'{key}:{secret}'.encode()).decode()
    return {'Authorization': f'Basic {encoded}', 'Content-Type': 'application/json'}


def _storage_config(session) -> dict:
    from AI_Evaluator_Backend.azure_storage import (
        AZURE_ACCOUNT_KEY,
        AZURE_ACCOUNT_NAME,
        AZURE_CONTAINER_RECORDINGS,
        _ensure_container,
    )

    # Agora's servers write the file to Azure themselves using these
    # credentials — they never touch our upload helpers, and they do NOT
    # create the container. If it is missing, Agora's upload fails after the
    # session (nothing here would report it), so make sure it exists first.
    _ensure_container(AZURE_CONTAINER_RECORDINGS)

    return {
        'vendor': _VENDOR_AZURE,
        'region': int(getattr(settings, 'AGORA_RECORDING_AZURE_REGION', 0)),
        'bucket': AZURE_CONTAINER_RECORDINGS,
        'accessKey': AZURE_ACCOUNT_NAME,
        'secretKey': AZURE_ACCOUNT_KEY,
        # Blob path prefix → cloudrec/<session_id>/...
        'fileNamePrefix': ['cloudrec', str(session.id)],
    }


def start_recording(session) -> Optional[dict]:
    """Acquire + start cloud recording for the session's channel.

    Returns {'resource_id', 'sid'} on success (also persisted on the
    session), or None if disabled / failed.
    """
    if not is_enabled():
        logger.debug('cloud_recording: disabled, skipping start.')
        return None

    channel = session.agora_channel_name
    if not channel:
        logger.warning('cloud_recording: no channel on session %s.', session.id)
        return None

    app_id = settings.AGORA_APP_ID
    try:
        from agora_service.token_builder import ROLE_PUBLISHER, build_rtc_token

        headers = _auth_header()

        # 1. acquire a resource id
        acquire = requests.post(
            f'{_base_url()}/{app_id}/cloud_recording/acquire',
            json={
                'cname': channel,
                'uid': str(RECORDING_UID),
                'clientRequest': {'resourceExpiredHour': 24},
            },
            headers=headers, timeout=30,
        )
        if acquire.status_code not in (200, 201):
            detail = _actionable_agora_error(acquire, 'acquire')
            logger.error('cloud_recording: %s', detail)
            _mark_start_failure(session, detail)
            return None
        resource_id = acquire.json()['resourceId']

        # 2. start recording (mix mode → single composite mp4)
        token = build_rtc_token(
            channel_name=channel, uid=RECORDING_UID, role=ROLE_PUBLISHER,
        )
        start = requests.post(
            f'{_base_url()}/{app_id}/cloud_recording/resourceid/{resource_id}/mode/mix/start',
            json={
                'cname': channel,
                'uid': str(RECORDING_UID),
                'clientRequest': {
                    'token': token,
                    'recordingConfig': {
                        'channelType': 0,       # 0 = communication (rtc mode)
                        'streamTypes': 2,       # 2 = audio + video
                        'maxIdleTime': 300,
                        'subscribeUidGroup': 0,
                    },
                    'recordingFileConfig': {'avFileType': ['hls', 'mp4']},
                    'storageConfig': _storage_config(session),
                },
            },
            headers=headers, timeout=30,
        )
        if start.status_code not in (200, 201):
            detail = _actionable_agora_error(start, 'start')
            logger.error('cloud_recording: %s', detail)
            _mark_start_failure(session, detail)
            return None
        sid = start.json()['sid']

        session.agora_recording_resource_id = resource_id
        session.agora_recording_sid = sid
        session.save(update_fields=[
            'agora_recording_resource_id', 'agora_recording_sid',
        ])
        logger.info('cloud_recording: started channel=%s sid=%s', channel, sid)
        return {'resource_id': resource_id, 'sid': sid}

    except Exception as exc:
        logger.exception('cloud_recording: start error for session %s', session.id)
        _mark_start_failure(session, str(exc))
        return None


def stop_recording(session) -> Optional[dict]:
    """Stop recording; return {'url', 'started_at'} for the composite mp4.

    ``url`` matches the format azure_storage upload helpers produce, so the CV
    runner's blob download works unchanged. ``started_at`` is an aware datetime
    built from Agora's ``sliceStartTime`` — the wall-clock instant that maps to
    video position 00:00:00, which is what makes question timecodes seekable.
    It is None when Agora omits it.

    Returns None when nothing was recorded.
    """
    if not is_enabled():
        return None
    resource_id = session.agora_recording_resource_id
    sid = session.agora_recording_sid
    if not (resource_id and sid):
        logger.debug('cloud_recording: nothing to stop for session %s.', session.id)
        return None

    app_id = settings.AGORA_APP_ID
    channel = session.agora_channel_name
    try:
        resp = requests.post(
            f'{_base_url()}/{app_id}/cloud_recording/resourceid/{resource_id}/sid/{sid}/mode/mix/stop',
            json={
                'cname': channel,
                'uid': str(RECORDING_UID),
                'clientRequest': {},
            },
            headers=_auth_header(), timeout=30,
        )
        result = None
        stopped = False
        if resp.status_code in (200, 201):
            stopped = True
            server_response = resp.json().get('serverResponse', {})
            file_list = server_response.get('fileList', [])
            mp4 = next(
                (f for f in file_list if str(f.get('fileName', '')).endswith('.mp4')),
                file_list[0] if file_list else None,
            )
            if mp4:
                result = {
                    'url': _blob_url_for(mp4['fileName']),
                    'started_at': _slice_start_to_datetime(mp4.get('sliceStartTime')),
                }
            logger.info('cloud_recording: stopped sid=%s file=%s', sid,
                        result['url'] if result else None)
        else:
            logger.error('cloud_recording: stop failed %d %s',
                         resp.status_code, resp.text[:400])

        # Preserve the handles after an HTTP failure so an idempotent
        # finalization retry can stop the same recording. Once Agora confirms
        # the stop, clear them even if its response did not contain a file.
        if stopped:
            session.agora_recording_resource_id = ''
            session.agora_recording_sid = ''
            session.save(update_fields=[
                'agora_recording_resource_id', 'agora_recording_sid',
            ])
        return result

    except Exception:
        logger.exception('cloud_recording: stop error for session %s', session.id)
        return None


def _slice_start_to_datetime(slice_start_time) -> Optional['datetime']:
    """Agora sliceStartTime (epoch ms) → aware datetime, or None if unusable."""
    from datetime import datetime, timezone

    try:
        ms = int(slice_start_time)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


def _blob_url_for(file_name: str) -> str:
    """Build the same blob URL shape azure_storage upload helpers return, so
    the CV runner can download it with the account credentials.

    Must name the SAME container as _storage_config's bucket — this URL is
    what the analysis and the examiner's playback both resolve back to.
    """
    from AI_Evaluator_Backend.azure_storage import (
        AZURE_ACCOUNT_NAME,
        AZURE_CONTAINER_RECORDINGS,
    )

    return (
        f'https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/'
        f'{AZURE_CONTAINER_RECORDINGS}/{file_name}'
    )
