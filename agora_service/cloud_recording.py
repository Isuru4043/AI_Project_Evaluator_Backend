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

_BASE_URL = 'https://api.agora.io/v1/apps'

# UID the recording client joins as — must not collide with real participants.
RECORDING_UID = 88888

# Agora storageConfig vendor code for Microsoft Azure.
_VENDOR_AZURE = 2


def is_enabled() -> bool:
    return getattr(settings, 'AGORA_CLOUD_RECORDING_ENABLED', False)


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
        AZURE_CONTAINER_VIDEOS,
    )

    return {
        'vendor': _VENDOR_AZURE,
        'region': int(getattr(settings, 'AGORA_RECORDING_AZURE_REGION', 0)),
        'bucket': AZURE_CONTAINER_VIDEOS,
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
            f'{_BASE_URL}/{app_id}/cloud_recording/acquire',
            json={
                'cname': channel,
                'uid': str(RECORDING_UID),
                'clientRequest': {'resourceExpiredHour': 24},
            },
            headers=headers, timeout=15,
        )
        if acquire.status_code not in (200, 201):
            logger.error('cloud_recording: acquire failed %d %s',
                         acquire.status_code, acquire.text[:400])
            return None
        resource_id = acquire.json()['resourceId']

        # 2. start recording (mix mode → single composite mp4)
        token = build_rtc_token(
            channel_name=channel, uid=RECORDING_UID, role=ROLE_PUBLISHER,
        )
        start = requests.post(
            f'{_BASE_URL}/{app_id}/cloud_recording/resourceid/{resource_id}/mode/mix/start',
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
            headers=headers, timeout=15,
        )
        if start.status_code not in (200, 201):
            logger.error('cloud_recording: start failed %d %s',
                         start.status_code, start.text[:400])
            return None
        sid = start.json()['sid']

        session.agora_recording_resource_id = resource_id
        session.agora_recording_sid = sid
        session.save(update_fields=[
            'agora_recording_resource_id', 'agora_recording_sid',
        ])
        logger.info('cloud_recording: started channel=%s sid=%s', channel, sid)
        return {'resource_id': resource_id, 'sid': sid}

    except Exception:
        logger.exception('cloud_recording: start error for session %s', session.id)
        return None


def stop_recording(session) -> Optional[str]:
    """Stop recording; return the Azure blob URL of the composite mp4, or None.

    The returned URL matches the format azure_storage upload helpers produce,
    so the CV runner's blob download works unchanged.
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
            f'{_BASE_URL}/{app_id}/cloud_recording/resourceid/{resource_id}/sid/{sid}/mode/mix/stop',
            json={
                'cname': channel,
                'uid': str(RECORDING_UID),
                'clientRequest': {},
            },
            headers=_auth_header(), timeout=30,
        )
        blob_url = None
        if resp.status_code in (200, 201):
            server_response = resp.json().get('serverResponse', {})
            file_list = server_response.get('fileList', [])
            mp4 = next(
                (f for f in file_list if str(f.get('fileName', '')).endswith('.mp4')),
                file_list[0] if file_list else None,
            )
            if mp4:
                blob_url = _blob_url_for(mp4['fileName'])
            logger.info('cloud_recording: stopped sid=%s file=%s', sid, blob_url)
        else:
            logger.error('cloud_recording: stop failed %d %s',
                         resp.status_code, resp.text[:400])

        session.agora_recording_resource_id = ''
        session.agora_recording_sid = ''
        session.save(update_fields=[
            'agora_recording_resource_id', 'agora_recording_sid',
        ])
        return blob_url

    except Exception:
        logger.exception('cloud_recording: stop error for session %s', session.id)
        return None


def _blob_url_for(file_name: str) -> str:
    """Build the same blob URL shape azure_storage upload helpers return, so
    the CV runner can download it with the account credentials."""
    from AI_Evaluator_Backend.azure_storage import (
        AZURE_ACCOUNT_NAME,
        AZURE_CONTAINER_VIDEOS,
    )

    return (
        f'https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/'
        f'{AZURE_CONTAINER_VIDEOS}/{file_name}'
    )
