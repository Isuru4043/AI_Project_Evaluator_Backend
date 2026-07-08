
import base64
import logging
import requests
from typing import Optional

from django.conf import settings

logger = logging.getLogger(__name__)

# Agora STT REST API base URL
_BASE_URL = 'https://api.agora.io'


def _get_auth_header() -> dict:
    """
    Build the HTTP Basic Auth header required by Agora's REST API.
    Uses AGORA_CUSTOMER_KEY and AGORA_CUSTOMER_SECRET.
    """
    key = settings.AGORA_CUSTOMER_KEY
    secret = settings.AGORA_CUSTOMER_SECRET
    if not key or not secret:
        raise ValueError(
            'Agora REST API credentials are not configured. '
            'Set AGORA_CUSTOMER_KEY and AGORA_CUSTOMER_SECRET in .env.'
        )
    credentials = f'{key}:{secret}'
    encoded = base64.b64encode(credentials.encode()).decode()
    return {'Authorization': f'Basic {encoded}'}


def is_enabled() -> bool:
    """Check if Agora STT is enabled in settings."""
    return getattr(settings, 'AGORA_STT_ENABLED', False)


def start_stt(session) -> Optional[str]:
    """
    Start the Agora STT bot in the session's Agora channel.

    The bot joins the channel as a subscriber (listens to all audio) and
    publishes transcribed text back into the channel as data-stream
    messages (Protobuf encoded).

    Args:
        session: An ``EvaluationSession`` model instance. Must have
                 ``agora_channel_name`` already set.

    Returns:
        The STT task ID string, or None if STT is disabled / failed.
    """
    if not is_enabled():
        logger.debug('agora_stt: STT is disabled, skipping start.')
        return None

    channel = session.agora_channel_name
    if not channel:
        logger.warning('agora_stt: No channel name on session %s, cannot start STT.', session.id)
        return None

    app_id = settings.AGORA_APP_ID

    try:
        headers = _get_auth_header()
        headers['Content-Type'] = 'application/json'

        payload = {
            'languages': ['en-US'],
            'maxIdleTime': 300,          # auto-stop after 5 min silence
            'rtcConfig': {
                'channelName': channel,
                'subBotUid': str(99999),  # UID for the STT bot
            },
            'captionConfig': {
                'storage': {},  # can be configured for cloud storage
            },
        }

        url = f'{_BASE_URL}/api/v1/projects/{app_id}/join'
        response = requests.post(url, json=payload, headers=headers, timeout=15)

        if response.status_code in (200, 201):
            data = response.json()
            task_id = data.get('taskId', '')
            logger.info('agora_stt: Started STT for channel=%s task_id=%s', channel, task_id)

            # Persist the task ID so we can stop it later
            session.agora_stt_task_id = task_id
            session.save(update_fields=['agora_stt_task_id'])
            return task_id
        else:
            logger.error(
                'agora_stt: Failed to start STT. status=%d body=%s',
                response.status_code, response.text[:500],
            )
            return None

    except requests.RequestException as exc:
        logger.error('agora_stt: Network error starting STT: %s', exc)
        return None
    except Exception as exc:
        logger.error('agora_stt: Unexpected error starting STT: %s', exc)
        return None


def stop_stt(session) -> None:
    """
    Stop the running Agora STT bot for this session.

    Args:
        session: An ``EvaluationSession`` model instance with a valid
                 ``agora_stt_task_id``.
    """
    if not is_enabled():
        return

    task_id = session.agora_stt_task_id
    if not task_id:
        logger.debug('agora_stt: No STT task_id on session %s, nothing to stop.', session.id)
        return

    app_id = settings.AGORA_APP_ID

    try:
        headers = _get_auth_header()
        headers['Content-Type'] = 'application/json'

        url = f'{_BASE_URL}/api/v1/projects/{app_id}/leave'
        payload = {'taskId': task_id}
        response = requests.post(url, json=payload, headers=headers, timeout=15)

        if response.status_code in (200, 201):
            logger.info('agora_stt: Stopped STT for task_id=%s', task_id)
        else:
            logger.warning(
                'agora_stt: Stop STT returned status=%d body=%s',
                response.status_code, response.text[:500],
            )

        # Clear the task ID regardless of success
        session.agora_stt_task_id = ''
        session.save(update_fields=['agora_stt_task_id'])

    except requests.RequestException as exc:
        logger.error('agora_stt: Network error stopping STT: %s', exc)
    except Exception as exc:
        logger.error('agora_stt: Unexpected error stopping STT: %s', exc)
