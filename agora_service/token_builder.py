"""
Agora RTC Token Builder — generates temporary tokens for clients to join
a video/audio channel.

Uses the official ``agora-token`` package which implements Agora's
AccessToken2 algorithm (HMAC-SHA256 based).
"""

import hashlib
import logging
import time

from django.conf import settings

logger = logging.getLogger(__name__)

# Role constants expected by the agora-token package.
ROLE_PUBLISHER = 1    # can send audio/video
ROLE_SUBSCRIBER = 2   # can only receive


def _uid_from_user_id(user_id) -> int:
    """
    Convert a Django UUID (or any string) to a 32-bit unsigned int that
    Agora accepts as a numeric UID.

    We hash the UUID and take the lower 31 bits (Agora UIDs must be
    positive 32-bit integers, so we avoid the sign bit).
    """
    digest = hashlib.sha256(str(user_id).encode()).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF  # 31-bit positive int


def build_rtc_token(
    channel_name: str,
    uid: int,
    role: int = ROLE_PUBLISHER,
    expire_seconds: int = 86400,
) -> str:
    """
    Build an Agora RTC token for a user to join *channel_name*.

    Args:
        channel_name:   Agora channel (typically ``str(session.id)``).
        uid:            Numeric user ID (use ``_uid_from_user_id``).
        role:           ROLE_PUBLISHER or ROLE_SUBSCRIBER.
        expire_seconds: Token validity in seconds (default 24 h).

    Returns:
        Token string ready to be sent to the frontend Agora SDK.

    Raises:
        ValueError: If Agora credentials are not configured.
    """
    app_id = settings.AGORA_APP_ID
    app_certificate = settings.AGORA_APP_CERTIFICATE

    if not app_id or not app_certificate:
        raise ValueError(
            'Agora credentials are not configured. '
            'Set AGORA_APP_ID and AGORA_APP_CERTIFICATE in your .env file.'
        )

    try:
        from agora_token_builder import RtcTokenBuilder
    except ImportError:
        raise ImportError(
            'The agora-token-builder package is not installed. '
            'Run: pip install agora-token-builder'
        )

    expire_ts = int(time.time()) + expire_seconds

    # RtcTokenBuilder.buildTokenWithUid takes:
    # app_id, app_certificate, channel_name, uid, role, privilege_expired_ts
    token = RtcTokenBuilder.buildTokenWithUid(
        app_id,
        app_certificate,
        channel_name,
        uid,
        role,
        expire_ts,
    )

    logger.info(
        'agora: built RTC token for channel=%s uid=%d role=%s expires_in=%ds',
        channel_name, uid, 'pub' if role == ROLE_PUBLISHER else 'sub', expire_seconds,
    )
    return token
