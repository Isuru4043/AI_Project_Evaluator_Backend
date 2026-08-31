from dataclasses import dataclass

from django.core import signing
from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from physical_evaluation.models import PhysicalKioskAccess, PhysicalRecordingUpload


RECORDING_TOKEN_SALT = 'physical-recording-upload-v1'


def make_recording_upload_token(upload):
    return signing.dumps(
        {'upload_id': str(upload.id), 'session_id': str(upload.run.session_id)},
        salt=RECORDING_TOKEN_SALT,
        compress=True,
    )


@dataclass(frozen=True)
class PhysicalKioskPrincipal:
    """Authenticated DRF principal that has no ordinary examiner privileges."""

    access_id: object
    role: str = 'physical_kiosk'
    is_authenticated: bool = True
    is_anonymous: bool = False
    # Attribution endpoints distinguish a trusted room device from a normal
    # account through this capability flag. Without it, a valid kiosk token
    # was authenticated and then rejected by session-participant checks.
    is_kiosk: bool = True
    is_station: bool = False
    id: None = None
    pk: None = None


class PhysicalKioskAuthentication(BaseAuthentication):
    """Authenticate a kiosk through its opaque, revocable custom header."""

    header_name = 'X-Physical-Kiosk-Token'

    def authenticate(self, request):
        raw_token = request.headers.get(self.header_name)
        if not raw_token:
            return None

        digest = PhysicalKioskAccess.digest_token(raw_token)
        access = PhysicalKioskAccess.objects.select_related(
            'config__project', 'opened_by__user',
        ).filter(token_digest=digest).first()
        if access is None:
            raise AuthenticationFailed('Invalid physical kiosk token.')
        if not access.is_active:
            raise AuthenticationFailed('The physical kiosk session has expired or been closed.')

        now = timezone.now()
        PhysicalKioskAccess.objects.filter(pk=access.pk).update(last_activity_at=now)
        access.last_activity_at = now
        return PhysicalKioskPrincipal(access.id), access

    def authenticate_header(self, request):
        return self.header_name


@dataclass(frozen=True)
class PhysicalRecordingPrincipal:
    upload_id: object
    role: str = 'physical_recording_upload'
    is_authenticated: bool = True
    is_anonymous: bool = False
    id: None = None
    pk: None = None


class PhysicalRecordingAuthentication(BaseAuthentication):
    """A short-lived capability that survives closing the kiosk panel."""

    header_name = 'X-Physical-Recording-Token'
    max_age_seconds = 2 * 60 * 60

    def authenticate(self, request):
        raw_token = request.headers.get(self.header_name)
        if not raw_token:
            return None
        try:
            payload = signing.loads(
                raw_token,
                salt=RECORDING_TOKEN_SALT,
                max_age=self.max_age_seconds,
            )
            upload = PhysicalRecordingUpload.objects.select_related(
                'run__session__project',
            ).get(
                id=payload['upload_id'],
                run__session_id=payload['session_id'],
            )
        except (signing.BadSignature, KeyError, PhysicalRecordingUpload.DoesNotExist):
            raise AuthenticationFailed('Invalid or expired physical recording token.')
        return PhysicalRecordingPrincipal(upload.id), upload

    def authenticate_header(self, request):
        return self.header_name
