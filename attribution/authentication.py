"""Exam-station authentication.

A physical exam station runs the CV engine as its own process (see
exam-station-cv's `--backend-url`), outside any browser session, so it has no
cookie and no kiosk lease. It authenticates with a shared secret instead,
sent as `X-Station-Token`.

The secret grants exactly one capability: posting speaker evidence and the
end-of-session artifact for a session. It cannot read anything, cannot touch
scores, and cannot act on a session that is not in progress — so a leaked
station token cannot be used to change a grade, only to add evidence an
examiner still has to accept.
"""

import hmac
import logging

from django.conf import settings
from rest_framework import authentication, exceptions

logger = logging.getLogger(__name__)

HEADER = 'HTTP_X_STATION_TOKEN'


class ExamStation:
    """Non-user principal for a station process.

    DRF's IsAuthenticated only checks `is_authenticated`, so this deliberately
    presents as authenticated while being no Django user at all — there is no
    account behind an exam station, and giving it one would hand it whatever
    that account could do.
    """

    is_authenticated = True
    is_active = True
    is_staff = False
    is_superuser = False
    is_station = True
    # Views that branch on kiosk-ness treat a station the same way: a trusted
    # room device acting for whoever is in front of it, not for a person.
    is_kiosk = True
    id = None
    pk = None

    def __str__(self):
        return 'exam-station'


class ExamStationAuthentication(authentication.BaseAuthentication):
    """Authenticates a station by shared secret."""

    def authenticate(self, request):
        token = request.META.get(HEADER, '')
        if not token:
            return None  # let the next authenticator try

        expected = getattr(settings, 'EXAM_STATION_TOKEN', '')
        if not expected:
            logger.warning(
                'Station token presented but EXAM_STATION_TOKEN is unset.'
            )
            raise exceptions.AuthenticationFailed(
                'Exam-station access is not configured on this deployment.'
            )

        if not hmac.compare_digest(str(token), str(expected)):
            raise exceptions.AuthenticationFailed('Invalid station token.')

        return (ExamStation(), None)

    def authenticate_header(self, request):
        return 'X-Station-Token'


def is_station_principal(user) -> bool:
    """True for any trusted room device: an exam-station sidecar or the kiosk.

    These are NOT Django users - the sidecar is a dataclass-like principal and
    the kiosk is a frozen dataclass with no `id`. Callers must ask this BEFORE
    running any ORM lookup against the principal, because filtering
    `ExaminerProfile.objects.filter(user=<dataclass>)` raises rather than
    returning empty, turning a permission check into a 500.

    The kiosk identifies itself by `role`, not by an `is_kiosk` flag, so both
    shapes are recognised here rather than at each call site.
    """
    if getattr(user, 'is_station', False):
        return True
    if getattr(user, 'is_kiosk', False):
        return True
    return getattr(user, 'role', None) == 'physical_kiosk'
