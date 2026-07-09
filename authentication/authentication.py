"""
Cookie-aware JWT authentication for DRF.

Extends SimpleJWT's ``JWTAuthentication`` so the access token can be read from
an HttpOnly cookie (set at login) in addition to the ``Authorization`` header.

Precedence:
    1. ``Authorization: Bearer <token>`` header (API clients, backwards compat)
    2. The HttpOnly access-token cookie (browser + SSR requests)
"""

from django.conf import settings
from rest_framework_simplejwt.authentication import JWTAuthentication


class CookieJWTAuthentication(JWTAuthentication):
    """Authenticate using the Authorization header, falling back to the cookie."""

    def authenticate(self, request):
        header = self.get_header(request)

        if header is None:
            raw_token = request.COOKIES.get(settings.AUTH_COOKIE_ACCESS_NAME)
            if not raw_token:
                return None
        else:
            raw_token = self.get_raw_token(header)
            if raw_token is None:
                return None

        validated_token = self.get_validated_token(raw_token)
        return self.get_user(validated_token), validated_token
