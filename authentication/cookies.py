"""
Helpers for issuing and clearing the HttpOnly JWT auth cookies.

The access + refresh tokens are stored in HttpOnly cookies instead of the
response body so that browser-side JavaScript can never read them (XSS-safe).
Cookie flags are environment-aware and driven by settings:

    AUTH_COOKIE_ACCESS_NAME   e.g. "access_token"
    AUTH_COOKIE_REFRESH_NAME  e.g. "refresh_token"
    AUTH_COOKIE_DOMAIN        e.g. ".vivasense.tech" in prod, None in dev
    AUTH_COOKIE_SECURE        True in prod (https), False in dev (http)
    AUTH_COOKIE_SAMESITE      "Lax" (frontend + API are same-site subdomains)
"""

from django.conf import settings


def _access_max_age() -> int:
    return int(settings.SIMPLE_JWT['ACCESS_TOKEN_LIFETIME'].total_seconds())


def _refresh_max_age() -> int:
    return int(settings.SIMPLE_JWT['REFRESH_TOKEN_LIFETIME'].total_seconds())


def set_auth_cookies(response, access: str, refresh: str | None = None):
    """Attach the access (and optionally refresh) token as HttpOnly cookies."""
    common = {
        'domain': settings.AUTH_COOKIE_DOMAIN,
        'secure': settings.AUTH_COOKIE_SECURE,
        'httponly': True,
        'samesite': settings.AUTH_COOKIE_SAMESITE,
        'path': '/',
    }

    response.set_cookie(
        settings.AUTH_COOKIE_ACCESS_NAME,
        access,
        max_age=_access_max_age(),
        **common,
    )

    if refresh is not None:
        response.set_cookie(
            settings.AUTH_COOKIE_REFRESH_NAME,
            refresh,
            max_age=_refresh_max_age(),
            **common,
        )

    return response


def clear_auth_cookies(response):
    """Delete both auth cookies (used on logout / auth failure)."""
    for name in (settings.AUTH_COOKIE_ACCESS_NAME, settings.AUTH_COOKIE_REFRESH_NAME):
        response.delete_cookie(
            name,
            domain=settings.AUTH_COOKIE_DOMAIN,
            path='/',
            samesite=settings.AUTH_COOKIE_SAMESITE,
        )
    return response
