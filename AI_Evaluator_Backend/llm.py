"""Central Vertex AI Gemini client configuration."""

from functools import lru_cache

from django.conf import settings
from django.core.exceptions import ImproperlyConfigured


@lru_cache(maxsize=1)
def get_llm():
    """Return the shared google-genai client configured for Vertex AI.

    Authentication is resolved by Google Application Default Credentials. When
    GOOGLE_APPLICATION_CREDENTIALS is set, settings.py converts a repository-
    relative JSON path to an absolute path before this client is created.
    """
    project = settings.GOOGLE_CLOUD_PROJECT
    location = settings.GOOGLE_CLOUD_LOCATION

    if not project:
        raise ImproperlyConfigured(
            'GOOGLE_CLOUD_PROJECT must be configured before using Gemini.'
        )
    if not location:
        raise ImproperlyConfigured(
            'GOOGLE_CLOUD_LOCATION must be configured before using Gemini.'
        )

    from google import genai

    return genai.Client(
        vertexai=True,
        project=project,
        location=location,
    )
