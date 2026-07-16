import json
import os

from django.conf import settings
from google import genai


class GeminiService:
    def __init__(self):
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._client = None

        import logging
        _logger = logging.getLogger(__name__)
        try:
            self._client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION,
            )
        except Exception as exc:
            _logger.error("Failed to initialize Vertex AI client: %s", exc)
            raise

    def is_enabled(self):
        return self._client is not None

    def summarize_code(self, code_excerpt):
        if not self.is_enabled():
            return None

        prompt = (
            "You are a senior code reviewer. Summarize the project briefly. "
            "Focus on architecture, main components, and notable quality issues.\n\n"
            "CODE EXCERPT:\n"
            f"{code_excerpt}\n\n"
            "Return 2-3 paragraphs."
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text.strip()

    def generate_questions(self, code_excerpt, max_questions=8):
        if not self.is_enabled():
            return []

        prompt = (
            "Generate viva questions using ONLY the provided source code excerpt. "
            "Do not use SonarQube data, issue lists, or quality metrics. Focus on "
            "important source-code sections such as routes, components, services, "
            "validation, database access, auth, helpers, error handling, and data flow. "
            "Use Bloom's taxonomy (Understand, Apply, Analyze, Evaluate, Create).\n\n"
            "SOURCE CODE EXCERPT:\n"
            f"{code_excerpt}\n\n"
            f"Return a JSON array of up to {max_questions} objects with keys: "
            "question, blooms_level, source, code_reference, reasoning."
        )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        raw_text = response.text.strip()
        return _parse_questions(raw_text)


def _parse_questions(raw_text):
    try:
        start = raw_text.find("[")
        end = raw_text.rfind("]")
        if start == -1 or end == -1:
            return []
        return json.loads(raw_text[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return []

