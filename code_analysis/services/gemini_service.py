import json
import os


class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self._model = None

        if self.api_key:
            try:
                import google.generativeai as genai
            except ImportError:
                return

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def is_enabled(self):
        return self._model is not None

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
        response = self._model.generate_content(prompt)
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
        response = self._model.generate_content(prompt)
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
