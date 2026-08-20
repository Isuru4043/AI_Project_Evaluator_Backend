"""Typed failures raised at the question-safety boundary."""


class QuestionGenerationUnavailableError(RuntimeError):
    """No generated or deterministic fallback question passed Tier 1."""

    code = "safe_question_unavailable"

    def __init__(
        self,
        message: str = (
            "A safe viva question could not be generated. Please retry the "
            "request."
        ),
    ) -> None:
        super().__init__(message)
