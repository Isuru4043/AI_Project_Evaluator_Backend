"""Shared Bloom-level helpers for planning and generation."""


BLOOM_TO_DIFFICULTY = {
    "Remember": "easy",
    "Understand": "easy",
    "Apply": "medium",
    "Analyze": "medium",
    "Evaluate": "hard",
    "Create": "hard",
}


def bloom_to_difficulty(bloom: str) -> str:
    return BLOOM_TO_DIFFICULTY.get(bloom, "medium")

