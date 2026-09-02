from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path

from .contracts import BenchmarkCase, ModelSpec


class BenchmarkConfigurationError(ValueError):
    pass


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[3] / "benchmarks" / "model_registry.json"


def load_model_registry(path: str | Path | None = None) -> list[ModelSpec]:
    registry_path = Path(path) if path else default_registry_path()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkConfigurationError(
            f"Model registry not found: {registry_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkConfigurationError(
            f"Invalid model registry JSON: {registry_path}: {exc}"
        ) from exc

    models = [ModelSpec.from_dict(item) for item in payload.get("models", [])]
    ids = [item.id for item in models]
    if not models:
        raise BenchmarkConfigurationError("Model registry contains no models.")
    if len(ids) != len(set(ids)):
        raise BenchmarkConfigurationError("Model registry contains duplicate IDs.")
    return models


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    dataset_path = Path(path)
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                prompt_file = str(value.get("prompt_file") or "").strip()
                if prompt_file:
                    prompt_path = (dataset_path.parent / prompt_file).resolve()
                    try:
                        prompt_path.relative_to(dataset_path.parent.resolve())
                    except ValueError as exc:
                        raise BenchmarkConfigurationError(
                            f"prompt_file escapes the dataset directory: {prompt_file}"
                        ) from exc
                    value["prompt"] = prompt_path.read_text(encoding="utf-8")
                resolved_images = []
                for image_file in value.get("image_files") or []:
                    image_reference = str(image_file).strip()
                    image_path = (dataset_path.parent / image_reference).resolve()
                    try:
                        image_path.relative_to(dataset_path.parent.resolve())
                    except ValueError as exc:
                        raise BenchmarkConfigurationError(
                            f"image_file escapes the dataset directory: {image_reference}"
                        ) from exc
                    if not image_path.is_file():
                        raise BenchmarkConfigurationError(
                            f"image_file not found: {image_reference}"
                        )
                    mime_type = mimetypes.guess_type(image_path.name)[0]
                    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
                        raise BenchmarkConfigurationError(
                            f"Unsupported benchmark image type: {image_reference}"
                        )
                    resolved_images.append({
                        "path": str(image_path),
                        "mime_type": mime_type,
                    })
                if resolved_images:
                    value["images"] = resolved_images
                case = BenchmarkCase.from_dict(value)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise BenchmarkConfigurationError(
                    f"Invalid case at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if case.case_id in seen:
                raise BenchmarkConfigurationError(
                    f"Duplicate case_id {case.case_id!r} in {dataset_path}."
                )
            seen.add(case.case_id)
            cases.append(case)
    if not cases:
        raise BenchmarkConfigurationError(f"Dataset has no cases: {dataset_path}")
    return cases


def configured_key(spec: ModelSpec) -> bool:
    return bool(os.getenv(spec.api_key_env, "").strip())
