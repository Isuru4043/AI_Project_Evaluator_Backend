from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Optional

import requests

from .contracts import BenchmarkCase, ModelResponse, ModelSpec


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.attempt_count = 0
        self.latency_ms = 0


@dataclass
class AdapterCall:
    response: ModelResponse
    attempt_count: int
    latency_ms: int


def _safe_error(response: requests.Response) -> str:
    """Return a useful provider error without ever including credentials."""
    try:
        body = response.json()
    except ValueError:
        body = response.text[:500]
    return f"HTTP {response.status_code}: {str(body)[:500]}"


def _retry_after_seconds(response: requests.Response) -> float:
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(
                0.0,
                parsedate_to_datetime(value).timestamp() - time.time(),
            )
        except (TypeError, ValueError, OverflowError):
            return 0.0


class ProviderAdapter:
    def __init__(
        self,
        *,
        session: Optional[requests.Session] = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.sleeper = sleeper

    def invoke(self, spec: ModelSpec, case: BenchmarkCase, api_key: str) -> AdapterCall:
        started = time.perf_counter()
        last_error: Optional[ProviderError] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._invoke_once(spec, case, api_key)
                return AdapterCall(
                    response=response,
                    attempt_count=attempt + 1,
                    latency_ms=max(0, int((time.perf_counter() - started) * 1000)),
                )
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt >= self.max_retries:
                    exc.attempt_count = attempt + 1
                    exc.latency_ms = max(
                        0,
                        int((time.perf_counter() - started) * 1000),
                    )
                    raise
                self.sleeper(min(8.0, 0.5 * (2**attempt)))
        raise last_error or ProviderError("Provider call failed.")

    def _post_json(
        self,
        *,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> requests.Response:
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"Network error: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc
        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            error = ProviderError(
                _safe_error(response),
                status_code=response.status_code,
                retryable=retryable,
            )
            if retryable:
                delay = _retry_after_seconds(response)
                if delay:
                    self.sleeper(min(30.0, delay))
            raise error
        return response

    def _invoke_once(
        self,
        spec: ModelSpec,
        case: BenchmarkCase,
        api_key: str,
    ) -> ModelResponse:
        raise NotImplementedError


def _encoded_images(case: BenchmarkCase) -> list[tuple[str, str]]:
    return [
        (image.mime_type, base64.b64encode(Path(image.path).read_bytes()).decode("ascii"))
        for image in case.images
    ]


class OpenAICompatibleAdapter(ProviderAdapter):
    def _invoke_once(self, spec, case, api_key):
        messages = []
        if case.system_prompt:
            messages.append({"role": "system", "content": case.system_prompt})
        if case.images:
            content: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{data}"},
                }
                for mime_type, data in _encoded_images(case)
            ]
            content.append({"type": "text", "text": case.prompt})
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": case.prompt})
        response = self._post_json(
            url=spec.endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **spec.request_headers,
            },
            payload={
                "model": spec.model,
                "messages": messages,
                "max_tokens": case.max_output_tokens,
            },
        )
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError("Provider returned no choices.")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        usage = body.get("usage") or {}
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = completion_details.get("reasoning_tokens")
        visible_output_tokens = output_tokens
        if output_tokens is not None and reasoning_tokens is not None:
            visible_output_tokens = max(0, int(output_tokens) - int(reasoning_tokens))
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)
        return ModelResponse(
            text=str(content or "").strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            visible_output_tokens=visible_output_tokens,
            reasoning_tokens=reasoning_tokens,
            provider_model=str(body.get("model") or ""),
            request_id=str(body.get("id") or response.headers.get("x-request-id", "")),
            finish_reason=str(choice.get("finish_reason") or ""),
            raw_usage=dict(usage),
        )


class AnthropicCompatibleAdapter(ProviderAdapter):
    def _invoke_once(self, spec, case, api_key):
        user_content: Any = case.prompt
        if case.images:
            user_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": data,
                    },
                }
                for mime_type, data in _encoded_images(case)
            ]
            user_content.append({"type": "text", "text": case.prompt})
        payload: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": case.max_output_tokens,
            "messages": [{"role": "user", "content": user_content}],
        }
        if case.system_prompt:
            payload["system"] = case.system_prompt
        response = self._post_json(
            url=spec.endpoint,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
                **spec.request_headers,
            },
            payload=payload,
        )
        body = response.json()
        content = body.get("content") or []
        text = "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = body.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = None
        if input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)
        return ModelResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            visible_output_tokens=output_tokens,
            reasoning_tokens=None,
            provider_model=str(body.get("model") or ""),
            request_id=str(body.get("id") or response.headers.get("request-id", "")),
            finish_reason=str(body.get("stop_reason") or ""),
            raw_usage=dict(usage),
        )


class GeminiDeveloperAdapter(ProviderAdapter):
    def _invoke_once(self, spec, case, api_key):
        contents = []
        if case.system_prompt:
            contents.append({"role": "user", "parts": [{"text": case.system_prompt}]})
        user_parts = [
            {"inline_data": {"mime_type": mime_type, "data": data}}
            for mime_type, data in _encoded_images(case)
        ]
        user_parts.append({"text": case.prompt})
        contents.append({"role": "user", "parts": user_parts})
        response = self._post_json(
            url=spec.endpoint.format(model=spec.model),
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
                **spec.request_headers,
            },
            payload={
                "contents": contents,
                "generationConfig": {
                    "maxOutputTokens": case.max_output_tokens,
                },
            },
        )
        body = response.json()
        candidates = body.get("candidates") or []
        if not candidates:
            raise ProviderError("Gemini returned no candidates.")
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(
            str(part.get("text", "")) for part in parts if isinstance(part, dict)
        )
        usage = body.get("usageMetadata") or {}
        input_tokens = usage.get("promptTokenCount")
        output_tokens = usage.get("candidatesTokenCount")
        reasoning_tokens = usage.get("thoughtsTokenCount")
        visible_output_tokens = output_tokens
        if reasoning_tokens is not None:
            output_tokens = int(output_tokens or 0) + int(reasoning_tokens)
        total_tokens = usage.get("totalTokenCount")
        return ModelResponse(
            text=text.strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            visible_output_tokens=visible_output_tokens,
            reasoning_tokens=reasoning_tokens,
            provider_model=str(body.get("modelVersion") or ""),
            request_id=str(response.headers.get("x-request-id", "")),
            finish_reason=str(candidate.get("finishReason") or ""),
            raw_usage=dict(usage),
        )


class VertexGeminiAdapter(ProviderAdapter):
    """Gemini through the same Vertex AI ADC client as the live VivaSense app."""

    def _invoke_once(self, spec, case, api_key):
        del api_key  # Vertex authenticates with ADC/service-account credentials.
        from AI_Evaluator_Backend.llm import get_llm

        prompt = case.prompt
        if case.system_prompt:
            prompt = f"{case.system_prompt}\n\n{case.prompt}"
        contents: Any = prompt
        if case.images:
            from google.genai import types

            contents = [
                types.Part.from_bytes(
                    data=Path(image.path).read_bytes(),
                    mime_type=image.mime_type,
                )
                for image in case.images
            ]
            contents.append(prompt)
        try:
            response = get_llm().models.generate_content(
                model=spec.model,
                contents=contents,
            )
        except Exception as exc:
            message = str(exc)
            lowered = message.lower()
            retryable = (
                "429" in lowered
                or "resource_exhausted" in lowered
                or "rate limit" in lowered
                or any(f"{code}" in lowered for code in range(500, 600))
            )
            raise ProviderError(
                f"Vertex AI error: {type(exc).__name__}: {message[:700]}",
                retryable=retryable,
            ) from exc

        usage_metadata = getattr(response, "usage_metadata", None)

        def usage_value(*names):
            for name in names:
                value = getattr(usage_metadata, name, None)
                if value is not None:
                    return int(value)
            return None

        input_tokens = usage_value("prompt_token_count", "input_token_count")
        visible_output_tokens = usage_value(
            "candidates_token_count",
            "output_token_count",
        )
        reasoning_tokens = usage_value("thoughts_token_count")
        output_tokens = visible_output_tokens
        if reasoning_tokens is not None:
            output_tokens = int(visible_output_tokens or 0) + reasoning_tokens
        total_tokens = usage_value("total_token_count")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens

        candidates = getattr(response, "candidates", None) or []
        finish_reason = ""
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", "") or "")
        raw_usage = {}
        if usage_metadata is not None:
            for source_name, output_name in (
                ("prompt_token_count", "prompt_token_count"),
                ("candidates_token_count", "candidates_token_count"),
                ("total_token_count", "total_token_count"),
                ("thoughts_token_count", "thoughts_token_count"),
                ("cached_content_token_count", "cached_content_token_count"),
            ):
                value = getattr(usage_metadata, source_name, None)
                if value is not None:
                    raw_usage[output_name] = int(value)
        return ModelResponse(
            text=str(getattr(response, "text", "") or "").strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            visible_output_tokens=visible_output_tokens,
            reasoning_tokens=reasoning_tokens,
            provider_model=str(getattr(response, "model_version", "") or ""),
            request_id=str(getattr(response, "response_id", "") or ""),
            finish_reason=finish_reason,
            raw_usage=raw_usage,
        )


def adapter_for(spec: ModelSpec, **kwargs) -> ProviderAdapter:
    if spec.provider == "gemini":
        return GeminiDeveloperAdapter(**kwargs)
    if spec.provider == "vertex_gemini":
        return VertexGeminiAdapter(**kwargs)
    if spec.provider == "anthropic_compatible":
        return AnthropicCompatibleAdapter(**kwargs)
    if spec.provider in {
        "groq",
        "cerebras",
        "mistral",
        "deepseek",
        "glm",
        "qwen",
        "openai_compatible",
    }:
        return OpenAICompatibleAdapter(**kwargs)
    raise ValueError(f"Unsupported benchmark provider: {spec.provider}")
