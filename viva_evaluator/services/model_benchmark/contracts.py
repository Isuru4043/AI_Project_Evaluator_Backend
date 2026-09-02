from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    model: str
    api_key_env: str
    endpoint: str
    billing_mode: str = "unknown"
    input_price_per_million_usd: Optional[float] = None
    output_price_per_million_usd: Optional[float] = None
    pricing_as_of: str = ""
    pricing_source: str = ""
    enabled: bool = True
    capabilities: tuple[str, ...] = ("text",)
    request_headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelSpec":
        return cls(
            **{
                **value,
                "capabilities": tuple(value.get("capabilities") or ("text",)),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capabilities"] = list(self.capabilities)
        return value


@dataclass(frozen=True)
class BenchmarkImage:
    path: str
    mime_type: str = "image/png"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    system_prompt: str = ""
    max_output_tokens: int = 512
    expected_response_format: str = "text"
    images: tuple[BenchmarkImage, ...] = ()
    required_capabilities: tuple[str, ...] = ("text",)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BenchmarkCase":
        return cls(
            case_id=str(value["case_id"]),
            category=str(value["category"]),
            prompt=str(value["prompt"]),
            system_prompt=str(value.get("system_prompt") or ""),
            max_output_tokens=max(1, int(value.get("max_output_tokens", 512))),
            expected_response_format=str(
                value.get("expected_response_format") or "text"
            ),
            images=tuple(
                BenchmarkImage(
                    path=str(item["path"]),
                    mime_type=str(item.get("mime_type") or "image/png"),
                )
                for item in value.get("images") or []
            ),
            required_capabilities=tuple(
                value.get("required_capabilities") or ("text",)
            ),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass
class ModelResponse:
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    visible_output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    provider_model: str = ""
    request_id: str = ""
    finish_reason: str = ""
    raw_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    schema_version: int
    run_id: str
    case_id: str
    category: str
    model_id: str
    provider: str
    requested_model: str
    provider_model: str
    status: str
    started_at: str
    latency_ms: int
    attempt_count: int
    retry_count: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    visible_output_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    list_price_equivalent_usd: Optional[float]
    paid_cap_debit_usd: Optional[float]
    actual_charged_cost_usd: Optional[float]
    billing_mode: str
    response_format_valid: bool
    strict_format_compliance: Optional[bool]
    response_text: str = ""
    request_id: str = ""
    finish_reason: str = ""
    expected_response_format: str = "text"
    case_metadata: dict[str, Any] = field(default_factory=dict)
    raw_usage: dict[str, Any] = field(default_factory=dict)
    error_type: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
