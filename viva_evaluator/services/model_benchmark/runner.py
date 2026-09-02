from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from .contracts import BenchmarkCase, BenchmarkResult, ModelSpec
from .parsing import parse_json_response
from .providers import ProviderAdapter, adapter_for


class ResponseValidationError(ValueError):
    pass


def validate_response(case: BenchmarkCase, text: str) -> Optional[bool]:
    if not str(text or "").strip():
        raise ResponseValidationError("Provider returned an empty final answer.")
    if case.expected_response_format == "json":
        parsed, strict = parse_json_response(text)
        if parsed is None:
            raise ResponseValidationError(
                "Expected JSON but received malformed, non-recoverable output."
            )
        return strict
    return None


def calculate_list_price(
    spec: ModelSpec,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    if (
        input_tokens is None
        or output_tokens is None
        or spec.input_price_per_million_usd is None
        or spec.output_price_per_million_usd is None
    ):
        return None
    return round(
        (
            int(input_tokens) * spec.input_price_per_million_usd
            + int(output_tokens) * spec.output_price_per_million_usd
        )
        / 1_000_000,
        8,
    )


def estimate_request_ceiling(spec: ModelSpec, case: BenchmarkCase) -> Optional[float]:
    if (
        spec.input_price_per_million_usd is None
        or spec.output_price_per_million_usd is None
    ):
        return None
    estimated_input_tokens = max(1, (len(case.system_prompt) + len(case.prompt) + 3) // 4)
    return calculate_list_price(spec, estimated_input_tokens, case.max_output_tokens)


@dataclass
class BudgetLedger:
    paid_cap_usd: float = 25.0
    paid_spend_usd: float = 0.0

    def check(self, spec: ModelSpec, case: BenchmarkCase) -> None:
        if spec.billing_mode != "paid":
            return
        ceiling = estimate_request_ceiling(spec, case)
        if ceiling is None:
            raise RuntimeError(
                f"Paid model {spec.id} has no pricing; refusing an unbounded request."
            )
        if self.paid_spend_usd + ceiling > self.paid_cap_usd:
            raise RuntimeError(
                f"Paid benchmark cap would be exceeded by {spec.id}: "
                f"${self.paid_spend_usd:.4f} spent, ${ceiling:.4f} request ceiling, "
                f"${self.paid_cap_usd:.2f} cap."
            )

    def record(self, spec: ModelSpec, list_price: Optional[float]) -> Optional[float]:
        if spec.billing_mode != "paid" or list_price is None:
            return 0.0 if spec.billing_mode != "paid" else None
        self.paid_spend_usd += list_price
        return list_price


class BenchmarkRunner:
    def __init__(
        self,
        *,
        output_path: str | Path,
        run_id: Optional[str] = None,
        budget: Optional[BudgetLedger] = None,
        adapter_factory: Callable[..., ProviderAdapter] = adapter_for,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or str(uuid.uuid4())
        self.budget = budget or BudgetLedger()
        self.adapter_factory = adapter_factory
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._restore_paid_cap_debits()

    def _restore_paid_cap_debits(self) -> None:
        """Keep a resumed result file inside the same cumulative paid cap."""
        if not self.output_path.exists():
            return
        restored = 0.0
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                    restored += float(value.get("paid_cap_debit_usd") or 0.0)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        self.budget.paid_spend_usd = max(self.budget.paid_spend_usd, restored)

    def completed_pairs(self) -> set[tuple[str, str]]:
        completed: set[tuple[str, str]] = set()
        if not self.output_path.exists():
            return completed
        with self.output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    item.get("status") == "success"
                    and int(item.get("schema_version", 0)) >= 2
                ):
                    completed.add((str(item.get("model_id")), str(item.get("case_id"))))
        return completed

    def run(
        self,
        *,
        models: Iterable[ModelSpec],
        cases: Iterable[BenchmarkCase],
        allow_paid: bool = False,
    ) -> list[BenchmarkResult]:
        completed = self.completed_pairs()
        results: list[BenchmarkResult] = []
        for case in cases:
            for spec in models:
                if not set(case.required_capabilities).issubset(spec.capabilities):
                    continue
                if (spec.id, case.case_id) in completed:
                    continue
                if spec.billing_mode == "paid" and not allow_paid:
                    raise RuntimeError(
                        f"{spec.id} is marked paid. Re-run with --allow-paid after reviewing the cap."
                    )
                self.budget.check(spec, case)
                result = self._run_one(spec, case)
                self._append(result)
                results.append(result)
        return results

    def _run_one(self, spec: ModelSpec, case: BenchmarkCase) -> BenchmarkResult:
        started = datetime.now(timezone.utc).isoformat()
        api_key = os.getenv(spec.api_key_env, "").strip()
        if not api_key:
            return self._error_result(
                spec,
                case,
                started,
                error=RuntimeError(f"Missing environment variable {spec.api_key_env}."),
            )
        adapter = self.adapter_factory(
            spec,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        try:
            call = adapter.invoke(spec, case, api_key)
            response = call.response
            list_price = calculate_list_price(
                spec,
                response.input_tokens,
                response.output_tokens,
            )
            paid_debit = self.budget.record(spec, list_price)
            try:
                strict_format_compliance = validate_response(case, response.text)
                status = "success"
                error_type = ""
                error_message = ""
            except ResponseValidationError as exc:
                strict_format_compliance = False
                status = "invalid_response"
                error_type = type(exc).__name__
                error_message = str(exc)
            return BenchmarkResult(
                schema_version=2,
                run_id=self.run_id,
                case_id=case.case_id,
                category=case.category,
                model_id=spec.id,
                provider=spec.provider,
                requested_model=spec.model,
                provider_model=response.provider_model,
                status=status,
                started_at=started,
                latency_ms=call.latency_ms,
                attempt_count=call.attempt_count,
                retry_count=max(0, call.attempt_count - 1),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                visible_output_tokens=getattr(
                    response,
                    "visible_output_tokens",
                    response.output_tokens,
                ),
                reasoning_tokens=getattr(response, "reasoning_tokens", None),
                list_price_equivalent_usd=list_price,
                paid_cap_debit_usd=paid_debit,
                actual_charged_cost_usd=None,
                billing_mode=spec.billing_mode,
                response_format_valid=status == "success",
                strict_format_compliance=strict_format_compliance,
                response_text=response.text,
                request_id=response.request_id,
                finish_reason=response.finish_reason,
                expected_response_format=case.expected_response_format,
                case_metadata=case.metadata,
                raw_usage=response.raw_usage,
                error_type=error_type,
                error_message=error_message,
            )
        except Exception as exc:
            return self._error_result(
                spec,
                case,
                started,
                error=exc,
                attempt_count=int(getattr(exc, "attempt_count", 0)),
                latency_ms=int(getattr(exc, "latency_ms", 0)),
            )

    def _error_result(
        self,
        spec,
        case,
        started,
        *,
        error,
        attempt_count=0,
        latency_ms=0,
    ):
        return BenchmarkResult(
            schema_version=2,
            run_id=self.run_id,
            case_id=case.case_id,
            category=case.category,
            model_id=spec.id,
            provider=spec.provider,
            requested_model=spec.model,
            provider_model="",
            status="error",
            started_at=started,
            latency_ms=max(0, latency_ms),
            attempt_count=max(0, attempt_count),
            retry_count=max(0, attempt_count - 1),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            visible_output_tokens=None,
            reasoning_tokens=None,
            list_price_equivalent_usd=None,
            paid_cap_debit_usd=None,
            actual_charged_cost_usd=None,
            billing_mode=spec.billing_mode,
            response_format_valid=False,
            strict_format_compliance=None,
            expected_response_format=case.expected_response_format,
            case_metadata=case.metadata,
            error_type=type(error).__name__,
            error_message=str(error)[:1000],
        )

    def _append(self, result: BenchmarkResult) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
