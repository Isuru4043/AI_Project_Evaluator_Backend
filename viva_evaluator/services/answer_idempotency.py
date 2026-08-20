"""Atomic answer-submission claims and completed-response replay."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import timedelta

from django.db import IntegrityError, transaction
from django.core.serializers.json import DjangoJSONEncoder
from django.utils import timezone

from viva_evaluator.models import VivaAnswerProcessingClaim


KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
LEASE_SECONDS = 180


class IdempotencyConflict(Exception):
    pass


@dataclass(frozen=True)
class ClaimResult:
    claim: VivaAnswerProcessingClaim
    action: str  # process, replay, or in_progress


def speaker_key(speaker_id: str) -> str:
    return "group" if speaker_id == "group" else f"student:{speaker_id}"


def request_fingerprint(*, answer_text, speech_metrics, speaker_id) -> str:
    canonical = json.dumps(
        {
            "answer_text": answer_text,
            "speech_metrics": speech_metrics or {},
            "speaker_id": str(speaker_id),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_idempotency_key(request, *, question_id, speaker_id) -> str:
    supplied = request.headers.get("Idempotency-Key") or request.data.get(
        "idempotency_key"
    )
    if supplied:
        supplied = str(supplied).strip()
        if not KEY_PATTERN.fullmatch(supplied):
            raise ValueError(
                "Idempotency-Key must be 8-160 characters containing only "
                "letters, numbers, '.', '_', ':' or '-'."
            )
        return supplied
    # Backwards-compatible deterministic key: old clients still receive exactly-once
    # processing for a question/speaker pair.
    return f"answer:{question_id}:{speaker_id}"


def acquire_claim(*, session, question, speaker, idempotency_key, request_hash):
    owner_token = uuid.uuid4()
    now = timezone.now()
    defaults = {
        "session": session,
        "speaker_key": speaker,
        "idempotency_key": idempotency_key,
        "request_hash": request_hash,
        "owner_token": owner_token,
        "lease_expires_at": now + timedelta(seconds=LEASE_SECONDS),
    }
    try:
        with transaction.atomic():
            claim, created = VivaAnswerProcessingClaim.objects.get_or_create(
                question=question,
                speaker_key=speaker,
                defaults=defaults,
            )
            if not created:
                claim = VivaAnswerProcessingClaim.objects.select_for_update().get(
                    pk=claim.pk
                )
            if claim.idempotency_key != idempotency_key:
                raise IdempotencyConflict(
                    "This question already has a submission with another idempotency key."
                )
            if claim.request_hash != request_hash:
                raise IdempotencyConflict(
                    "The idempotency key was already used with a different answer."
                )
            if created:
                return ClaimResult(claim, "process")
            if claim.status == VivaAnswerProcessingClaim.Status.COMPLETED:
                return ClaimResult(claim, "replay")
            if (
                claim.status == VivaAnswerProcessingClaim.Status.PROCESSING
                and claim.lease_expires_at > now
            ):
                return ClaimResult(claim, "in_progress")

            claim.status = VivaAnswerProcessingClaim.Status.PROCESSING
            claim.owner_token = owner_token
            claim.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)
            claim.error_code = ""
            claim.save(
                update_fields=[
                    "status", "owner_token", "lease_expires_at", "error_code",
                    "updated_at",
                ]
            )
            return ClaimResult(claim, "process")
    except IntegrityError:
        # A concurrent insert won one of the unique constraints. A session key
        # attached to another logical answer is a conflict, not a retry loop.
        winner = VivaAnswerProcessingClaim.objects.filter(
            session=session,
            idempotency_key=idempotency_key,
        ).first()
        if winner and (
            winner.question_id != question.id or winner.speaker_key != speaker
        ):
            raise IdempotencyConflict(
                "The idempotency key was already used for another answer."
            )
        if winner is None:
            winner = VivaAnswerProcessingClaim.objects.filter(
                question=question,
                speaker_key=speaker,
            ).first()
        if winner is None:
            raise
        if winner.idempotency_key != idempotency_key or winner.request_hash != request_hash:
            raise IdempotencyConflict(
                "This question already has a different answer submission."
            )
        if winner.status == VivaAnswerProcessingClaim.Status.COMPLETED:
            return ClaimResult(winner, "replay")
        return ClaimResult(winner, "in_progress")


def complete_claim(claim, payload, response_status=200):
    replay_payload = json.loads(json.dumps(payload, cls=DjangoJSONEncoder))
    VivaAnswerProcessingClaim.objects.filter(
        pk=claim.pk,
        owner_token=claim.owner_token,
        status=VivaAnswerProcessingClaim.Status.PROCESSING,
    ).update(
        status=VivaAnswerProcessingClaim.Status.COMPLETED,
        response_payload=replay_payload,
        response_status=response_status,
        error_code="",
        updated_at=timezone.now(),
    )


def fail_claim(claim, error_code="pipeline_error"):
    VivaAnswerProcessingClaim.objects.filter(
        pk=claim.pk,
        owner_token=claim.owner_token,
        status=VivaAnswerProcessingClaim.Status.PROCESSING,
    ).update(
        status=VivaAnswerProcessingClaim.Status.FAILED,
        error_code=error_code,
        updated_at=timezone.now(),
    )
