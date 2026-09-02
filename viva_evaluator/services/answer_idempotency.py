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


def has_client_key(request) -> bool:
    """Whether the caller chose the key, rather than us deriving one.

    It decides how strict a repeat submission is treated. A client that picks
    its own key is making a promise about the payload, so reusing that key for
    different content is a contract violation worth rejecting. A key we derived
    ourselves from question and speaker carries no such promise: the same
    student pressing submit again after a failed turn legitimately sends a
    slightly different transcript, and refusing it strands them on the question.
    """
    return bool(
        request.headers.get("Idempotency-Key") or request.data.get("idempotency_key")
    )


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


def acquire_claim(
    *, session, question, speaker, idempotency_key, request_hash, strict_hash=True,
):
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

            # A worker can disappear after acquiring the claim but before it
            # persists an answer (process restart, dropped DB connection, hard
            # timeout). Once that lease has expired, binding the question
            # forever to the first transcript hash makes recovery impossible:
            # speech recognition will rarely reproduce byte-identical text.
            #
            # Replacing the request identity is safe only when the abandoned
            # attempt saved no answer. Completed claims and failed/expired
            # claims that already persisted an answer remain immutable.
            recoverable = (
                claim.status == VivaAnswerProcessingClaim.Status.FAILED
                or (
                    claim.status == VivaAnswerProcessingClaim.Status.PROCESSING
                    and claim.lease_expires_at <= now
                )
            )
            answer_was_persisted = claim.question.answers.filter(
                deduplication_key=claim.speaker_key,
            ).exists()
            # A persisted answer is immutable, so a client-chosen key may not
            # be re-pointed at different content. With a derived key it is safe
            # to retry: persistence is keyed on the same speaker and returns
            # the answer already stored rather than writing a second one, so
            # the retry only finishes the turn that failed after saving.
            if not created and recoverable and (
                not answer_was_persisted or not strict_hash
            ):
                claim.status = VivaAnswerProcessingClaim.Status.PROCESSING
                claim.idempotency_key = idempotency_key
                claim.request_hash = request_hash
                claim.owner_token = owner_token
                claim.lease_expires_at = now + timedelta(seconds=LEASE_SECONDS)
                claim.response_payload = None
                claim.response_status = 200
                claim.error_code = ""
                claim.save(
                    update_fields=[
                        "status", "idempotency_key", "request_hash",
                        "owner_token", "lease_expires_at", "response_payload",
                        "response_status", "error_code", "updated_at",
                    ]
                )
                return ClaimResult(claim, "process")

            if claim.idempotency_key != idempotency_key:
                raise IdempotencyConflict(
                    "This question already has a submission with another idempotency key."
                )
            if claim.request_hash != request_hash and strict_hash:
                raise IdempotencyConflict(
                    "The idempotency key was already used with a different answer."
                )
            # A derived key falls through: a completed turn replays its stored
            # response and the student's screen moves on, which is what a
            # duplicate submit of an already-answered question should do.
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
        if winner.idempotency_key != idempotency_key or (
            strict_hash and winner.request_hash != request_hash
        ):
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
