"""Translate pipeline results into the stable public API response shapes."""

from typing import Dict


def persisted_validation_metadata(question) -> Dict:
    """Return stable validation metadata from a persisted question extension."""
    try:
        extension = question.extension
    except Exception:
        extension = None
    audit = dict(getattr(extension, "generation_audit", None) or {})
    validation = audit.get("validation") or {}
    critic = audit.get("critic") or {}
    tts = audit.get("tts") or {}
    cache_key = str(tts.get("cache_key") or "")
    audio_url = None
    if cache_key and tts.get("status") in {"ready", "pending"}:
        try:
            from viva_evaluator.services.tts import generate_instant_tts_signed_url
            audio_url = generate_instant_tts_signed_url(cache_key)
        except Exception:
            audio_url = None

    return {
        "validation_status": getattr(
            extension,
            "validation_status",
            validation.get("status", "not_applicable"),
        ),
        "validation_degraded": bool(
            getattr(
                extension,
                "validation_degraded",
                validation.get("degraded", False),
            )
        ),
        "degradation_reason": validation.get("degradation_reason", ""),
        "fallback_used": bool(
            getattr(
                extension,
                "fallback_used",
                validation.get("fallback_used", False),
            )
        ),
        "critic_available": critic.get("available", True),
        "critic_passed": critic.get("passed"),
        "candidate_hash": audit.get("candidate_hash", ""),
        "socratic_intent": audit.get("socratic_intent", ""),
        "source_reference_ids": audit.get("source_reference_ids", []),
        "tts_status": tts.get("status", "disabled"),
        "audio_url": audio_url,
    }


def present_resumed_session(session, question) -> Dict:
    extension = getattr(question, "extension", None)
    return {
        "message": "Session resumed.",
        "session_id": str(session.id),
        "question_id": str(question.id),
        "question_text": question.question_text,
        "blooms_level": question.blooms_level,
        "difficulty": extension.difficulty_level if extension else "medium",
        "criterion": (
            extension.criteria.criteria_name
            if extension and extension.criteria
            else (question.viva_topic_name or "")
        ),
        "question_number": question.question_order,
        **persisted_validation_metadata(question),
    }


def present_opening_question(session, question, planned, validated) -> Dict:
    tts_metadata = dict(getattr(validated, "tts_metadata", {}) or {})
    cache_key = str(tts_metadata.get("cache_key") or "")
    audio_url = None
    if cache_key and validated.tts_status in {"ready", "pending"}:
        try:
            from viva_evaluator.services.tts import generate_instant_tts_signed_url
            audio_url = generate_instant_tts_signed_url(cache_key)
        except Exception:
            audio_url = None

    return {
        "message": "Session started.",
        "session_id": str(session.id),
        "question_id": str(question.id),
        "question_text": question.question_text,
        "blooms_level": question.blooms_level,
        "difficulty": validated.difficulty,
        "criterion": planned.plan.topic.name,
        "question_number": question.question_order,
        "tier1_passed": validated.tier1_passed,
        "tier1_failures": list(validated.tier1_failures),
        "critic_passed": validated.critic_passed,
        "critic_critique": validated.critic_critique,
        "critic_scores": dict(validated.critic_scores),
        "candidate_hash": validated.candidate_hash,
        "socratic_intent": validated.socratic_intent,
        "source_reference_ids": list(validated.source_reference_ids),
        "validation_status": validated.validation_status,
        "validation_degraded": validated.validation_degraded,
        "degradation_reason": validated.degradation_reason,
        "fallback_used": validated.fallback_used,
        "critic_available": validated.critic_available,
        "tts_status": validated.tts_status,
        "audio_url": audio_url,
    }


def present_duplicate(session, next_question=None) -> Dict:
    response = {
        "answer_saved": True,
        "duplicate_ignored": True,
        "session_complete": session.status == "completed",
        "message": "This answer was already received.",
    }
    if next_question is None:
        return response

    extension = getattr(next_question, "extension", None)
    response.update(
        {
            "session_complete": False,
            "message": (
                "This answer was already received. Continuing with the next "
                "unanswered question."
            ),
            "next_question": {
                "question_id": str(next_question.id),
                "question_text": next_question.question_text,
                "blooms_level": next_question.blooms_level,
                "difficulty": (
                    extension.difficulty_level if extension else "medium"
                ),
                "criterion": (
                    extension.criteria.criteria_name
                    if extension and extension.criteria
                    else next_question.viva_topic_name
                ),
                "question_number": next_question.question_order,
                **persisted_validation_metadata(next_question),
            },
        }
    )
    return response


def build_rubric_payload(computation: Dict) -> Dict:
    analysis = computation.get("analysis") or {}
    return {
        "correctness": analysis.get("correctness", {}),
        "depth": analysis.get("depth", {}),
        "consistency": analysis.get("consistency", {}),
        "soft_score": computation.get("soft_score"),
        "reasoning": analysis.get("reasoning", ""),
        "gap_identified": analysis.get("gap_identified", ""),
        "revealed_assumption": analysis.get("revealed_assumption", ""),
        "contradicts_code_flag": analysis.get("contradicts_code_flag", False),
        "charitable": analysis.get("charitable", {"applied": False}),
        "consistency_adjustment": analysis.get(
            "consistency_adjustment",
            {"applied": False},
        ),
        "self_correction": analysis.get(
            "self_correction",
            {"applied": False},
        ),
        "fairness_routing": analysis.get(
            "fairness_routing",
            {
                "requested_checks": [],
                "routing_reasons": [],
                "llm_calls": 0,
                "max_llm_calls": 1,
            },
        ),
    }


def build_detailed_analysis(computation: Dict, rubric_payload: Dict) -> Dict:
    return {
        "rubric": rubric_payload,
        "strategy": computation.get("strategy", {}),
        "speech_confidence": computation.get("speech_confidence") or {},
        "llm_telemetry": computation.get("llm_telemetry") or {},
    }


def present_turn(computation: Dict, persisted) -> Dict:
    if computation.get("clarification"):
        return _present_clarification(computation, persisted.question)

    rubric = build_rubric_payload(computation)
    confidence = computation.get("speech_confidence") or {}
    if computation.get("session_complete"):
        return {
            "answer_saved": True,
            "session_complete": True,
            "termination_reason": computation.get("termination_reason"),
            "rubric": rubric,
            "speech_confidence": confidence,
            "message": (
                "All termination conditions satisfied — session complete."
            ),
        }

    if computation.get("paused_by_examiner"):
        return {
            "answer_saved": True,
            "session_complete": False,
            "paused_by_examiner": True,
            "rubric": rubric,
            "speech_confidence": confidence,
        }

    payload = computation["next_question_payload"]
    question_data = payload["question_data"]
    next_question = persisted.question
    tts_meta = dict(question_data.get("tts_metadata") or {})
    cache_key = str(tts_meta.get("cache_key") or "")
    audio_url = None
    tts_status = question_data.get("tts_status", "disabled")
    if cache_key and tts_status in {"ready", "pending"}:
        try:
            from viva_evaluator.services.tts import generate_instant_tts_signed_url
            audio_url = generate_instant_tts_signed_url(cache_key)
        except Exception:
            audio_url = None

    return {
        "answer_saved": True,
        "session_complete": False,
        "rubric": rubric,
        "speech_confidence": confidence,
        "strategy": {
            "bloom_level": payload["bloom_level"],
            "socratic_intent": payload["socratic_intent"],
            "p_lt": round(payload["p_lt"], 3),
            "rationale": computation.get("strategy", {}).get("rationale", ""),
        },
        "next_question": {
            "question_id": str(next_question.id),
            "question_text": next_question.question_text,
            "blooms_level": payload["bloom_level"],
            "difficulty": payload["difficulty"],
            "criterion": payload["topic"]["topic_name"],
            "question_number": next_question.question_order,
            "tier1_passed": question_data.get("tier1_passed", False),
            "tier1_failures": question_data.get("tier1_failures", []),
            "critic_passed": question_data.get("critic_passed"),
            "critic_critique": question_data.get("critic_critique", ""),
            "critic_scores": question_data.get("critic_scores", {}),
            "attempts": question_data.get("attempts", 1),
            "candidate_hash": question_data.get("candidate_hash", ""),
            "source_reference_ids": question_data.get(
                "source_reference_ids",
                [],
            ),
            "validation_status": question_data.get(
                "validation_status",
                "rejected",
            ),
            "validation_degraded": question_data.get(
                "validation_degraded",
                False,
            ),
            "degradation_reason": question_data.get(
                "degradation_reason",
                "",
            ),
            "fallback_used": question_data.get("fallback_used", False),
            "critic_available": question_data.get("critic_available", True),
            "tts_status": tts_status,
            "audio_url": audio_url,
        },
    }


def _present_clarification(computation: Dict, question) -> Dict:
    triage = computation.get("triage") or {}
    payload = computation["clarified_question_payload"]
    question_data = payload["question_data"]
    is_restate = triage.get("label") == "GARBLED_TRANSCRIPTION"
    message = (
        "I didn't catch that clearly — could you say your answer again? "
        "This was not scored."
        if is_restate
        else "It looks like the question may have been unclear. "
        "Here's a clearer version — this was not scored."
    )
    return {
        "answer_saved": True,
        "scored": False,
        "clarification": True,
        "clarification_attempt": computation.get("clarification_attempt"),
        "triage": {
            "label": triage.get("label"),
            "rationale": triage.get("rationale"),
        },
        "message": message,
        "next_question": {
            "question_id": str(question.id),
            "question_text": question.question_text,
            "blooms_level": question.blooms_level,
            "difficulty": question_data.get(
                "difficulty",
                payload["difficulty"],
            ),
            "criterion": payload["topic"]["topic_name"],
            "question_number": question.question_order,
            "is_clarification": not is_restate,
            "is_restate": is_restate,
            "validation_status": question_data.get(
                "validation_status",
                "rejected",
            ),
            "validation_degraded": question_data.get(
                "validation_degraded",
                False,
            ),
            "fallback_used": question_data.get("fallback_used", False),
            "tts_status": question_data.get("tts_status", "disabled"),
        },
    }
