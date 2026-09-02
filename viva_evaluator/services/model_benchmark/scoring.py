from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .parsing import parse_json_response


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _safe_number(value: Any):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _f1(expected: set, actual: set) -> tuple[float, float, float]:
    if not expected and not actual:
        return 1.0, 1.0, 1.0
    overlap = len(expected & actual)
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _rubric_index(sections: list[dict[str, Any]]) -> tuple[dict, dict]:
    section_index = {}
    criterion_index = {}
    for section in sections:
        section_name = _normalize(section.get("name"))
        if not section_name:
            continue
        section_index[section_name] = section
        for criterion in section.get("criteria") or []:
            criterion_name = _normalize(criterion.get("name"))
            if criterion_name:
                criterion_index[(section_name, criterion_name)] = criterion
    return section_index, criterion_index


def score_rubric_structure(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    response, strict_json = parse_json_response(response_text)
    if response is None:
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not valid JSON.",
        }
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response JSON must be an object.",
        }

    gold = gold_payload["objective_checks"]
    expected_sections, expected_criteria = _rubric_index(gold.get("sections") or [])
    actual_sections, actual_criteria = _rubric_index(response.get("sections") or [])

    section_precision, section_recall, section_f1 = _f1(
        set(expected_sections), set(actual_sections)
    )
    criterion_precision, criterion_recall, criterion_f1 = _f1(
        set(expected_criteria), set(actual_criteria)
    )

    numeric_checks: list[bool] = []
    descriptor_complete = 0
    required_bands = {"weak", "satisfactory", "good", "excellent"}
    for key, expected in expected_sections.items():
        actual = actual_sections.get(key)
        if actual is None:
            numeric_checks.extend([False, False])
            continue
        numeric_checks.extend([
            _safe_number(actual.get("marks")) == _safe_number(expected.get("marks")),
            _safe_number(actual.get("criterion_weight_total_percent"))
            == _safe_number(expected.get("criterion_weight_total_percent")),
        ])
    for key, expected in expected_criteria.items():
        actual = actual_criteria.get(key)
        numeric_checks.append(
            actual is not None
            and _safe_number(actual.get("weight_percent"))
            == _safe_number(expected.get("weight_percent"))
        )
        if actual is not None:
            descriptors = actual.get("descriptors") or {}
            if all(
                isinstance(descriptors.get(band), list)
                and any(str(item).strip() for item in descriptors[band])
                for band in required_bands
            ):
                descriptor_complete += 1
    numeric_accuracy = (
        sum(numeric_checks) / len(numeric_checks) if numeric_checks else 0.0
    )

    expected_bands = {
        _normalize(item.get("name")): (
            _safe_number(item.get("minimum")),
            _safe_number(item.get("maximum")),
            _normalize(item.get("source_grade_label")),
        )
        for item in gold.get("performance_bands") or []
    }
    actual_bands = {
        _normalize(item.get("name")): (
            _safe_number(item.get("minimum")),
            _safe_number(item.get("maximum")),
            _normalize(item.get("source_grade_label")),
        )
        for item in response.get("performance_bands") or []
        if _normalize(item.get("name"))
    }
    band_accuracy = (
        sum(actual_bands.get(key) == value for key, value in expected_bands.items())
        / len(expected_bands)
        if expected_bands
        else 1.0
    )
    descriptor_completeness = (
        descriptor_complete / len(expected_criteria) if expected_criteria else 1.0
    )
    represented_marks_correct = (
        _safe_number(response.get("represented_marks_total"))
        == _safe_number(gold.get("represented_marks_total"))
    )

    overall = 100 * (
        0.05
        + 0.15 * section_f1
        + 0.20 * criterion_f1
        + 0.20 * numeric_accuracy
        + 0.15 * band_accuracy
        + 0.20 * descriptor_completeness
        + 0.05 * float(represented_marks_correct)
    )
    return {
        "scorer_version": 1,
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "overall_score": round(overall, 2),
        "section_precision": round(section_precision, 4),
        "section_recall": round(section_recall, 4),
        "section_f1": round(section_f1, 4),
        "criterion_precision": round(criterion_precision, 4),
        "criterion_recall": round(criterion_recall, 4),
        "criterion_f1": round(criterion_f1, 4),
        "numeric_accuracy": round(numeric_accuracy, 4),
        "performance_band_accuracy": round(band_accuracy, 4),
        "descriptor_completeness": round(descriptor_completeness, 4),
        "represented_marks_correct": represented_marks_correct,
        "hallucinated_section_count": len(set(actual_sections) - set(expected_sections)),
        "hallucinated_criterion_count": len(set(actual_criteria) - set(expected_criteria)),
    }


def _answer_equal(actual: Any, expected: Any, mode: str) -> bool:
    if mode == "number":
        left, right = _safe_number(actual), _safe_number(expected)
        return left is not None and right is not None and abs(left - right) <= 1e-6
    if mode == "boolean":
        return isinstance(actual, bool) and actual is expected
    if mode == "set":
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False
        return {_normalize(item) for item in actual} == {_normalize(item) for item in expected}
    if mode == "ordered_list":
        if not isinstance(actual, list) or not isinstance(expected, list):
            return False
        return [_normalize(item) for item in actual] == [_normalize(item) for item in expected]
    return _normalize(actual) == _normalize(expected)


def _valid_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    valid = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start, end = int(item["line_start"]), int(item["line_end"])
        except (KeyError, TypeError, ValueError):
            continue
        source_id = str(item.get("source_id") or "").strip()
        if source_id and start > 0 and end >= start:
            valid.append({"source_id": source_id, "line_start": start, "line_end": end})
    return valid


def _evidence_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return (
        actual["source_id"] == str(expected["source_id"])
        and actual["line_start"] <= int(expected["line_end"])
        and actual["line_end"] >= int(expected["line_start"])
    )


def _evidence_specificity(actual: dict[str, Any], expected: dict[str, Any]) -> float:
    """Reward a supporting citation while penalizing needlessly broad ranges."""
    if not _evidence_matches(actual, expected):
        return 0.0
    actual_length = actual["line_end"] - actual["line_start"] + 1
    expected_length = int(expected["line_end"]) - int(expected["line_start"]) + 1
    return min(1.0, expected_length / max(1, actual_length))


def score_code_understanding(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    response, strict_json = parse_json_response(response_text)
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not a valid JSON object.",
        }
    raw_answers = response.get("answers")
    if not isinstance(raw_answers, list):
        return {
            "scorer_version": 1,
            "valid_json": True,
            "strict_json_compliance": strict_json,
            "overall_score": 0.0,
            "error": "Response answers must be an array.",
        }

    actual_answers = {
        str(item.get("question_id")): item
        for item in raw_answers
        if isinstance(item, dict) and item.get("question_id") is not None
    }
    gold_questions = {
        str(item["question_id"]): item for item in gold_payload.get("questions") or []
    }
    answer_checks = []
    expected_evidence: list[dict[str, Any]] = []
    actual_evidence: list[dict[str, Any]] = []
    per_question = []
    for question_id, expected in gold_questions.items():
        actual = actual_answers.get(question_id)
        accepted_answers = [
            expected.get("answer"),
            *(expected.get("accepted_answers") or []),
        ]
        answer_correct = bool(actual) and any(
            _answer_equal(
                actual.get("answer"),
                accepted,
                expected.get("answer_mode", "exact"),
            )
            for accepted in accepted_answers
        )
        answer_checks.append(answer_correct)
        expected_items = [
            {**item, "question_id": question_id}
            for item in expected.get("evidence") or []
        ]
        actual_items = [
            {**item, "question_id": question_id}
            for item in _valid_evidence(actual.get("evidence") if actual else None)
        ]
        expected_evidence.extend(expected_items)
        actual_evidence.extend(actual_items)
        matched_expected = sum(
            any(
                item["question_id"] == candidate["question_id"]
                and _evidence_matches(candidate, item)
                for candidate in actual_items
            )
            for item in expected_items
        )
        per_question.append({
            "question_id": question_id,
            "answer_correct": answer_correct,
            "evidence_recall": (
                round(matched_expected / len(expected_items), 4) if expected_items else 1.0
            ),
        })

    evidence_overlap = sum(
        max(
            (
                _evidence_specificity(item, candidate)
                for candidate in expected_evidence
                if candidate["question_id"] == item["question_id"]
            ),
            default=0.0,
        )
        for item in actual_evidence
    )
    evidence_precision = (
        evidence_overlap / len(actual_evidence) if actual_evidence else 0.0
    )
    evidence_recall_count = sum(
        any(
            candidate["question_id"] == item["question_id"]
            and _evidence_matches(candidate, item)
            for candidate in actual_evidence
        )
        for item in expected_evidence
    )
    evidence_recall = (
        evidence_recall_count / len(expected_evidence) if expected_evidence else 1.0
    )
    evidence_f1 = (
        2 * evidence_precision * evidence_recall / (evidence_precision + evidence_recall)
        if evidence_precision + evidence_recall else 0.0
    )
    answer_accuracy = sum(answer_checks) / len(answer_checks) if answer_checks else 0.0
    extra_answer_count = len(set(actual_answers) - set(gold_questions))
    case_id_correct = str(response.get("case_id") or "") == str(gold_payload.get("case_id"))
    penalty = min(10.0, extra_answer_count * 2.0) + (0.0 if case_id_correct else 5.0)
    overall = max(0.0, 100 * (0.8 * answer_accuracy + 0.2 * evidence_f1) - penalty)
    return {
        "scorer_version": 1,
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "case_id_correct": case_id_correct,
        "overall_score": round(overall, 2),
        "answer_accuracy": round(answer_accuracy, 4),
        "evidence_precision": round(evidence_precision, 4),
        "evidence_recall": round(evidence_recall, 4),
        "evidence_f1": round(evidence_f1, 4),
        "missing_answer_count": len(set(gold_questions) - set(actual_answers)),
        "unsupported_answer_count": extra_answer_count,
        "per_question": per_question,
    }


def score_visual_understanding(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    """Score objective diagram answers and their visible-label grounding."""
    response, strict_json = parse_json_response(response_text)
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not a valid JSON object.",
        }
    raw_answers = response.get("answers")
    if not isinstance(raw_answers, list):
        return {
            "scorer_version": 1,
            "valid_json": True,
            "strict_json_compliance": strict_json,
            "overall_score": 0.0,
            "error": "Response answers must be an array.",
        }

    actual_answers = {
        str(item.get("question_id")): item
        for item in raw_answers
        if isinstance(item, dict) and item.get("question_id") is not None
    }
    gold_questions = {
        str(item["question_id"]): item for item in gold_payload.get("questions") or []
    }
    answer_checks: list[bool] = []
    expected_labels: set[tuple[str, str]] = set()
    actual_labels: set[tuple[str, str]] = set()
    visible_labels = {
        _normalize(label)
        for label in gold_payload.get("visible_labels") or []
        if _normalize(label)
    }
    per_question = []
    for question_id, expected in gold_questions.items():
        actual = actual_answers.get(question_id)
        accepted_answers = [
            expected.get("answer"),
            *(expected.get("accepted_answers") or []),
        ]
        answer_correct = bool(actual) and any(
            _answer_equal(
                actual.get("answer"),
                accepted,
                expected.get("answer_mode", "exact"),
            )
            for accepted in accepted_answers
        )
        answer_checks.append(answer_correct)
        expected_for_question = {
            (question_id, _normalize(label))
            for label in expected.get("evidence_labels") or []
            if _normalize(label)
        }
        actual_for_question = {
            (question_id, _normalize(label))
            for label in (actual.get("evidence_labels") if actual else []) or []
            if _normalize(label)
        }
        expected_labels.update(expected_for_question)
        actual_labels.update(actual_for_question)
        label_overlap = len(expected_for_question & actual_for_question)
        per_question.append({
            "question_id": question_id,
            "answer_correct": answer_correct,
            "evidence_label_recall": round(
                label_overlap / len(expected_for_question), 4
            ) if expected_for_question else 1.0,
        })

    expected_overlap = len(expected_labels & actual_labels)
    evidence_recall = (
        expected_overlap / len(expected_labels) if expected_labels else 1.0
    )
    visible_actual_count = sum(
        label in visible_labels for _, label in actual_labels
    )
    evidence_precision = (
        visible_actual_count / len(actual_labels) if actual_labels else 0.0
    )
    evidence_f1 = (
        2 * evidence_precision * evidence_recall
        / (evidence_precision + evidence_recall)
        if evidence_precision + evidence_recall else 0.0
    )
    answer_accuracy = sum(answer_checks) / len(answer_checks) if answer_checks else 0.0
    extra_answer_count = len(set(actual_answers) - set(gold_questions))
    case_id_correct = str(response.get("case_id") or "") == str(gold_payload.get("case_id"))
    penalty = min(10.0, extra_answer_count * 2.0) + (0.0 if case_id_correct else 5.0)
    overall = max(0.0, 100 * (0.85 * answer_accuracy + 0.15 * evidence_f1) - penalty)
    return {
        "scorer_version": 1,
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "case_id_correct": case_id_correct,
        "overall_score": round(overall, 2),
        "answer_accuracy": round(answer_accuracy, 4),
        "evidence_label_precision": round(evidence_precision, 4),
        "evidence_label_recall": round(evidence_recall, 4),
        "evidence_label_f1": round(evidence_f1, 4),
        "missing_answer_count": len(set(gold_questions) - set(actual_answers)),
        "unsupported_answer_count": extra_answer_count,
        "unsupported_evidence_label_count": len(actual_labels) - visible_actual_count,
        "per_question": per_question,
    }


def score_knowledge_preparation(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    """Score evidence selection, claim typing, citations, and brief coverage."""
    response, strict_json = parse_json_response(response_text)
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not a valid JSON object.",
        }

    labels = ("fact", "alternative", "limitation", "reject")
    gold_classes = {
        claim_id: label
        for label in labels
        for claim_id in gold_payload.get(f"{label}_ids") or []
    }
    actual_classes: dict[str, str] = {}
    duplicate_claim_ids: set[str] = set()
    for label in labels:
        raw_ids = response.get(f"{label}_ids")
        if not isinstance(raw_ids, list):
            raw_ids = []
        for raw_id in raw_ids:
            claim_id = str(raw_id)
            if claim_id in actual_classes:
                duplicate_claim_ids.add(claim_id)
            else:
                actual_classes[claim_id] = label

    expected_ids = set(gold_classes)
    actual_ids = set(actual_classes)
    correct_classifications = sum(
        actual_classes.get(claim_id) == expected_label
        for claim_id, expected_label in gold_classes.items()
    )
    classification_accuracy = (
        correct_classifications / len(gold_classes) if gold_classes else 1.0
    )

    per_label: dict[str, dict[str, float]] = {}
    for label in labels:
        expected = {
            claim_id for claim_id, expected_label in gold_classes.items()
            if expected_label == label
        }
        actual = {
            claim_id for claim_id, actual_label in actual_classes.items()
            if actual_label == label
        }
        precision, recall, f1 = _f1(expected, actual)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    expected_citations = {
        (str(claim_id), str(source_id))
        for claim_id, source_ids in (gold_payload.get("citation_map") or {}).items()
        for source_id in source_ids
    }
    actual_citations = {
        (str(claim_id), str(source_id))
        for claim_id, source_ids in (response.get("citation_map") or {}).items()
        if isinstance(source_ids, list)
        for source_id in source_ids
    } if isinstance(response.get("citation_map"), dict) else set()
    citation_precision, citation_recall, citation_f1 = _f1(
        expected_citations, actual_citations
    )

    accepted_ids = {
        claim_id for claim_id, label in gold_classes.items() if label != "reject"
    }
    raw_brief_claim_ids = response.get("brief_claim_ids")
    brief_claim_ids = {
        str(item) for item in raw_brief_claim_ids
    } if isinstance(raw_brief_claim_ids, list) else set()
    brief_supported_ids = brief_claim_ids & accepted_ids
    brief_unsupported_ids = brief_claim_ids - accepted_ids
    brief_coverage = (
        len(brief_supported_ids) / len(accepted_ids) if accepted_ids else 1.0
    )
    brief_text_present = bool(str(response.get("brief") or "").strip())
    case_id_correct = str(response.get("case_id") or "") == str(
        gold_payload.get("case_id")
    )

    base_score = 100 * (
        0.50 * classification_accuracy
        + 0.25 * citation_f1
        + 0.15 * brief_coverage
        + 0.05 * float(brief_text_present)
        + 0.05 * float(strict_json)
    )
    penalty = (
        (0.0 if case_id_correct else 5.0)
        + min(10.0, 2.0 * len(brief_unsupported_ids))
        + min(5.0, 1.0 * len(duplicate_claim_ids))
        + min(5.0, 1.0 * len(actual_ids - expected_ids))
    )
    return {
        "scorer_version": 1,
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "case_id_correct": case_id_correct,
        "overall_score": round(max(0.0, base_score - penalty), 2),
        "claim_classification_accuracy": round(classification_accuracy, 4),
        "per_label": per_label,
        "citation_precision": round(citation_precision, 4),
        "citation_recall": round(citation_recall, 4),
        "citation_f1": round(citation_f1, 4),
        "brief_claim_coverage": round(brief_coverage, 4),
        "brief_text_present": brief_text_present,
        "missing_claim_count": len(expected_ids - actual_ids),
        "unknown_claim_count": len(actual_ids - expected_ids),
        "duplicate_claim_count": len(duplicate_claim_ids),
        "unsupported_brief_claim_count": len(brief_unsupported_ids),
    }


def score_answer_assessment(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    """Score controlled answer assessments without an LLM judge."""
    response, strict_json = parse_json_response(response_text)
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not a valid JSON object.",
        }
    raw_items = response.get("assessments")
    if not isinstance(raw_items, list):
        return {
            "scorer_version": 1,
            "valid_json": True,
            "strict_json_compliance": strict_json,
            "overall_score": 0.0,
            "error": "Response assessments must be an array.",
        }

    actual = {
        str(item.get("item_id")): item
        for item in raw_items
        if isinstance(item, dict) and item.get("item_id") is not None
    }
    expected = {
        str(item["item_id"]): item for item in gold_payload.get("assessments") or []
    }
    categorical_checks: list[bool] = []
    score_checks: list[bool] = []
    absolute_errors: list[float] = []
    misconception_scores: list[float] = []
    attribution_scores: list[float] = []
    rationale_checks: list[bool] = []
    per_item = []

    for item_id, gold in expected.items():
        observed = actual.get(item_id) or {}
        item_categories = [
            _normalize(observed.get(field)) == _normalize(gold.get(field))
            for field in ("triage", "criterion_id", "bloom_alignment", "decision")
        ]
        categorical_checks.extend(item_categories)
        item_score_checks = []
        for field, bounds in (gold.get("score_ranges") or {}).items():
            value = _safe_number(observed.get(field))
            if bounds is None:
                correct = observed.get(field) is None
            else:
                low, high = float(bounds[0]), float(bounds[1])
                correct = value is not None and low <= value <= high
                if value is not None:
                    absolute_errors.append(abs(value - ((low + high) / 2)))
            score_checks.append(correct)
            item_score_checks.append(correct)

        expected_misconceptions = {
            _normalize(value) for value in gold.get("misconception_labels") or []
        }
        actual_misconceptions = {
            _normalize(value) for value in observed.get("misconception_labels") or []
        } if isinstance(observed.get("misconception_labels"), list) else set()
        misconception_f1 = _f1(expected_misconceptions, actual_misconceptions)[2]
        misconception_scores.append(misconception_f1)

        expected_participants = {
            _normalize(value) for value in gold.get("participant_ids") or []
        }
        actual_participants = {
            _normalize(value) for value in observed.get("participant_ids") or []
        } if isinstance(observed.get("participant_ids"), list) else set()
        attribution_f1 = _f1(expected_participants, actual_participants)[2]
        attribution_scores.append(attribution_f1)
        rationale_present = bool(str(observed.get("rationale") or "").strip())
        rationale_checks.append(rationale_present)
        per_item.append({
            "item_id": item_id,
            "categorical_accuracy": round(sum(item_categories) / 4, 4),
            "score_range_accuracy": round(
                sum(item_score_checks) / len(item_score_checks), 4
            ) if item_score_checks else 1.0,
            "misconception_f1": round(misconception_f1, 4),
            "attribution_f1": round(attribution_f1, 4),
        })

    categorical_accuracy = (
        sum(categorical_checks) / len(categorical_checks)
        if categorical_checks else 0.0
    )
    score_range_accuracy = (
        sum(score_checks) / len(score_checks) if score_checks else 0.0
    )
    misconception_f1 = (
        sum(misconception_scores) / len(misconception_scores)
        if misconception_scores else 0.0
    )
    attribution_f1 = (
        sum(attribution_scores) / len(attribution_scores)
        if attribution_scores else 0.0
    )
    rationale_completeness = (
        sum(rationale_checks) / len(rationale_checks) if rationale_checks else 0.0
    )
    case_id_correct = str(response.get("case_id") or "") == str(
        gold_payload.get("case_id")
    )
    extra_items = len(set(actual) - set(expected))
    overall = 100 * (
        0.30 * categorical_accuracy
        + 0.25 * score_range_accuracy
        + 0.15 * misconception_f1
        + 0.10 * attribution_f1
        + 0.10 * rationale_completeness
        + 0.05 * float(strict_json)
        + 0.05 * float(case_id_correct)
    ) - min(10.0, 2.0 * extra_items)
    return {
        "scorer_version": 1,
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "case_id_correct": case_id_correct,
        "overall_score": round(max(0.0, overall), 2),
        "categorical_accuracy": round(categorical_accuracy, 4),
        "score_range_accuracy": round(score_range_accuracy, 4),
        "score_mae_to_gold_midpoint": round(
            sum(absolute_errors) / len(absolute_errors), 4
        ) if absolute_errors else None,
        "misconception_f1": round(misconception_f1, 4),
        "attribution_f1": round(attribution_f1, 4),
        "rationale_completeness": round(rationale_completeness, 4),
        "missing_item_count": len(set(expected) - set(actual)),
        "extra_item_count": extra_items,
        "per_item": per_item,
    }


def _question_tokens(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def score_question_quality(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    response, strict_json = parse_json_response(response_text)
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not a valid JSON object.",
        }
    task_type = gold_payload.get("task_type")
    case_id_correct = str(response.get("case_id") or "") == str(
        gold_payload.get("case_id")
    )

    if task_type == "critic":
        raw = response.get("reviews")
        if not isinstance(raw, list):
            return {
                "scorer_version": 1,
                "valid_json": True,
                "overall_score": 0.0,
                "error": "Response reviews must be an array.",
            }
        actual = {
            str(item.get("candidate_id")): item
            for item in raw if isinstance(item, dict) and item.get("candidate_id")
        }
        expected = {
            str(item["candidate_id"]): item
            for item in gold_payload.get("reviews") or []
        }
        verdict_checks = []
        issue_scores = []
        rationale_checks = []
        for candidate_id, gold in expected.items():
            observed = actual.get(candidate_id) or {}
            verdict_checks.append(
                _normalize(observed.get("verdict")) == _normalize(gold.get("verdict"))
            )
            expected_issues = {_normalize(v) for v in gold.get("issue_labels") or []}
            actual_issues = {
                _normalize(v) for v in observed.get("issue_labels") or []
            } if isinstance(observed.get("issue_labels"), list) else set()
            issue_scores.append(_f1(expected_issues, actual_issues)[2])
            rationale_checks.append(bool(str(observed.get("rationale") or "").strip()))
        verdict_accuracy = sum(verdict_checks) / len(verdict_checks) if verdict_checks else 0.0
        issue_f1 = sum(issue_scores) / len(issue_scores) if issue_scores else 0.0
        rationale_completeness = (
            sum(rationale_checks) / len(rationale_checks) if rationale_checks else 0.0
        )
        coverage = len(set(actual) & set(expected)) / len(expected) if expected else 1.0
        overall = 100 * (
            0.50 * verdict_accuracy
            + 0.30 * issue_f1
            + 0.10 * rationale_completeness
            + 0.05 * coverage
            + 0.025 * float(strict_json)
            + 0.025 * float(case_id_correct)
        )
        return {
            "scorer_version": 1,
            "task_type": "critic",
            "valid_json": True,
            "strict_json_compliance": strict_json,
            "case_id_correct": case_id_correct,
            "overall_score": round(overall, 2),
            "verdict_accuracy": round(verdict_accuracy, 4),
            "issue_label_f1": round(issue_f1, 4),
            "rationale_completeness": round(rationale_completeness, 4),
            "candidate_coverage": round(coverage, 4),
            "missing_candidate_count": len(set(expected) - set(actual)),
        }

    raw = response.get("questions")
    if not isinstance(raw, list):
        return {
            "scorer_version": 1,
            "valid_json": True,
            "overall_score": 0.0,
            "error": "Response questions must be an array.",
        }
    actual = {
        str(item.get("context_id")): item
        for item in raw if isinstance(item, dict) and item.get("context_id")
    }
    expected = {
        str(item["context_id"]): item
        for item in gold_payload.get("questions") or []
    }
    planning_checks = []
    source_scores = []
    keyword_scores = []
    form_scores = []
    unsupported_sources = 0
    per_item = []
    for context_id, gold in expected.items():
        observed = actual.get(context_id) or {}
        planning = [
            _normalize(observed.get("target_bloom")) == _normalize(gold.get("target_bloom")),
            _normalize(observed.get("socratic_intent")) == _normalize(gold.get("socratic_intent")),
        ]
        planning_checks.extend(planning)
        expected_sources = {str(v) for v in gold.get("source_chunk_ids") or []}
        actual_sources = {
            str(v) for v in observed.get("source_chunk_ids") or []
        } if isinstance(observed.get("source_chunk_ids"), list) else set()
        source_scores.append(_f1(expected_sources, actual_sources)[2])
        unsupported_sources += len(actual_sources - expected_sources)

        question = str(observed.get("question_text") or "").strip()
        tokens = _question_tokens(question)
        groups = gold.get("required_keyword_groups") or []
        keyword_coverage = (
            sum(bool(tokens & {_normalize(v) for v in group}) for group in groups)
            / len(groups) if groups else 1.0
        )
        keyword_scores.append(keyword_coverage)
        recent = [_question_tokens(value) for value in gold.get("recent_questions") or []]
        max_similarity = max((_jaccard(tokens, item) for item in recent), default=0.0)
        word_count = len(re.findall(r"\b\w+\b", question))
        prohibited = [
            _normalize(value) for value in gold.get("prohibited_phrases") or []
        ]
        normalized_question = _normalize(question)
        form = [
            question.count("?") == 1,
            8 <= word_count <= 45,
            not any(value and value in normalized_question for value in prohibited),
            max_similarity < 0.72,
        ]
        form_score = sum(form) / len(form)
        form_scores.append(form_score)
        per_item.append({
            "context_id": context_id,
            "planning_accuracy": round(sum(planning) / 2, 4),
            "source_f1": round(source_scores[-1], 4),
            "keyword_coverage": round(keyword_coverage, 4),
            "form_score": round(form_score, 4),
            "max_recent_similarity": round(max_similarity, 4),
        })
    planning_accuracy = sum(planning_checks) / len(planning_checks) if planning_checks else 0.0
    source_f1 = sum(source_scores) / len(source_scores) if source_scores else 0.0
    keyword_coverage = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
    form_score = sum(form_scores) / len(form_scores) if form_scores else 0.0
    coverage = len(set(actual) & set(expected)) / len(expected) if expected else 1.0
    overall = 100 * (
        0.30 * planning_accuracy
        + 0.25 * source_f1
        + 0.20 * keyword_coverage
        + 0.15 * form_score
        + 0.05 * coverage
        + 0.025 * float(strict_json)
        + 0.025 * float(case_id_correct)
    ) - min(10.0, unsupported_sources * 2.0)
    return {
        "scorer_version": 1,
        "task_type": "generation",
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "case_id_correct": case_id_correct,
        "overall_score": round(max(0.0, overall), 2),
        "planning_accuracy": round(planning_accuracy, 4),
        "source_f1": round(source_f1, 4),
        "keyword_coverage": round(keyword_coverage, 4),
        "form_score": round(form_score, 4),
        "context_coverage": round(coverage, 4),
        "unsupported_source_count": unsupported_sources,
        "per_item": per_item,
    }


def score_session_reporting(
    response_text: str,
    gold_payload: dict[str, Any],
) -> dict[str, Any]:
    response, strict_json = parse_json_response(response_text)
    if not isinstance(response, dict):
        return {
            "scorer_version": 1,
            "valid_json": False,
            "overall_score": 0.0,
            "error": "Response is not a valid JSON object.",
        }
    case_id_correct = str(response.get("case_id") or "") == str(
        gold_payload.get("case_id") or ""
    )
    raw_reports = response.get("reports")
    if not isinstance(raw_reports, list):
        return {
            "scorer_version": 1,
            "valid_json": True,
            "overall_score": 0.0,
            "error": "Response reports must be an array.",
        }
    actual_reports = {
        str(item.get("session_id")): item
        for item in raw_reports
        if isinstance(item, dict) and item.get("session_id")
    }
    expected_reports = {
        str(item["session_id"]): item for item in gold_payload.get("reports") or []
    }
    participant_expected = 0
    participant_found = 0
    score_checks: list[bool] = []
    out_of_checks: list[bool] = []
    criterion_checks: list[bool] = []
    attribution_scores: list[float] = []
    summary_checks: list[bool] = []
    flag_scores: list[float] = []
    extra_participants = 0
    forbidden_grade_fields = 0
    per_session = []
    for session_id, gold_report in expected_reports.items():
        observed_report = actual_reports.get(session_id) or {}
        raw_participants = observed_report.get("participant_results") or []
        participants = {
            str(item.get("participant_id")): item
            for item in raw_participants
            if isinstance(item, dict) and item.get("participant_id")
        }
        expected_participants = {
            str(item["participant_id"]): item
            for item in gold_report.get("participant_results") or []
        }
        participant_expected += len(expected_participants)
        participant_found += len(set(participants) & set(expected_participants))
        extra_participants += len(set(participants) - set(expected_participants))
        session_score_checks = []
        for participant_id, gold_participant in expected_participants.items():
            observed = participants.get(participant_id) or {}
            score_ok = (
                _safe_number(observed.get("final_score"))
                == _safe_number(gold_participant.get("final_score"))
            )
            out_of_ok = (
                _safe_number(observed.get("score_out_of"))
                == _safe_number(gold_participant.get("score_out_of"))
            )
            score_checks.append(score_ok)
            out_of_checks.append(out_of_ok)
            session_score_checks.append(score_ok and out_of_ok)
            expected_questions = {
                str(value) for value in gold_participant.get("answered_question_ids") or []
            }
            actual_questions = {
                str(value) for value in observed.get("answered_question_ids") or []
            } if isinstance(observed.get("answered_question_ids"), list) else set()
            attribution_scores.append(_f1(expected_questions, actual_questions)[2])
            expected_criteria = gold_participant.get("criterion_scores") or {}
            actual_criteria = observed.get("criterion_scores") or {}
            for criterion, expected_value in expected_criteria.items():
                criterion_checks.append(
                    _safe_number(actual_criteria.get(criterion))
                    == _safe_number(expected_value)
                )
            forbidden_grade_fields += int(
                "grade" in observed or "letter_grade" in observed
            )
        expected_summary = gold_report.get("session_summary") or {}
        actual_summary = observed_report.get("session_summary") or {}
        for key, expected_value in expected_summary.items():
            summary_checks.append(
                _safe_number(actual_summary.get(key)) == _safe_number(expected_value)
            )
        expected_flags = {_normalize(v) for v in gold_report.get("flags") or []}
        actual_flags = {
            _normalize(v) for v in observed_report.get("flags") or []
        } if isinstance(observed_report.get("flags"), list) else set()
        flag_scores.append(_f1(expected_flags, actual_flags)[2])
        per_session.append({
            "session_id": session_id,
            "participant_coverage": round(
                len(set(participants) & set(expected_participants))
                / len(expected_participants), 4
            ) if expected_participants else 1.0,
            "participant_scores_correct": all(session_score_checks),
        })
    participant_coverage = (
        participant_found / participant_expected if participant_expected else 1.0
    )
    score_accuracy = sum(score_checks) / len(score_checks) if score_checks else 0.0
    out_of_accuracy = sum(out_of_checks) / len(out_of_checks) if out_of_checks else 0.0
    criterion_accuracy = (
        sum(criterion_checks) / len(criterion_checks) if criterion_checks else 0.0
    )
    attribution_f1 = (
        sum(attribution_scores) / len(attribution_scores)
        if attribution_scores else 0.0
    )
    summary_accuracy = (
        sum(summary_checks) / len(summary_checks) if summary_checks else 0.0
    )
    flag_f1 = sum(flag_scores) / len(flag_scores) if flag_scores else 0.0
    overall = 100 * (
        0.10 * participant_coverage
        + 0.25 * score_accuracy
        + 0.10 * out_of_accuracy
        + 0.15 * criterion_accuracy
        + 0.15 * attribution_f1
        + 0.10 * summary_accuracy
        + 0.05 * flag_f1
        + 0.05 * float(strict_json)
        + 0.05 * float(case_id_correct)
    ) - min(15.0, 3.0 * extra_participants + 5.0 * forbidden_grade_fields)
    return {
        "scorer_version": 1,
        "valid_json": True,
        "strict_json_compliance": strict_json,
        "case_id_correct": case_id_correct,
        "overall_score": round(max(0.0, overall), 2),
        "participant_coverage": round(participant_coverage, 4),
        "final_score_accuracy": round(score_accuracy, 4),
        "score_out_of_accuracy": round(out_of_accuracy, 4),
        "criterion_score_accuracy": round(criterion_accuracy, 4),
        "question_attribution_f1": round(attribution_f1, 4),
        "session_summary_accuracy": round(summary_accuracy, 4),
        "flag_f1": round(flag_f1, 4),
        "extra_participant_count": extra_participants,
        "forbidden_grade_field_count": forbidden_grade_fields,
        "per_session": per_session,
    }


def score_result_record(
    result: dict[str, Any],
    *,
    dataset_root: str | Path,
) -> dict[str, Any]:
    scored = {
        "schema_version": 1,
        "run_id": result.get("run_id"),
        "model_id": result.get("model_id"),
        "case_id": result.get("case_id"),
        "category": result.get("category"),
        "response_status": result.get("status"),
    }
    if result.get("status") not in {"success", "invalid_response"}:
        return {**scored, "score_status": "not_scored", "reason": "response_not_available"}

    gold_reference = (result.get("case_metadata") or {}).get("gold_label")
    if not gold_reference:
        return {**scored, "score_status": "not_scored", "reason": "gold_label_missing"}
    gold_path = Path(dataset_root) / str(result.get("category")) / gold_reference
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    if result.get("category") == "rubric_understanding":
        metrics = score_rubric_structure(result.get("response_text", ""), gold)
    elif result.get("category") == "code_understanding":
        metrics = score_code_understanding(result.get("response_text", ""), gold)
    elif result.get("category") == "visual_understanding":
        metrics = score_visual_understanding(result.get("response_text", ""), gold)
    elif result.get("category") == "knowledge_preparation":
        metrics = score_knowledge_preparation(result.get("response_text", ""), gold)
    elif result.get("category") == "answer_assessment":
        metrics = score_answer_assessment(result.get("response_text", ""), gold)
    elif result.get("category") == "question_quality":
        metrics = score_question_quality(result.get("response_text", ""), gold)
    elif result.get("category") == "reporting":
        metrics = score_session_reporting(result.get("response_text", ""), gold)
    else:
        return {**scored, "score_status": "not_scored", "reason": "scorer_not_implemented"}
    return {
        **scored,
        "score_status": (
            "scored"
            if result.get("status") == "success"
            else "scored_from_recoverable_json"
        ),
        "gold_review_status": gold.get("review_status"),
        "metrics": metrics,
    }
