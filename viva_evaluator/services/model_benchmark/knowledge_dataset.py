from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LABELS = ("fact", "alternative", "limitation", "reject")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_case(case: dict[str, Any]) -> None:
    source_ids = [str(item["source_id"]) for item in case.get("sources") or []]
    claim_ids = [str(item["claim_id"]) for item in case.get("claims") or []]
    if not source_ids or len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{case.get('case_id')}: source IDs must be non-empty and unique")
    if not claim_ids or len(claim_ids) != len(set(claim_ids)):
        raise ValueError(f"{case.get('case_id')}: claim IDs must be non-empty and unique")
    known_sources = set(source_ids)
    for claim in case["claims"]:
        label = str(claim.get("label") or "")
        if label not in LABELS:
            raise ValueError(f"{case['case_id']}: unsupported claim label {label!r}")
        cited = {str(item) for item in claim.get("source_ids") or []}
        if label == "reject" and cited:
            raise ValueError(f"{case['case_id']}: rejected claims cannot cite evidence")
        if label != "reject" and not cited:
            raise ValueError(f"{case['case_id']}: accepted claims require evidence")
        if not cited.issubset(known_sources):
            raise ValueError(f"{case['case_id']}: claim cites an unknown source")


def _build_prompt(case: dict[str, Any]) -> str:
    lines = [
        "You are preparing a concise technical knowledge brief for VivaSense.",
        "Use only the frozen evidence package below. Do not use outside knowledge.",
        "Classify every candidate claim exactly once as fact, alternative, limitation, or reject.",
        "A rejected claim is unsupported or contradicted by the supplied evidence.",
        "Citations must use only the supplied source IDs.",
        "",
        f"Case: {case['case_id']} — {case['title']}",
        f"VivaSense context: {case['context']}",
        "",
        "FROZEN EVIDENCE",
    ]
    for source in case["sources"]:
        lines.extend([
            f"[{source['source_id']}] {source['title']}",
            f"Publisher: {source['publisher']}",
            f"URL: {source['url']}",
            f"Evidence: {source['evidence']}",
            "",
        ])
    lines.append("CANDIDATE CLAIMS")
    for claim in case["claims"]:
        lines.append(f"[{claim['claim_id']}] {claim['text']}")
    lines.extend([
        "",
        "Return one strict JSON object with no Markdown fences or commentary:",
        "{",
        f'  "case_id": "{case["case_id"]}",',
        '  "fact_ids": ["claim IDs"],',
        '  "alternative_ids": ["claim IDs"],',
        '  "limitation_ids": ["claim IDs"],',
        '  "reject_ids": ["claim IDs"],',
        '  "citation_map": {"accepted claim ID": ["source IDs"]},',
        '  "brief_claim_ids": ["accepted claim IDs actually used in the brief"],',
        '  "brief": "A 90-140 word technical brief covering purpose, suitable use, one alternative, and important limitations. Cite source IDs inline like [S1]."',
        "}",
        "",
        "Include every candidate claim ID exactly once across the four classification arrays.",
        "Do not cite rejected claims or include them in brief_claim_ids.",
    ])
    return "\n".join(lines)


def build_knowledge_dataset(spec_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    spec_file = Path(spec_path)
    output = Path(output_root)
    payload = json.loads(spec_file.read_text(encoding="utf-8"))
    cases = payload.get("cases") or []
    case_ids = [str(item.get("case_id")) for item in cases]
    if not cases or len(case_ids) != len(set(case_ids)):
        raise ValueError("Knowledge cases must be non-empty and have unique IDs")

    prompts = output / "prompts"
    gold_dir = output / "gold"
    sources_dir = output / "sources"
    prompts.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)

    case_records = []
    manifest_cases = []
    for case in cases:
        _validate_case(case)
        case_id = str(case["case_id"])
        prompt = _build_prompt(case)
        prompt_rel = f"prompts/{case_id}.txt"
        gold_rel = f"gold/{case_id}.gold.json"
        sources_rel = f"sources/{case_id}.sources.json"
        (output / prompt_rel).write_text(prompt, encoding="utf-8")

        source_payload = {
            "schema_version": 1,
            "case_id": case_id,
            "frozen_at": payload["frozen_at"],
            "sources": case["sources"],
        }
        source_text = json.dumps(source_payload, ensure_ascii=False, indent=2) + "\n"
        (output / sources_rel).write_text(source_text, encoding="utf-8")

        by_label = {
            label: [
                str(item["claim_id"])
                for item in case["claims"]
                if item["label"] == label
            ]
            for label in LABELS
        }
        citation_map = {
            str(item["claim_id"]): [str(value) for value in item["source_ids"]]
            for item in case["claims"]
            if item["label"] != "reject"
        }
        gold = {
            "schema_version": 1,
            "case_id": case_id,
            "category": "knowledge_preparation",
            "review_status": "draft_pending_examiner_review",
            "fact_ids": by_label["fact"],
            "alternative_ids": by_label["alternative"],
            "limitation_ids": by_label["limitation"],
            "reject_ids": by_label["reject"],
            "citation_map": citation_map,
            "accepted_claims": [
                {"claim_id": item["claim_id"], "text": item["text"]}
                for item in case["claims"]
                if item["label"] != "reject"
            ],
        }
        (output / gold_rel).write_text(
            json.dumps(gold, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        source_hash = _sha256_text(_canonical_json(source_payload))
        case_records.append({
            "case_id": case_id,
            "category": "knowledge_preparation",
            "prompt_file": prompt_rel,
            "system_prompt": (
                "Prepare an evidence-grounded technical brief using only the supplied "
                "frozen sources. Return strict JSON and reject unsupported claims."
            ),
            "max_output_tokens": 1000,
            "expected_response_format": "json",
            "required_capabilities": ["text"],
            "metadata": {
                "gold_label": gold_rel,
                "sources_file": sources_rel,
                "technology": case["technology"],
                "source_sha256": source_hash,
                "prompt_sha256": _sha256_text(prompt),
                "review_status": "draft_pending_examiner_review",
            },
        })
        manifest_cases.append({
            "case_id": case_id,
            "technology": case["technology"],
            "title": case["title"],
            "gold_label": gold_rel,
            "sources_file": sources_rel,
            "source_sha256": source_hash,
            "prompt_sha256": _sha256_text(prompt),
            "source_urls": [item["url"] for item in case["sources"]],
        })

    with (output / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for record in case_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "category": "knowledge_preparation",
        "pilot_case_count": len(cases),
        "frozen_at": payload["frozen_at"],
        "gold_review_status": "draft_pending_examiner_review",
        "evidence_policy": (
            "All providers receive identical frozen paraphrases of official documentation; "
            "provider-side browsing is prohibited."
        ),
        "external_run_policy": (
            "Requires fresh explicit authorization before prompts are sent to providers."
        ),
        "cases": manifest_cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
