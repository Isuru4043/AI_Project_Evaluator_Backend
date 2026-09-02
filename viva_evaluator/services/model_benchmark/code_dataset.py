from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodeDatasetError(ValueError):
    """Raised when a code-understanding dataset cannot be built safely."""


@dataclass(frozen=True)
class RepositorySnapshot:
    alias: str
    root: Path
    revision: str


def _git_revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CodeDatasetError(f"Unable to resolve Git revision for {root}") from exc


def _safe_source_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise CodeDatasetError(f"Source escapes repository root: {relative_path}") from exc
    if not candidate.is_file():
        raise CodeDatasetError(f"Source file does not exist: {relative_path}")
    return candidate


def _source_excerpt(snapshot: RepositorySnapshot, source: dict[str, Any]) -> dict[str, Any]:
    relative_path = str(source["path"]).replace("\\", "/")
    path = _safe_source_path(snapshot.root, relative_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(1, int(source["line_start"]))
    end = min(len(lines), int(source["line_end"]))
    if start > end:
        raise CodeDatasetError(
            f"Invalid line range {start}-{end} for {snapshot.alias}:{relative_path}"
        )
    selected = lines[start - 1 : end]
    numbered = "\n".join(
        f"{line_number:4}: {text}"
        for line_number, text in enumerate(selected, start=start)
    )
    raw = "\n".join(selected)
    return {
        "source_id": str(source["source_id"]),
        "repo": snapshot.alias,
        "path": relative_path,
        "line_start": start,
        "line_end": end,
        "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "numbered_text": numbered,
    }


def _render_prompt(case: dict[str, Any], excerpts: list[dict[str, Any]]) -> str:
    source_blocks = []
    for source in excerpts:
        language = "typescript" if source["path"].endswith((".ts", ".tsx")) else "python"
        source_blocks.append(
            f'### {source["source_id"]}: {source["repo"]}/{source["path"]} '
            f'(lines {source["line_start"]}-{source["line_end"]})\n'
            f"```{language}\n{source['numbered_text']}\n```"
        )
    questions = "\n".join(
        f'{index}. [{question["question_id"]}] {question["prompt"]}'
        for index, question in enumerate(case["questions"], start=1)
    )
    return f"""You are evaluating code understanding, not proposing changes.
Use only the supplied source excerpts. Do not infer behavior that is not supported by them.

Case: {case['case_id']} — {case['title']}

{chr(10).join(source_blocks)}

Questions:
{questions}

Return one JSON object and no Markdown fences:
{{
  "case_id": "{case['case_id']}",
  "answers": [
    {{
      "question_id": "question ID from above",
      "answer": "a scalar, boolean, number, or JSON array as appropriate",
      "evidence": [
        {{"source_id": "B1", "line_start": 1, "line_end": 2}}
      ]
    }}
  ]
}}

Include exactly one answer for every question. Evidence must cite the smallest supplied line range that supports the answer.
"""


def build_code_dataset(
    *,
    spec_path: str | Path,
    output_root: str | Path,
    repository_roots: dict[str, str | Path],
) -> dict[str, Any]:
    spec_path = Path(spec_path)
    output_root = Path(output_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    snapshots = {
        alias: RepositorySnapshot(alias, Path(root).resolve(), _git_revision(Path(root).resolve()))
        for alias, root in repository_roots.items()
    }

    prompts_dir = output_root / "prompts"
    gold_dir = output_root / "gold"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)
    case_records: list[dict[str, Any]] = []
    manifest_cases: list[dict[str, Any]] = []

    seen: set[str] = set()
    for case in spec.get("cases") or []:
        case_id = str(case["case_id"])
        if case_id in seen:
            raise CodeDatasetError(f"Duplicate case ID: {case_id}")
        seen.add(case_id)
        excerpts = []
        source_ids: set[str] = set()
        for source in case.get("sources") or []:
            source_id = str(source["source_id"])
            if source_id in source_ids:
                raise CodeDatasetError(f"Duplicate source ID {source_id} in {case_id}")
            source_ids.add(source_id)
            repo = str(source["repo"])
            if repo not in snapshots:
                raise CodeDatasetError(f"Unknown repository alias {repo!r} in {case_id}")
            excerpts.append(_source_excerpt(snapshots[repo], source))

        excerpt_index = {item["source_id"]: item for item in excerpts}

        question_ids = [str(item["question_id"]) for item in case.get("questions") or []]
        if not question_ids or len(question_ids) != len(set(question_ids)):
            raise CodeDatasetError(f"Questions are empty or duplicated in {case_id}")
        for question in case["questions"]:
            for evidence in question.get("evidence") or []:
                evidence_source = str(evidence["source_id"])
                if evidence_source not in source_ids:
                    raise CodeDatasetError(
                        f"Unknown evidence source {evidence['source_id']} in {case_id}"
                    )
                excerpt = excerpt_index[evidence_source]
                evidence_start = int(evidence["line_start"])
                evidence_end = int(evidence["line_end"])
                if (
                    evidence_start < excerpt["line_start"]
                    or evidence_end > excerpt["line_end"]
                    or evidence_end < evidence_start
                ):
                    raise CodeDatasetError(
                        f"Gold evidence escapes {evidence_source} in {case_id}"
                    )

        prompt_name = f"{case_id}.txt"
        gold_name = f"{case_id}.gold.json"
        (prompts_dir / prompt_name).write_text(
            _render_prompt(case, excerpts), encoding="utf-8"
        )
        gold_payload = {
            "schema_version": 1,
            "case_id": case_id,
            "category": "code_understanding",
            "review_status": "draft_pending_examiner_review",
            "questions": case["questions"],
        }
        (gold_dir / gold_name).write_text(
            json.dumps(gold_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        metadata = {
            "gold_label": f"gold/{gold_name}",
            "case_type": case.get("case_type", "single_unit"),
            "title": case["title"],
            "source_snapshot": [
                {key: value for key, value in excerpt.items() if key != "numbered_text"}
                for excerpt in excerpts
            ],
            "repository_revisions": {
                alias: snapshots[alias].revision
                for alias in sorted({excerpt["repo"] for excerpt in excerpts})
            },
        }
        case_records.append({
            "case_id": case_id,
            "category": "code_understanding",
            "prompt_file": f"prompts/{prompt_name}",
            "system_prompt": (
                "Answer solely from the provided code. Return strict JSON with exact, "
                "line-based evidence and no Markdown."
            ),
            "max_output_tokens": int(case.get("max_output_tokens", 1200)),
            "expected_response_format": "json",
            "metadata": metadata,
        })
        manifest_cases.append({"case_id": case_id, **metadata})

    cases_path = output_root / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in case_records),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "category": "code_understanding",
        "review_status": "draft_pending_examiner_review",
        "case_count": len(case_records),
        "repository_revisions": {
            alias: snapshot.revision for alias, snapshot in snapshots.items()
        },
        "cases": manifest_cases,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
