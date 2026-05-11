import os
import subprocess
from pathlib import Path

from django.conf import settings
from django.utils import timezone
from requests import HTTPError

from core.models import CodeSubmission, GeneratedVivaQuestion
from .gemini_service import GeminiService
from .repo_service import cleanup_path, clone_repo, iter_code_files, safe_extract_zip
from .sonarqube_service import SonarQubeService


class CodeAnalysisService:
    def __init__(self):
        self.sonar = SonarQubeService()
        self.gemini = GeminiService()

    def analyze_submission(self, code_submission_id):
        submission = CodeSubmission.objects.get(id=code_submission_id)
        repo_path = None

        try:
            submission.analysis_status = CodeSubmission.AnalysisStatus.FETCHING
            submission.save(update_fields=["analysis_status"])

            repo_path = self._prepare_repo(submission)
            language, build_system, build_command = detect_language_and_build(repo_path)

            submission.language_detected = language
            submission.build_system_detected = build_system
            submission.save(update_fields=[
                "language_detected",
                "build_system_detected",
            ])

            project_key = submission.sonar_project_key or f"code-{submission.id.hex}"
            submission.sonar_project_key = project_key
            submission.sonar_report_url = self.sonar.dashboard_url(project_key)
            submission.analysis_status = CodeSubmission.AnalysisStatus.SCANNING
            submission.save(update_fields=[
                "sonar_project_key",
                "sonar_report_url",
                "analysis_status",
            ])

            self.sonar.ensure_project(project_key, f"Code Submission {submission.id}")

            if submission.build_command and submission.build_command.strip():
                run_build_command(submission.build_command, repo_path)

            self.sonar.run_scanner(
                project_key=project_key,
                base_dir=repo_path,
                extra_args=_scanner_extra_args(),
            )
            submission.sonar_task_id = read_sonar_task_id(repo_path)

            code_excerpt = collect_code_excerpt(repo_path)
            if self.gemini.is_enabled():
                submission.code_summary = self.gemini.summarize_code(code_excerpt)
                question_excerpt = collect_question_focus_excerpt(repo_path)
                questions = self.gemini.generate_questions(question_excerpt)
                for question in questions:
                    GeneratedVivaQuestion.objects.create(
                        code_submission=submission,
                        question_text=question.get("question", ""),
                        blooms_level=question.get("blooms_level"),
                        source_type=GeneratedVivaQuestion.SourceType.CODE,
                        code_reference=question.get("code_reference"),
                        sonar_issue_reference=None,
                        reasoning=question.get("reasoning"),
                    )
                submission.questions_generated_at = timezone.now() if questions else None
            else:
                submission.code_summary = None
                submission.questions_generated_at = None

            submission.analysis_status = CodeSubmission.AnalysisStatus.SCANNING
            submission.save(update_fields=[
                "sonar_task_id",
                "code_summary",
                "questions_generated_at",
                "analysis_status",
            ])

        except Exception as exc:
            submission.analysis_status = CodeSubmission.AnalysisStatus.FAILED
            submission.analysis_error = str(exc)
            submission.save(update_fields=["analysis_status", "analysis_error"])
        finally:
            cleanup_path(repo_path)

    def refresh_submission(self, code_submission_id):
        submission = CodeSubmission.objects.get(id=code_submission_id)
        needs_summary_refresh = _summary_needs_refresh(submission.sonar_summary)

        if submission.analysis_status != CodeSubmission.AnalysisStatus.SCANNING and not needs_summary_refresh:
            return submission

        if submission.analysis_status == CodeSubmission.AnalysisStatus.SCANNING and not submission.sonar_task_id:
            return submission

        if submission.analysis_status == CodeSubmission.AnalysisStatus.SCANNING:
            task = self.sonar.get_task_status(submission.sonar_task_id)
            task_status = task.get("status")
            if task_status in ("PENDING", "IN_PROGRESS"):
                return submission

            if task_status != "SUCCESS":
                submission.analysis_status = CodeSubmission.AnalysisStatus.FAILED
                submission.analysis_error = f"SonarCloud task {task_status}"
                submission.save(update_fields=["analysis_status", "analysis_error"])
                return submission

        try:
            submission.sonar_summary = self.sonar.get_summary(submission.sonar_project_key)
        except HTTPError as exc:
            response = getattr(exc, "response", None)
            if response is not None and response.status_code == 404:
                submission.sonar_summary = {}
                submission.quality_status = CodeSubmission.QualityStatus.UNKNOWN
                submission.quality_reason = "SonarCloud metrics not yet available; retry later."
            else:
                raise
        else:
            submission.analyzed_at = timezone.now()
            quality_status, quality_reason = compute_quality_status(submission.sonar_summary)
            submission.quality_status = quality_status
            submission.quality_reason = quality_reason

        if submission.analysis_status == CodeSubmission.AnalysisStatus.COMPLETED:
            submission.save(update_fields=[
                "sonar_summary",
                "analyzed_at",
                "quality_status",
                "quality_reason",
            ])
            return submission

        submission.analysis_status = CodeSubmission.AnalysisStatus.COMPLETED
        submission.save(update_fields=[
            "sonar_summary",
            "analyzed_at",
            "quality_status",
            "quality_reason",
            "analysis_status",
        ])

        return submission

    def _prepare_repo(self, submission):
        if submission.source_type == CodeSubmission.SourceType.GITHUB:
            return clone_repo(submission.github_url)

        if submission.zip_file:
            return safe_extract_zip(submission.zip_file.path)

        raise ValueError("No source provided for analysis.")


def detect_language_and_build(repo_path):
    repo_path = Path(repo_path)
    files = {p.name for p in repo_path.glob("**/*") if p.is_file()}

    if "pom.xml" in files:
        return "java", "maven", "mvn -q test"
    if "build.gradle" in files or "build.gradle.kts" in files:
        return "java", "gradle", "gradle test"
    if "package.json" in files:
        return "javascript", "npm", "npm install && npm run build"
    if "requirements.txt" in files or "pyproject.toml" in files:
        return "python", "pip", ""
    if "CMakeLists.txt" in files:
        return "cpp", "cmake", "cmake -S . -B build && cmake --build build"
    if "Makefile" in files:
        return "cpp", "make", "make"

    extensions = {}
    for file_path in repo_path.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1

    if extensions:
        language = max(extensions, key=extensions.get).lstrip(".")
        return language, "unknown", ""

    return "unknown", "unknown", ""


def collect_code_excerpt(repo_path):
    repo_root = Path(repo_path)
    max_chars = getattr(settings, "CODE_ANALYSIS_MAX_PROMPT_CHARS", 20000)
    max_file_bytes = 200000
    total = 0
    parts = []

    for file_path in iter_code_files(repo_path):
        if total >= max_chars:
            break
        try:
            if file_path.stat().st_size > max_file_bytes:
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        header = f"\n# File: {file_path.relative_to(repo_root)}\n"
        remaining = max_chars - total
        chunk = (header + content)[:remaining]
        parts.append(chunk)
        total += len(chunk)

    return "".join(parts)


def collect_question_focus_excerpt(repo_path):
    repo_root = Path(repo_path)
    max_chars = getattr(settings, "CODE_ANALYSIS_QUESTION_PROMPT_CHARS", 12000)
    max_file_bytes = 200000
    max_files = getattr(settings, "CODE_ANALYSIS_QUESTION_MAX_FILES", 8)
    total = 0
    parts = []

    files = list(iter_code_files(repo_path))
    ranked_files = sorted(files, key=_question_file_score, reverse=True)

    for file_path in ranked_files[:max_files]:
        if total >= max_chars:
            break
        try:
            if file_path.stat().st_size > max_file_bytes:
                continue
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        header = f"\n# Focus File: {file_path.relative_to(repo_root)}\n"
        remaining = max_chars - total
        chunk = (header + content)[:remaining]
        parts.append(chunk)
        total += len(chunk)

    return "".join(parts)


def run_build_command(command, repo_path):
    if not command:
        return

    subprocess.run(
        command,
        cwd=repo_path,
        shell=True,
        check=True,
    )


def _scanner_extra_args():
    extra = []
    python_version = os.getenv("CODE_ANALYSIS_PYTHON_VERSION")
    if python_version:
        extra.append(f"-Dsonar.python.version={python_version}")

    tests = os.getenv("CODE_ANALYSIS_TEST_PATHS")
    if tests:
        extra.append(f"-Dsonar.tests={tests}")

    exclusions = os.getenv("CODE_ANALYSIS_EXCLUSIONS")
    if exclusions:
        extra.append(f"-Dsonar.exclusions={exclusions}")

    return extra


def _question_file_score(file_path):
    path_text = str(file_path).lower()
    file_name = file_path.name.lower()

    if file_name.startswith("test") or "test" in file_name or "spec" in file_name:
        return -100

    keywords = {
        "action": 6,
        "api": 5,
        "route": 5,
        "service": 5,
        "controller": 5,
        "helper": 5,
        "util": 4,
        "model": 4,
        "schema": 4,
        "form": 4,
        "auth": 4,
        "login": 4,
        "submit": 4,
        "analysis": 4,
        "question": 4,
        "page": 3,
        "component": 3,
        "job": 2,
        "listing": 2,
        "submission": 2,
    }

    score = 0
    for keyword, weight in keywords.items():
        if keyword in path_text or keyword in file_name:
            score += weight

    if file_path.suffix.lower() in {".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".go", ".php"}:
        score += 1

    return score


def compute_quality_status(sonar_summary):
    if not sonar_summary:
        return CodeSubmission.QualityStatus.UNKNOWN, "No Sonar summary available."

    max_rating = float(getattr(settings, "CODE_ANALYSIS_MAX_RATING", 2))
    min_coverage = float(getattr(settings, "CODE_ANALYSIS_MIN_COVERAGE", 0))
    max_duplication = float(getattr(settings, "CODE_ANALYSIS_MAX_DUPLICATION", 5))

    reasons = []

    ratings = {
        "reliability_rating": "Reliability",
        "security_rating": "Security",
        "sqale_rating": "Maintainability",
    }

    for metric, label in ratings.items():
        value = _measure_float(sonar_summary, metric)
        if value is None:
            continue
        if value > max_rating:
            reasons.append(f"{label} rating {value} exceeds {max_rating}")

    coverage = _measure_float(sonar_summary, "coverage")
    if coverage is not None and coverage < min_coverage:
        reasons.append(f"Coverage {coverage}% below {min_coverage}%")

    duplication = _measure_float(sonar_summary, "duplicated_lines_density")
    if duplication is not None and duplication > max_duplication:
        reasons.append(f"Duplication {duplication}% above {max_duplication}%")

    if not reasons:
        return CodeSubmission.QualityStatus.PASSED, "All thresholds satisfied."

    return CodeSubmission.QualityStatus.FAILED, "; ".join(reasons)


def _measure_float(sonar_summary, metric_key):
    measures = sonar_summary.get("measures", []) if sonar_summary else []
    for measure in measures:
        if measure.get("metric") == metric_key:
            try:
                return float(measure.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def _summary_needs_refresh(sonar_summary):
    if not sonar_summary:
        return True

    return any(
        _measure_float(sonar_summary, metric_key) is None
        for metric_key in (
            "reliability_rating",
            "sqale_rating",
            "security_rating",
            "security_hotspots",
            "security_hotspots_reviewed",
        )
    )


def read_sonar_task_id(repo_path):
    task_file = Path(repo_path, ".scannerwork", "report-task.txt")
    if not task_file.exists():
        return None

    try:
        content = task_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    for line in content.splitlines():
        if line.startswith("ceTaskId="):
            return line.split("=", 1)[-1].strip()

    return None
