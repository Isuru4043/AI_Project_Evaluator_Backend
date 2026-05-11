import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from django.conf import settings


DEFAULT_ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".go",
    ".rb",
    ".php",
    ".kt",
    ".swift",
    ".rs",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".xml",
    ".html",
    ".css",
}

EXCLUDED_DIRS = {
    ".git",
    ".scannerwork",
    "node_modules",
    "venv",
    ".venv",
    "dist",
    "build",
    "target",
    "__pycache__",
}


def clone_repo(repo_url):
    temp_dir = tempfile.mkdtemp(prefix="code_repo_")
    git_bin = os.getenv("GIT_BIN", "git")
    subprocess.run(
        [git_bin, "clone", "--depth", "1", repo_url, temp_dir],
        check=True,
        capture_output=True,
        text=True,
    )
    return temp_dir


def safe_extract_zip(zip_path):
    temp_dir = tempfile.mkdtemp(prefix="code_zip_")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.infolist():
            member_path = Path(temp_dir, member.filename)
            if not str(member_path.resolve()).startswith(str(Path(temp_dir).resolve())):
                raise ValueError("Invalid zip entry path.")
            zip_ref.extract(member, temp_dir)
    return temp_dir


def cleanup_path(path):
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


def iter_code_files(base_dir):
    allowed_exts = set(
        getattr(settings, "CODE_ANALYSIS_ALLOWED_EXTENSIONS", DEFAULT_ALLOWED_EXTENSIONS)
    )
    for root, dirs, files in os.walk(base_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for file_name in files:
            file_path = Path(root, file_name)
            if file_path.suffix.lower() in allowed_exts:
                yield file_path
