import os
import time
import subprocess

import requests


class SonarQubeService:
    def __init__(self):
        self.host = os.getenv("SONAR_HOST_URL", "https://sonarcloud.io")
        self.organization = os.getenv("SONAR_ORG_KEY")
        self.token = os.getenv("SONAR_TOKEN")
        self.scanner_bin = os.getenv("SONAR_SCANNER_BIN", "sonar-scanner")

        self.session = requests.Session()
        if self.token:
            self.session.auth = (self.token, "")

    def _assert_configured(self):
        if not self.organization:
            raise ValueError("SONAR_ORG_KEY is not configured.")
        if not self.token:
            raise ValueError("SONAR_TOKEN is not configured.")

    def ensure_project(self, project_key, project_name):
        self._assert_configured()
        if self._project_exists(project_key):
            return

        response = self.session.post(
            f"{self.host}/api/projects/create",
            data={
                "organization": self.organization,
                "project": project_key,
                "name": project_name,
            },
            timeout=30,
        )
        response.raise_for_status()

    def _project_exists(self, project_key):
        response = self.session.get(
            f"{self.host}/api/projects/search",
            params={
                "projects": project_key,
                "organization": self.organization,
            },
            timeout=30,
        )
        response.raise_for_status()
        return bool(response.json().get("components"))

    def run_scanner(self, project_key, base_dir, extra_args=None):
        self._assert_configured()

        command = [
            self.scanner_bin,
            f"-Dsonar.projectKey={project_key}",
            f"-Dsonar.organization={self.organization}",
            f"-Dsonar.host.url={self.host}",
            f"-Dsonar.token={self.token}",
            "-Dsonar.sources=.",
        ]

        if extra_args:
            command.extend(extra_args)

        subprocess.run(
            command,
            cwd=base_dir,
            check=True,
        )

    def get_summary(self, project_key):
        self._assert_configured()
        metrics = (
            "bugs,vulnerabilities,code_smells,security_rating,reliability_rating,"
            "sqale_rating,security_hotspots,security_hotspots_reviewed,coverage,duplicated_lines_density,ncloc"
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                measures = self.session.get(
                    f"{self.host}/api/measures/component",
                    params={
                        "component": project_key,
                        "organization": self.organization,
                        "metricKeys": metrics,
                    },
                    timeout=30,
                )
                measures.raise_for_status()
                break
            except requests.exceptions.HTTPError as exc:
                if exc.response.status_code == 404 and attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise

        measures_data = measures.json()

        issues = self.session.get(
            f"{self.host}/api/issues/search",
            params={
                "componentKeys": project_key,
                "organization": self.organization,
                "types": "BUG,VULNERABILITY,CODE_SMELL",
                "s": "SEVERITY",
                "ps": 20,
            },
            timeout=30,
        )
        issues.raise_for_status()
        issues_data = issues.json()

        return {
            "measures": measures_data.get("component", {}).get("measures", []),
            "issues": issues_data.get("issues", []),
            "total_issues": issues_data.get("total"),
        }

    def get_task_status(self, task_id):
        self._assert_configured()
        response = self.session.get(
            f"{self.host}/api/ce/task",
            params={"id": task_id},
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("task", {})

    def wait_for_analysis(self, project_key, timeout_seconds=None):
        self._assert_configured()
        if timeout_seconds is None:
            timeout_seconds = int(os.getenv("CODE_ANALYSIS_SONAR_TIMEOUT", "300"))
        start = time.time()
        while time.time() - start < timeout_seconds:
            response = self.session.get(
                f"{self.host}/api/ce/component",
                params={"component": project_key},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("current"):
                return
            time.sleep(3)

        raise TimeoutError("SonarCloud analysis did not finish in time.")

    def dashboard_url(self, project_key):
        return f"{self.host}/dashboard?id={project_key}"
