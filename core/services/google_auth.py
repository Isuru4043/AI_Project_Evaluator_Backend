import json
import os
import tempfile
from pathlib import Path


def configure_google_credentials() -> None:
    credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    if not credentials_json:
        return

    credentials_data = json.loads(credentials_json)

    credentials_path = (
        Path(tempfile.gettempdir()) / "google-credentials.json"
    )

    credentials_path.write_text(
        json.dumps(credentials_data),
        encoding="utf-8",
    )

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
        credentials_path
    )