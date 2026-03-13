from __future__ import annotations

import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
MAX_MESSAGE_LENGTH = 300
MAX_TITLE_LENGTH = 100

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tims_frontier.reporting.notify import log_line, send_notification


def load_payload() -> dict:
    if len(sys.argv) < 2:
        return {}
    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        log_line("invalid JSON payload")
        return {}
    if not isinstance(payload, dict):
        log_line("payload is not a JSON object")
        return {}
    return payload


def should_notify(payload: dict) -> bool:
    return payload.get("type") == "agent-turn-complete"


def _workspace_name(payload: dict) -> str | None:
    cwd = payload.get("cwd")
    if not cwd:
        return None
    try:
        return Path(str(cwd)).name or None
    except Exception:
        return None


def _run_id(payload: dict) -> str | None:
    direct_keys = ("run_id", "run-id", "runId")
    for key in direct_keys:
        value = payload.get(key)
        if value:
            return str(value)

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key in direct_keys:
            value = metadata.get(key)
            if value:
                return str(value)
    return None


def build_title(payload: dict) -> str:
    parts = ["Codex"]
    workspace = _workspace_name(payload)
    run_id = _run_id(payload)
    if workspace:
        parts.append(workspace)
    if run_id:
        parts.append(run_id)
    title = " | ".join(parts)
    if len(title) > MAX_TITLE_LENGTH:
        title = title[: MAX_TITLE_LENGTH - 3] + "..."
    return title


def build_message(payload: dict) -> str:
    message = payload.get("last-assistant-message") or "Codex turn complete."
    message = " ".join(str(message).split())
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[: MAX_MESSAGE_LENGTH - 3] + "..."
    return message


def main() -> int:
    payload = load_payload()
    payload_type = payload.get("type", "<missing>")
    log_line(f"notify.py invoked type={payload_type}")
    if not should_notify(payload):
        log_line("skipped notification")
        return 0

    title = build_title(payload)
    message = build_message(payload)
    return 0 if send_notification(title, message) else 1


if __name__ == "__main__":
    raise SystemExit(main())
