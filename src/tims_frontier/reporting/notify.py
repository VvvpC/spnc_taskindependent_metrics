from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import tomllib
from typing import Any, Mapping

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_FILE = REPO_ROOT / "scripts" / "notify" / "notify_log.txt"
SECRETS_PATH = Path.home() / ".codex" / "notify_secrets.toml"
MAX_MESSAGE_LENGTH = 300
MAX_TITLE_LENGTH = 100


def log_line(message: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def _truncate(value: str, limit: int) -> str:
    compact = " ".join(str(value).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def load_secrets() -> tuple[str | None, str | None]:
    app_token = os.getenv("PUSHOVER_APP_TOKEN")
    user_key = os.getenv("PUSHOVER_USER_KEY")
    if app_token and user_key:
        return app_token, user_key

    if not SECRETS_PATH.exists():
        return app_token, user_key

    with SECRETS_PATH.open("rb") as handle:
        data = tomllib.load(handle)

    pushover = data.get("pushover", {})
    return app_token or pushover.get("app_token"), user_key or pushover.get("user_key")


def send_notification(title: str, message: str) -> bool:
    app_token, user_key = load_secrets()
    if not app_token or not user_key:
        log_line(f"missing Pushover credentials at {SECRETS_PATH}")
        return False

    safe_title = _truncate(title, MAX_TITLE_LENGTH)
    safe_message = _truncate(message, MAX_MESSAGE_LENGTH)

    try:
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": app_token,
                "user": user_key,
                "title": safe_title,
                "message": safe_message,
                "priority": 0,
            },
            timeout=15,
        )
        response.raise_for_status()
    except Exception as exc:
        log_line(f"notification failed: {type(exc).__name__}: {exc}")
        return False

    log_line(f"notification sent title={safe_title}")
    return True


def _format_family_counts(family: str, counts: Mapping[str, Any]) -> str:
    attempted = counts.get("attempted", 0)
    completed = counts.get("completed", 0)
    failed = counts.get("failed", 0)
    pruned = counts.get("pruned", 0)
    return (
        f"{family}: attempted={attempted}, completed={completed}, "
        f"failed={failed}, pruned={pruned}"
    )


def build_run_completion_title(*, run_id: str, status: str) -> str:
    return f"TIMs Study {status} | {REPO_ROOT.name} | {run_id}"


def build_run_completion_message(
    *,
    study_id: str,
    status: str,
    trial_counts: Mapping[str, Any] | None = None,
    extra_message: str | None = None,
) -> str:
    parts = [f"study={study_id}", f"status={status}"]
    if trial_counts:
        for family in ("uniform", "heterogeneous"):
            counts = trial_counts.get(family)
            if isinstance(counts, Mapping):
                parts.append(_format_family_counts(family, counts))
    if extra_message:
        parts.append(f"detail={extra_message}")
    return " | ".join(parts)


def send_run_completion_notification(
    *,
    study_id: str,
    run_id: str,
    status: str,
    trial_counts: Mapping[str, Any] | None = None,
    extra_message: str | None = None,
) -> bool:
    title = build_run_completion_title(run_id=run_id, status=status)
    message = build_run_completion_message(
        study_id=study_id,
        status=status,
        trial_counts=trial_counts,
        extra_message=extra_message,
    )
    return send_notification(title, message)
