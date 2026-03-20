from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable


class GitOperationError(RuntimeError):
    """Raised when a git subprocess fails."""


def _run_git(repo_root: str | Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitOperationError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


def current_branch(repo_root: str | Path) -> str:
    return _run_git(repo_root, "branch", "--show-current").stdout.strip()


def current_commit(repo_root: str | Path, *, short: bool = False) -> str:
    args = ("rev-parse", "--short", "HEAD") if short else ("rev-parse", "HEAD")
    return _run_git(repo_root, *args).stdout.strip()


def status_entries(repo_root: str | Path) -> list[dict[str, str]]:
    result = _run_git(repo_root, "status", "--porcelain", "--untracked-files=all")
    entries: list[dict[str, str]] = []
    for raw_line in result.stdout.splitlines():
        if not raw_line:
            continue
        status = raw_line[:2]
        path_text = raw_line[3:]
        if " -> " in path_text:
            path_text = path_text.split(" -> ", maxsplit=1)[1]
        entries.append({"status": status, "path": path_text})
    return entries


def validate_train_only_changes(repo_root: str | Path, *, editable_paths: Iterable[str]) -> list[str]:
    allowed = {Path(item).as_posix() for item in editable_paths}
    disallowed: list[str] = []
    for entry in status_entries(repo_root):
        path_text = Path(entry["path"]).as_posix()
        if path_text in allowed:
            continue
        disallowed.append(path_text)
    return disallowed


def add_and_commit(repo_root: str | Path, *, pathspecs: list[str], message: str) -> str:
    _run_git(repo_root, "add", "--", *pathspecs)
    _run_git(repo_root, "commit", "--no-verify", "-m", message)
    return current_commit(repo_root)


def hard_reset_to(repo_root: str | Path, commit: str) -> str:
    _run_git(repo_root, "reset", "--hard", commit)
    return current_commit(repo_root)
