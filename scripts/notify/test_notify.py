from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTIFY_SCRIPT = Path(__file__).resolve().parent / "codex_notify.py"


def main() -> int:
    payload = {
        "type": "agent-turn-complete",
        "last-assistant-message": "This is a repository-scoped Codex notification test.",
        "cwd": str(REPO_ROOT),
        "run_id": "repo_notify_test",
    }
    subprocess.run(
        [sys.executable, str(NOTIFY_SCRIPT), json.dumps(payload, ensure_ascii=False)],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
