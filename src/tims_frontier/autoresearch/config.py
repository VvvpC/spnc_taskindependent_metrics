from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .common import REPO_ROOT, load_structured_file, utc_now_iso


DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "autoresearch_v1" / "default.json"


def timestamp_run_id() -> str:
    return utc_now_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")


def _resolve_relative_path(repo_root: Path, raw_value: str) -> str:
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate.as_posix()
    return (repo_root / candidate).resolve().as_posix()


def load_autoresearch_config(config_path: str | None = None) -> dict[str, Any]:
    resolved_path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not resolved_path.is_absolute():
        resolved_path = (REPO_ROOT / resolved_path).resolve()
    loaded = load_structured_file(resolved_path)
    config = deepcopy(loaded)
    storage = config.setdefault("storage", {})
    storage["config_path"] = resolved_path.as_posix()
    storage["run_root"] = _resolve_relative_path(REPO_ROOT, str(storage["run_root"]))
    storage["processed_root"] = _resolve_relative_path(REPO_ROOT, str(storage["processed_root"]))
    storage["active_run_pointer"] = _resolve_relative_path(REPO_ROOT, str(storage["active_run_pointer"]))
    return config


def build_run_dir(config: Mapping[str, Any], *, run_id: str) -> str:
    return (Path(str(config["storage"]["run_root"])) / run_id).as_posix()
