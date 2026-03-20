from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SOURCE_SUBDIRS = [
    "src",
    "src/Morphology_Research",
    "src/Project",
    "src/Optuna_TaskIndependent_Metrics",
    "src/ParetoFront_CQandMC",
    "src/Plot_Functions",
    "src/Test_Temporary",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: Any) -> str:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    return output_path.as_posix()


def read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    input_path = Path(path)
    if not input_path.exists():
        return None
    with input_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return loaded


def load_structured_file(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".json":
        with input_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif input_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"PyYAML is required to read {input_path}. Use a JSON config in this environment instead."
            ) from exc
        with input_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    else:
        raise ValueError(f"Unsupported config format for {input_path}")
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object mapping in {input_path}")
    return loaded


def tail_lines(path: str | Path, *, n_lines: int) -> list[str]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return [line.rstrip("\n") for line in lines[-n_lines:]]


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def derive_seed_bundle(
    *,
    global_seed: int,
    family: str,
    family_trial_index: int,
    matched_across_families: bool = False,
) -> dict[str, int]:
    trial_scope = (global_seed, family_trial_index, "trial")
    if not matched_across_families:
        trial_scope = (global_seed, family, family_trial_index, "trial")
    trial_seed = stable_seed(*trial_scope)
    return {
        "trial_seed": trial_seed,
        "input_signal_seed": stable_seed(trial_seed, "input_signal"),
        "mask_seed": stable_seed(trial_seed, "mask"),
        "morphology_seed": stable_seed(trial_seed, "morphology"),
    }


def bootstrap_legacy_source_paths() -> list[str]:
    inserted: list[str] = []
    for relative_path in reversed(LEGACY_SOURCE_SUBDIRS):
        absolute_path = (REPO_ROOT / relative_path).resolve()
        absolute_text = absolute_path.as_posix()
        if absolute_text in sys.path:
            continue
        sys.path.insert(0, absolute_text)
        inserted.append(absolute_text)
    return inserted
