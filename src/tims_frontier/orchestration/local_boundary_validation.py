from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tims_frontier.orchestration.parameter_pair_sweep import run_parameter_pair_sweep_study
from tims_frontier.reporting import send_run_completion_notification


REPO_ROOT = Path(__file__).resolve().parents[3]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_local_boundary_definition(
    study_config: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    mapping_cfg = study_config.get("local_boundary_validation")
    if not isinstance(mapping_cfg, Mapping):
        raise ValueError("Local boundary validation study requires a top-level 'local_boundary_validation' mapping.")

    windows = mapping_cfg.get("windows")
    if not isinstance(windows, Mapping) or not windows:
        raise ValueError("'local_boundary_validation.windows' must be a non-empty mapping.")

    window_order = mapping_cfg.get("window_order")
    if window_order is None:
        ordered_windows = [str(name) for name in windows.keys()]
    else:
        if not isinstance(window_order, Sequence) or isinstance(window_order, (str, bytes)):
            raise ValueError("'local_boundary_validation.window_order' must be a list of window names.")
        ordered_windows = [str(name) for name in window_order]
        if set(ordered_windows) != set(windows.keys()):
            raise ValueError("'local_boundary_validation.window_order' must match the window keys exactly.")

    resolved_windows: dict[str, dict[str, Any]] = {}
    for window_name in ordered_windows:
        entry = windows[window_name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Window '{window_name}' must map to a config object.")

        x_parameter = str(entry["x_parameter"])
        y_parameter = str(entry["y_parameter"])
        if x_parameter == y_parameter:
            raise ValueError(f"Window '{window_name}' must reference two different parameters.")

        x_cfg = entry.get("x")
        y_cfg = entry.get("y")
        if not isinstance(x_cfg, Mapping) or not isinstance(y_cfg, Mapping):
            raise ValueError(f"Window '{window_name}' must provide both 'x' and 'y' range configs.")

        fixed_values = entry.get("fixed", {})
        if not isinstance(fixed_values, Mapping):
            raise ValueError(f"Window '{window_name}.fixed' must be a mapping when provided.")

        resolved_windows[window_name] = {
            "pair": str(entry["pair"]),
            "x_parameter": x_parameter,
            "x": {
                "kind": str(x_cfg["kind"]),
                "low": float(x_cfg["low"]),
                "high": float(x_cfg["high"]),
                "num_points": int(x_cfg["num_points"]),
            },
            "y_parameter": y_parameter,
            "y": {
                "kind": str(y_cfg["kind"]),
                "low": float(y_cfg["low"]),
                "high": float(y_cfg["high"]),
                "num_points": int(y_cfg["num_points"]),
            },
            "fixed": dict(fixed_values),
        }

    return ordered_windows, resolved_windows


def _derive_child_study_id(base_study_id: str, window_name: str) -> str:
    return f"{base_study_id}_{window_name}"


def _derive_child_run_id(base_run_id: str | None, window_name: str) -> str:
    prefix = base_run_id or "local_boundary_validation"
    return f"{prefix}_{window_name}"


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _results_root(meta_study_id: str, aggregate_run_id: str) -> Path:
    raw_root = REPO_ROOT / "results" / "raw" / meta_study_id / aggregate_run_id
    raw_root.mkdir(parents=True, exist_ok=True)
    return raw_root


def _prepare_child_config(
    study_config: Mapping[str, Any],
    *,
    base_study_id: str,
    window_name: str,
    window_definition: Mapping[str, Any],
) -> dict[str, Any]:
    child = deepcopy(dict(study_config))
    x_parameter = str(window_definition["x_parameter"])
    y_parameter = str(window_definition["y_parameter"])

    child["study"]["study_id"] = _derive_child_study_id(base_study_id, window_name)
    child["study"]["title"] = f"{study_config['study']['title']} | {window_name}"
    child["study"]["description"] = (
        f"{study_config['study']['description']} (window={window_name}, pair={window_definition['pair']})"
    )
    child["study"]["tags"] = list(child["study"].get("tags", [])) + ["local-boundary-validation-child"]
    child["exploration"]["method"] = "parameter_pair_sweep"
    child["exploration"]["study_mode"] = "paired_parameter_grid_sweep"
    child["exploration"]["optuna"]["n_trials_per_family"] = int(
        window_definition["x"]["num_points"]
    ) * int(window_definition["y"]["num_points"])

    child["parameter_pair_sweep"] = {
        "pair_order": [window_name],
        "parameters": {
            x_parameter: deepcopy(window_definition["x"]),
            y_parameter: deepcopy(window_definition["y"]),
        },
        "pairs": {
            window_name: {
                "x_parameter": x_parameter,
                "y_parameter": y_parameter,
            }
        },
    }

    for fixed_parameter, fixed_value in dict(window_definition.get("fixed", {})).items():
        if fixed_parameter == "beta_spread":
            child["families"]["heterogeneous"]["construction"]["morphology"]["beta_spread"]["value"] = float(
                fixed_value
            )
        elif fixed_parameter == "beta_prime":
            child["exploration"]["shared_non_geometric_search"]["beta_prime"]["value"] = float(fixed_value)
        elif fixed_parameter == "theta":
            child["exploration"]["shared_non_geometric_search"]["theta"]["value"] = float(fixed_value)
        elif fixed_parameter == "gamma":
            child["exploration"]["shared_non_geometric_search"]["gamma"]["value"] = float(fixed_value)
        elif fixed_parameter == "m0":
            child["exploration"]["shared_non_geometric_search"]["m0"]["value"] = float(fixed_value)
        elif fixed_parameter == "n_instances":
            child["families"]["heterogeneous"]["construction"]["morphology"]["n_instances"]["value"] = int(
                fixed_value
            )
        else:
            raise ValueError(f"Unsupported fixed parameter for local-validation window: {fixed_parameter}")

    return child


def run_local_boundary_validation_study(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Run a window-based local boundary validation meta-study."""

    window_order, windows = _load_local_boundary_definition(study_config)
    base_study_id = str(study_config["study"]["study_id"])
    aggregate_run_id = run_id or f"{base_study_id}_meta"
    started_at_utc = _utc_now_iso()

    child_runs: list[dict[str, Any]] = []
    for window_name in window_order:
        child_config = _prepare_child_config(
            study_config,
            base_study_id=base_study_id,
            window_name=window_name,
            window_definition=windows[window_name],
        )
        child_summary = run_parameter_pair_sweep_study(
            child_config,
            source_config_path=source_config_path,
            run_id=_derive_child_run_id(aggregate_run_id, window_name),
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            notify_on_completion=False,
        )
        child_runs.append(
            {
                "window_name": window_name,
                "pair": windows[window_name]["pair"],
                "x_parameter": windows[window_name]["x_parameter"],
                "x_range": dict(windows[window_name]["x"]),
                "y_parameter": windows[window_name]["y_parameter"],
                "y_range": dict(windows[window_name]["y"]),
                "fixed": dict(windows[window_name].get("fixed", {})),
                "study_id": child_summary["study_id"],
                "run_id": child_summary["run_id"],
                "trial_counts": child_summary["canonical_summary"]["trial_counts"],
                "canonical_artifacts": child_summary["canonical_summary"]["artifacts"],
                "parameter_pair_artifacts": child_summary["parameter_pair_artifacts"],
                "pair_sweep_manifest_path": child_summary["pair_sweep_manifest_path"],
                "parameter_pair_sweep_summary_path": child_summary["parameter_pair_sweep_summary_path"],
            }
        )

    finished_at_utc = _utc_now_iso()
    raw_root = _results_root(base_study_id, aggregate_run_id)
    manifest = {
        "meta_study_id": base_study_id,
        "aggregate_run_id": aggregate_run_id,
        "status": "completed",
        "window_order": window_order,
        "window_count": len(window_order),
        "windows": windows,
        "child_runs": child_runs,
        "provenance": {
            "git_commit": git_commit,
            "git_branch": git_branch,
            "python_version": python_version,
            "hostname": hostname,
        },
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }
    manifest_path = _write_json(raw_root / "local_boundary_validation_manifest.json", manifest)

    send_run_completion_notification(
        study_id=base_study_id,
        run_id=aggregate_run_id,
        status="completed",
        extra_message=f"window_count={len(window_order)}",
    )

    return {
        "meta_study_id": base_study_id,
        "aggregate_run_id": aggregate_run_id,
        "source_config_path": source_config_path,
        "window_order": window_order,
        "artifacts": {
            "local_boundary_validation_manifest": manifest_path,
        },
        "child_runs": child_runs,
    }
