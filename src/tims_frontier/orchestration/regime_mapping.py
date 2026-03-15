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


def _load_regime_mapping_definition(
    study_config: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    mapping_cfg = study_config.get("regime_mapping")
    if not isinstance(mapping_cfg, Mapping):
        raise ValueError("Regime-mapping study requires a top-level 'regime_mapping' mapping.")

    parameters = mapping_cfg.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("'regime_mapping.parameters' must be a non-empty mapping.")

    blocks = mapping_cfg.get("blocks")
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("'regime_mapping.blocks' must be a non-empty mapping.")

    block_order = mapping_cfg.get("block_order")
    if block_order is None:
        ordered_blocks = [str(name) for name in blocks.keys()]
    else:
        if not isinstance(block_order, Sequence) or isinstance(block_order, (str, bytes)):
            raise ValueError("'regime_mapping.block_order' must be a list of block names.")
        ordered_blocks = [str(name) for name in block_order]
        if set(ordered_blocks) != set(blocks.keys()):
            raise ValueError("'regime_mapping.block_order' must match the block keys exactly.")

    resolved_parameters: dict[str, dict[str, Any]] = {}
    for name, entry in parameters.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Regime-mapping parameter '{name}' must map to a config object.")
        resolved_parameters[str(name)] = {
            "kind": str(entry["kind"]),
            "low": float(entry["low"]),
            "high": float(entry["high"]),
            "num_points": int(entry["num_points"]),
        }

    resolved_blocks: dict[str, dict[str, Any]] = {}
    for block_name in ordered_blocks:
        entry = blocks[block_name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Block '{block_name}' must map to a config object.")
        x_parameter = str(entry["x_parameter"])
        y_parameter = str(entry["y_parameter"])
        if x_parameter not in resolved_parameters or y_parameter not in resolved_parameters:
            raise ValueError(f"Block '{block_name}' references unknown parameters: {x_parameter}, {y_parameter}")
        if x_parameter == y_parameter:
            raise ValueError(f"Block '{block_name}' must reference two different parameters.")
        fixed_values = entry.get("fixed", {})
        if not isinstance(fixed_values, Mapping):
            raise ValueError(f"Block '{block_name}.fixed' must be a mapping when provided.")
        resolved_blocks[block_name] = {
            "x_parameter": x_parameter,
            "y_parameter": y_parameter,
            "fixed": dict(fixed_values),
        }

    return ordered_blocks, resolved_parameters, resolved_blocks


def _derive_child_study_id(base_study_id: str, block_name: str) -> str:
    return f"{base_study_id}_{block_name}"


def _derive_child_run_id(base_run_id: str | None, block_name: str) -> str:
    prefix = base_run_id or "regime_mapping"
    return f"{prefix}_{block_name}"


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
    block_name: str,
    parameters: Mapping[str, dict[str, Any]],
    block_definition: Mapping[str, Any],
) -> dict[str, Any]:
    child = deepcopy(dict(study_config))
    x_parameter = str(block_definition["x_parameter"])
    y_parameter = str(block_definition["y_parameter"])

    child["study"]["study_id"] = _derive_child_study_id(base_study_id, block_name)
    child["study"]["title"] = f"{study_config['study']['title']} | {block_name}"
    child["study"]["description"] = (
        f"{study_config['study']['description']} (block={block_name})"
    )
    child["study"]["tags"] = list(child["study"].get("tags", [])) + ["regime-mapping-child"]
    child["exploration"]["method"] = "parameter_pair_sweep"
    child["exploration"]["study_mode"] = "paired_parameter_grid_sweep"
    child["exploration"]["optuna"]["n_trials_per_family"] = 121

    child["parameter_pair_sweep"] = {
        "pair_order": [block_name],
        "parameters": {
            x_parameter: deepcopy(parameters[x_parameter]),
            y_parameter: deepcopy(parameters[y_parameter]),
        },
        "pairs": {
            block_name: {
                "x_parameter": x_parameter,
                "y_parameter": y_parameter,
            }
        },
    }

    for fixed_parameter, fixed_value in dict(block_definition.get("fixed", {})).items():
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
            raise ValueError(f"Unsupported fixed parameter for regime-mapping block: {fixed_parameter}")

    return child


def run_regime_mapping_study(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Run a high-resolution regime-mapping meta-study across selected parameter-pair blocks."""

    block_order, parameters, blocks = _load_regime_mapping_definition(study_config)
    base_study_id = str(study_config["study"]["study_id"])
    aggregate_run_id = run_id or f"{base_study_id}_meta"
    started_at_utc = _utc_now_iso()

    child_runs: list[dict[str, Any]] = []
    for block_name in block_order:
        child_config = _prepare_child_config(
            study_config,
            base_study_id=base_study_id,
            block_name=block_name,
            parameters=parameters,
            block_definition=blocks[block_name],
        )
        child_summary = run_parameter_pair_sweep_study(
            child_config,
            source_config_path=source_config_path,
            run_id=_derive_child_run_id(aggregate_run_id, block_name),
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            notify_on_completion=False,
        )
        child_runs.append(
            {
                "block_name": block_name,
                "x_parameter": blocks[block_name]["x_parameter"],
                "y_parameter": blocks[block_name]["y_parameter"],
                "fixed": dict(blocks[block_name].get("fixed", {})),
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
        "block_order": block_order,
        "block_count": len(block_order),
        "parameters": parameters,
        "blocks": blocks,
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
    manifest_path = _write_json(raw_root / "regime_mapping_manifest.json", manifest)

    send_run_completion_notification(
        study_id=base_study_id,
        run_id=aggregate_run_id,
        status="completed",
        extra_message=f"block_count={len(block_order)}",
    )

    return {
        "meta_study_id": base_study_id,
        "aggregate_run_id": aggregate_run_id,
        "source_config_path": source_config_path,
        "block_order": block_order,
        "artifacts": {
            "regime_mapping_manifest": manifest_path,
        },
        "child_runs": child_runs,
    }
