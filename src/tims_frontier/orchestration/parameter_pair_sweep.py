from __future__ import annotations

import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import time
import traceback
from typing import Any, Mapping, Sequence

from tims_frontier.construction.builders import build_trial_spec
from tims_frontier.evaluation.tims import evaluate_trial_tims
from tims_frontier.orchestration.runner import finalize_run, initialize_run, persist_trial_outputs
from tims_frontier.orchestration.seeds import derive_seed_bundle
from tims_frontier.reporting import send_run_completion_notification
from tims_frontier.storage.records import make_failure_record, make_trial_record


PAIR_RESULT_FIELDNAMES = [
    "study_id",
    "run_id",
    "pair_name",
    "grid_index",
    "x_parameter",
    "x_index",
    "x_value",
    "y_parameter",
    "y_index",
    "y_value",
    "family",
    "family_role",
    "trial_id",
    "status",
    "failure_code",
    "failure_message",
    "nominal_beta_prime",
    "nominal_theta",
    "nominal_gamma",
    "nominal_m0",
    "nominal_n_instances",
    "nominal_beta_spread",
    "applied_beta_prime",
    "applied_theta",
    "applied_gamma",
    "applied_m0",
    "applied_n_instances",
    "applied_beta_spread",
    "Nvirt",
    "geometry_mode",
    "morphology_scheme",
    "morphology_weights_mode",
    "morphology_seed",
    "trial_seed",
    "input_signal_seed",
    "mask_seed",
    "MC",
    "KR",
    "GR",
    "CQ",
    "runtime_seconds",
    "started_at_utc",
    "finished_at_utc",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_traceback_file(logs_dir: str, trial_id: str, exception: BaseException) -> str:
    log_path = Path(logs_dir) / f"{trial_id}_traceback.txt"
    trace_text = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(trace_text, encoding="utf-8")
    return log_path.as_posix()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_RESULT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in PAIR_RESULT_FIELDNAMES})
    return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _build_baseline_nominal_values(resolved_config: Mapping[str, Any]) -> dict[str, float | int]:
    search_space = resolved_config["exploration"]["search_space"]
    hetero = resolved_config["families"]["heterogeneous"]["resolved_construction"]["morphology"]
    return {
        "beta_prime": float(search_space["beta_prime"]["value"]),
        "theta": float(search_space["theta"]["value"]),
        "gamma": float(search_space["gamma"]["value"]),
        "m0": float(search_space["m0"]["value"]),
        "n_instances": int(hetero["n_instances"]),
        "beta_spread": float(hetero["beta_spread"]),
    }


def _generate_parameter_values(*, kind: str, low: float, high: float, num_points: int) -> list[float | int]:
    if num_points == 1:
        values = [low]
    else:
        step = (high - low) / float(num_points - 1)
        values = [low + step * idx for idx in range(num_points)]

    if kind == "float":
        return [float(value) for value in values]
    if kind == "int":
        low_int = int(round(low))
        high_int = int(round(high))
        distinct_count = high_int - low_int + 1
        if distinct_count <= 0:
            raise ValueError(f"Invalid integer sweep bounds: [{low}, {high}]")
        if num_points >= distinct_count:
            return list(range(low_int, high_int + 1))

        rounded = [int(round(value)) for value in values]
        deduplicated: list[int] = []
        for value in rounded:
            if value not in deduplicated:
                deduplicated.append(value)
        if len(deduplicated) != num_points:
            raise ValueError(f"Integer sweep over [{low}, {high}] could not realize {num_points} distinct values.")
        return deduplicated
    raise ValueError(f"Unsupported parameter-pair sweep kind: {kind}")


def _resolve_parameter_pair_config(
    study_config: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, str]]]:
    pair_cfg = study_config.get("parameter_pair_sweep")
    if not isinstance(pair_cfg, Mapping):
        raise ValueError("Study config requires a top-level 'parameter_pair_sweep' mapping.")

    parameters = pair_cfg.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("'parameter_pair_sweep.parameters' must be a non-empty mapping.")

    pairs = pair_cfg.get("pairs")
    if not isinstance(pairs, Mapping) or not pairs:
        raise ValueError("'parameter_pair_sweep.pairs' must be a non-empty mapping.")

    pair_order = pair_cfg.get("pair_order")
    if pair_order is None:
        ordered_pairs = [str(name) for name in pairs.keys()]
    else:
        if not isinstance(pair_order, Sequence) or isinstance(pair_order, (str, bytes)):
            raise ValueError("'parameter_pair_sweep.pair_order' must be a list of pair names.")
        ordered_pairs = [str(name) for name in pair_order]
        if set(ordered_pairs) != set(pairs.keys()):
            raise ValueError("'parameter_pair_sweep.pair_order' must match the pair keys exactly.")

    resolved_parameters: dict[str, dict[str, Any]] = {}
    for name, entry in parameters.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Pair-sweep parameter '{name}' must map to a config object.")
        resolved_entry = {
            "kind": str(entry["kind"]),
            "low": float(entry["low"]),
            "high": float(entry["high"]),
            "num_points": int(entry["num_points"]),
        }
        resolved_entry["values"] = _generate_parameter_values(**resolved_entry)
        resolved_parameters[str(name)] = resolved_entry

    resolved_pairs: dict[str, dict[str, str]] = {}
    for pair_name in ordered_pairs:
        pair_entry = pairs[pair_name]
        if not isinstance(pair_entry, Mapping):
            raise ValueError(f"Pair '{pair_name}' must map to a config object.")
        x_parameter = str(pair_entry["x_parameter"])
        y_parameter = str(pair_entry["y_parameter"])
        if x_parameter not in resolved_parameters or y_parameter not in resolved_parameters:
            raise ValueError(f"Pair '{pair_name}' references unknown parameters: {x_parameter}, {y_parameter}")
        if x_parameter == y_parameter:
            raise ValueError(f"Pair '{pair_name}' must reference two different parameters.")
        resolved_pairs[pair_name] = {
            "x_parameter": x_parameter,
            "y_parameter": y_parameter,
        }

    return ordered_pairs, resolved_parameters, resolved_pairs


def _sampled_operational_params(nominal_values: Mapping[str, float | int]) -> dict[str, float]:
    return {
        "beta_prime": float(nominal_values["beta_prime"]),
        "theta": float(nominal_values["theta"]),
        "gamma": float(nominal_values["gamma"]),
        "m0": float(nominal_values["m0"]),
    }


def _heterogeneous_morphology_overrides(nominal_values: Mapping[str, float | int]) -> dict[str, Any]:
    return {
        "n_instances": int(nominal_values["n_instances"]),
        "beta_spread": float(nominal_values["beta_spread"]),
    }


def _make_trial_id(pair_name: str, grid_index: int, family: str) -> str:
    return f"{pair_name}_grid_{grid_index:02d}_{family}"


def _make_pair_row(
    *,
    resolved_config: Mapping[str, Any],
    pair_name: str,
    x_parameter: str,
    x_index: int,
    x_value: float | int,
    y_parameter: str,
    y_index: int,
    y_value: float | int,
    grid_index: int,
    build_spec: Any,
    trial_id: str,
    status: str,
    seed_bundle: Mapping[str, int],
    nominal_values: Mapping[str, float | int],
    metrics: Mapping[str, Any] | None = None,
    failure_code: str | None = None,
    failure_message: str | None = None,
    runtime_seconds: float | None = None,
    started_at_utc: str | None = None,
    finished_at_utc: str | None = None,
) -> dict[str, Any]:
    metrics = dict(metrics or {})
    morphology = build_spec.morphology
    reservoir_params = build_spec.reservoir_params
    return {
        "study_id": resolved_config["study"]["study_id"],
        "run_id": resolved_config["run"]["run_id"],
        "pair_name": pair_name,
        "grid_index": grid_index,
        "x_parameter": x_parameter,
        "x_index": x_index,
        "x_value": x_value,
        "y_parameter": y_parameter,
        "y_index": y_index,
        "y_value": y_value,
        "family": build_spec.family,
        "family_role": build_spec.family_role,
        "trial_id": trial_id,
        "status": status,
        "failure_code": failure_code,
        "failure_message": failure_message,
        "nominal_beta_prime": nominal_values["beta_prime"],
        "nominal_theta": nominal_values["theta"],
        "nominal_gamma": nominal_values["gamma"],
        "nominal_m0": nominal_values["m0"],
        "nominal_n_instances": nominal_values["n_instances"],
        "nominal_beta_spread": nominal_values["beta_spread"],
        "applied_beta_prime": reservoir_params.beta_prime,
        "applied_theta": reservoir_params.params["theta"],
        "applied_gamma": reservoir_params.params["gamma"],
        "applied_m0": reservoir_params.m0,
        "applied_n_instances": morphology.n_instances,
        "applied_beta_spread": morphology.beta_spread,
        "Nvirt": reservoir_params.Nvirt,
        "geometry_mode": build_spec.geometry_mode,
        "morphology_scheme": morphology.scheme,
        "morphology_weights_mode": morphology.weights_mode,
        "morphology_seed": morphology.morphology_seed,
        "trial_seed": seed_bundle["trial_seed"],
        "input_signal_seed": seed_bundle["input_signal_seed"],
        "mask_seed": seed_bundle["mask_seed"],
        "MC": metrics.get("MC"),
        "KR": metrics.get("KR"),
        "GR": metrics.get("GR"),
        "CQ": metrics.get("CQ"),
        "runtime_seconds": runtime_seconds,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
    }


def run_parameter_pair_sweep_study(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
    notify_on_completion: bool = True,
) -> dict[str, Any]:
    """Run a paired uniform-vs-heterogeneous parameter-pair TIMs sweep."""

    resolved_config, manifest = initialize_run(
        study_config,
        source_config_path=source_config_path,
        run_id=run_id,
    )
    baseline_nominal_values = _build_baseline_nominal_values(resolved_config)
    pair_order, parameter_specs, pair_definitions = _resolve_parameter_pair_config(study_config)

    global_seed = int(resolved_config["execution"]["seed_policy"]["global_seed"])
    matched_trial_seed_across_families = bool(
        resolved_config["execution"]["seed_policy"]["matched_trial_seed_across_families"]
    )
    logs_dir = str(resolved_config["storage"]["paths"]["logs_dir"])
    run_root = Path(resolved_config["storage"]["paths"]["run_root"])
    parameter_pairs_dir = run_root / "parameter_pairs"
    parameter_pairs_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: dict[str, list[dict[str, Any]]] = {pair_name: [] for pair_name in pair_order}
    artifact_paths: dict[str, str] = {}
    global_grid_counter = 0

    try:
        for pair_name in pair_order:
            pair_definition = pair_definitions[pair_name]
            x_parameter = pair_definition["x_parameter"]
            y_parameter = pair_definition["y_parameter"]
            x_values = parameter_specs[x_parameter]["values"]
            y_values = parameter_specs[y_parameter]["values"]

            for grid_index, ((x_index, x_value), (y_index, y_value)) in enumerate(
                itertools.product(enumerate(x_values, start=1), enumerate(y_values, start=1)),
                start=1,
            ):
                global_grid_counter += 1
                nominal_values = dict(baseline_nominal_values)
                nominal_values[x_parameter] = x_value
                nominal_values[y_parameter] = y_value

                for family in resolved_config["comparison"]["compared_families"]:
                    family = str(family)
                    trial_id = _make_trial_id(pair_name, grid_index, family)
                    started_at_utc = _utc_now_iso()
                    started_perf = time.perf_counter()
                    seed_bundle = derive_seed_bundle(
                        global_seed=global_seed,
                        family=family,
                        family_trial_index=global_grid_counter,
                        matched_across_families=matched_trial_seed_across_families,
                    )
                    sampled_params = _sampled_operational_params(nominal_values)
                    build_spec = build_trial_spec(
                        resolved_config,
                        family=family,
                        sampled_params=sampled_params,
                        seed_bundle=seed_bundle,
                        morphology_overrides=(
                            _heterogeneous_morphology_overrides(nominal_values)
                            if family == "heterogeneous"
                            else None
                        ),
                    )

                    try:
                        metrics = evaluate_trial_tims(build_spec, resolved_config, seed_bundle=seed_bundle)
                    except Exception as exc:
                        traceback_path = _write_traceback_file(logs_dir, trial_id, exc)
                        runtime_seconds = time.perf_counter() - started_perf
                        finished_at_utc = _utc_now_iso()
                        failed_record = make_trial_record(
                            resolved_config=resolved_config,
                            build_spec=build_spec,
                            trial_id=trial_id,
                            family_trial_index=global_grid_counter,
                            optuna_study_name=f"{pair_name}_parameter_pair_sweep",
                            optuna_trial_number=global_grid_counter - 1,
                            status="failed",
                            seed_bundle=seed_bundle,
                            sampled_params=sampled_params,
                            failure_code=type(exc).__name__,
                            failure_message=str(exc),
                            runtime_seconds=runtime_seconds,
                            started_at_utc=started_at_utc,
                            finished_at_utc=finished_at_utc,
                        )
                        failure_record = make_failure_record(
                            resolved_config=resolved_config,
                            trial_id=trial_id,
                            family=family,
                            optuna_trial_number=global_grid_counter - 1,
                            failure_code=type(exc).__name__,
                            exception_type=type(exc).__name__,
                            message=str(exc),
                            traceback_path=traceback_path,
                            trial_seed=int(seed_bundle["trial_seed"]),
                        )
                        persist_trial_outputs(
                            resolved_config,
                            manifest,
                            trial_record=failed_record,
                            failure_record=failure_record,
                        )
                        pair_rows[pair_name].append(
                            _make_pair_row(
                                resolved_config=resolved_config,
                                pair_name=pair_name,
                                x_parameter=x_parameter,
                                x_index=x_index,
                                x_value=x_value,
                                y_parameter=y_parameter,
                                y_index=y_index,
                                y_value=y_value,
                                grid_index=grid_index,
                                build_spec=build_spec,
                                trial_id=trial_id,
                                status="failed",
                                seed_bundle=seed_bundle,
                                nominal_values=nominal_values,
                                failure_code=type(exc).__name__,
                                failure_message=str(exc),
                                runtime_seconds=runtime_seconds,
                                started_at_utc=started_at_utc,
                                finished_at_utc=finished_at_utc,
                            )
                        )
                        continue

                    runtime_seconds = time.perf_counter() - started_perf
                    finished_at_utc = _utc_now_iso()
                    completed_record = make_trial_record(
                        resolved_config=resolved_config,
                        build_spec=build_spec,
                        trial_id=trial_id,
                        family_trial_index=global_grid_counter,
                        optuna_study_name=f"{pair_name}_parameter_pair_sweep",
                        optuna_trial_number=global_grid_counter - 1,
                        status="completed",
                        seed_bundle=seed_bundle,
                        sampled_params=sampled_params,
                        metrics=metrics,
                        runtime_seconds=runtime_seconds,
                        started_at_utc=started_at_utc,
                        finished_at_utc=finished_at_utc,
                    )
                    persist_trial_outputs(resolved_config, manifest, trial_record=completed_record)
                    pair_rows[pair_name].append(
                        _make_pair_row(
                            resolved_config=resolved_config,
                            pair_name=pair_name,
                            x_parameter=x_parameter,
                            x_index=x_index,
                            x_value=x_value,
                            y_parameter=y_parameter,
                            y_index=y_index,
                            y_value=y_value,
                            grid_index=grid_index,
                            build_spec=build_spec,
                            trial_id=trial_id,
                            status="completed",
                            seed_bundle=seed_bundle,
                            nominal_values=nominal_values,
                            metrics=metrics,
                            runtime_seconds=runtime_seconds,
                            started_at_utc=started_at_utc,
                            finished_at_utc=finished_at_utc,
                        )
                    )

        for pair_name, rows in pair_rows.items():
            artifact_paths[pair_name] = _write_csv(
                parameter_pairs_dir / f"{pair_name}.csv",
                rows,
            )

        summary = finalize_run(
            resolved_config,
            manifest,
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            status="completed",
        )
        pair_manifest = {
            "study_id": resolved_config["study"]["study_id"],
            "run_id": resolved_config["run"]["run_id"],
            "source_config_path": source_config_path,
            "baseline_nominal_values": baseline_nominal_values,
            "parameter_specs": parameter_specs,
            "pair_order": pair_order,
            "pair_definitions": pair_definitions,
            "families": list(resolved_config["comparison"]["compared_families"]),
            "parameter_pair_artifacts": artifact_paths,
            "canonical_artifacts": summary["artifacts"],
            "started_at_utc": manifest["lifecycle"]["started_at_utc"],
            "finished_at_utc": manifest["lifecycle"]["finished_at_utc"],
        }
        manifest_path = _write_json(run_root / "pair_sweep_manifest.json", pair_manifest)
        summary_path = _write_json(run_root / "parameter_pair_sweep_summary.json", pair_manifest)
        if notify_on_completion:
            send_run_completion_notification(
                study_id=str(resolved_config["study"]["study_id"]),
                run_id=str(summary["run_id"]),
                status="completed",
                trial_counts=summary.get("trial_counts"),
                extra_message=f"swept_pairs={len(pair_order)}",
            )
        return {
            "study_id": resolved_config["study"]["study_id"],
            "run_id": resolved_config["run"]["run_id"],
            "parameter_pair_artifacts": artifact_paths,
            "pair_sweep_manifest_path": manifest_path,
            "parameter_pair_sweep_summary_path": summary_path,
            "canonical_summary": summary,
        }
    except Exception as exc:
        summary = finalize_run(
            resolved_config,
            manifest,
            git_commit=git_commit,
            git_branch=git_branch,
            python_version=python_version,
            hostname=hostname,
            status="failed",
        )
        if notify_on_completion:
            send_run_completion_notification(
                study_id=str(resolved_config["study"]["study_id"]),
                run_id=str(summary["run_id"]),
                status="failed",
                trial_counts=summary.get("trial_counts"),
                extra_message=f"{type(exc).__name__}: {exc}",
            )
        raise
