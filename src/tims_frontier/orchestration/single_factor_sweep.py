from __future__ import annotations

import csv
from datetime import datetime, timezone
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


SWEEP_RESULT_FIELDNAMES = [
    "study_id",
    "run_id",
    "sweep_parameter",
    "sweep_index",
    "sweep_value",
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
    raise ValueError(f"Unsupported single-factor sweep kind: {kind}")


def _resolve_single_factor_config(study_config: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    sweep_cfg = study_config.get("single_factor_sweep")
    if not isinstance(sweep_cfg, Mapping):
        raise ValueError("Study config requires a top-level 'single_factor_sweep' mapping.")

    parameters = sweep_cfg.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ValueError("'single_factor_sweep.parameters' must be a non-empty mapping.")

    parameter_order = sweep_cfg.get("parameter_order")
    if parameter_order is None:
        ordered_names = [str(name) for name in parameters.keys()]
    else:
        if not isinstance(parameter_order, Sequence) or isinstance(parameter_order, (str, bytes)):
            raise ValueError("'single_factor_sweep.parameter_order' must be a list of parameter names.")
        ordered_names = [str(name) for name in parameter_order]
        if set(ordered_names) != set(parameters.keys()):
            raise ValueError("'single_factor_sweep.parameter_order' must match the parameter keys exactly.")

    resolved: dict[str, dict[str, Any]] = {}
    for name in ordered_names:
        entry = parameters[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Sweep parameter '{name}' must map to a config object.")
        resolved[name] = {
            "kind": str(entry["kind"]),
            "low": float(entry["low"]),
            "high": float(entry["high"]),
            "num_points": int(entry["num_points"]),
        }
        resolved[name]["values"] = _generate_parameter_values(**resolved[name])
    return ordered_names, resolved


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_RESULT_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in SWEEP_RESULT_FIELDNAMES})
    return path.as_posix()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path.as_posix()


def _sampled_operational_params(nominal_values: Mapping[str, float | int]) -> dict[str, float]:
    return {
        "beta_prime": float(nominal_values["beta_prime"]),
        "theta": float(nominal_values["theta"]),
        "gamma": float(nominal_values["gamma"]),
        "m0": float(nominal_values["m0"]),
    }


def _heterogeneous_morphology_overrides(
    sweep_parameter: str,
    nominal_values: Mapping[str, float | int],
) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    if sweep_parameter == "n_instances":
        overrides["n_instances"] = int(nominal_values["n_instances"])
    if sweep_parameter == "beta_spread":
        overrides["beta_spread"] = float(nominal_values["beta_spread"])
    return overrides or None


def _make_trial_id(sweep_parameter: str, sweep_index: int, family: str) -> str:
    return f"{sweep_parameter}_sweep_{sweep_index:02d}_{family}"


def _make_sweep_row(
    *,
    resolved_config: Mapping[str, Any],
    sweep_parameter: str,
    sweep_index: int,
    sweep_value: float | int,
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
        "sweep_parameter": sweep_parameter,
        "sweep_index": sweep_index,
        "sweep_value": sweep_value,
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


def run_single_factor_sweep_study(
    study_config: Mapping[str, Any],
    *,
    source_config_path: str,
    run_id: str | None = None,
    git_commit: str | None = None,
    git_branch: str | None = None,
    python_version: str | None = None,
    hostname: str | None = None,
) -> dict[str, Any]:
    """Run a paired uniform-vs-heterogeneous single-factor TIMs sweep."""

    resolved_config, manifest = initialize_run(
        study_config,
        source_config_path=source_config_path,
        run_id=run_id,
    )
    baseline_nominal_values = _build_baseline_nominal_values(resolved_config)
    parameter_order, parameter_specs = _resolve_single_factor_config(study_config)

    global_seed = int(resolved_config["execution"]["seed_policy"]["global_seed"])
    matched_trial_seed_across_families = bool(
        resolved_config["execution"]["seed_policy"]["matched_trial_seed_across_families"]
    )
    logs_dir = str(resolved_config["storage"]["paths"]["logs_dir"])
    run_root = Path(resolved_config["storage"]["paths"]["run_root"])
    parameter_sweeps_dir = run_root / "parameter_sweeps"
    parameter_sweeps_dir.mkdir(parents=True, exist_ok=True)

    parameter_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in parameter_order}
    artifact_paths: dict[str, str] = {}
    pair_index = 0

    try:
        for sweep_parameter in parameter_order:
            values = parameter_specs[sweep_parameter]["values"]
            for sweep_index, sweep_value in enumerate(values, start=1):
                pair_index += 1
                nominal_values = dict(baseline_nominal_values)
                nominal_values[sweep_parameter] = sweep_value
                for family in resolved_config["comparison"]["compared_families"]:
                    family = str(family)
                    trial_id = _make_trial_id(sweep_parameter, sweep_index, family)
                    started_at_utc = _utc_now_iso()
                    started_perf = time.perf_counter()
                    seed_bundle = derive_seed_bundle(
                        global_seed=global_seed,
                        family=family,
                        family_trial_index=pair_index,
                        matched_across_families=matched_trial_seed_across_families,
                    )
                    sampled_params = _sampled_operational_params(nominal_values)
                    build_spec = build_trial_spec(
                        resolved_config,
                        family=family,
                        sampled_params=sampled_params,
                        seed_bundle=seed_bundle,
                        morphology_overrides=(
                            _heterogeneous_morphology_overrides(sweep_parameter, nominal_values)
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
                            family_trial_index=pair_index,
                            optuna_study_name=f"{sweep_parameter}_single_factor_sweep",
                            optuna_trial_number=pair_index - 1,
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
                            optuna_trial_number=pair_index - 1,
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
                        parameter_rows[sweep_parameter].append(
                            _make_sweep_row(
                                resolved_config=resolved_config,
                                sweep_parameter=sweep_parameter,
                                sweep_index=sweep_index,
                                sweep_value=sweep_value,
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
                        family_trial_index=pair_index,
                        optuna_study_name=f"{sweep_parameter}_single_factor_sweep",
                        optuna_trial_number=pair_index - 1,
                        status="completed",
                        seed_bundle=seed_bundle,
                        sampled_params=sampled_params,
                        metrics=metrics,
                        runtime_seconds=runtime_seconds,
                        started_at_utc=started_at_utc,
                        finished_at_utc=finished_at_utc,
                    )
                    persist_trial_outputs(resolved_config, manifest, trial_record=completed_record)
                    parameter_rows[sweep_parameter].append(
                        _make_sweep_row(
                            resolved_config=resolved_config,
                            sweep_parameter=sweep_parameter,
                            sweep_index=sweep_index,
                            sweep_value=sweep_value,
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

        for sweep_parameter, rows in parameter_rows.items():
            artifact_paths[sweep_parameter] = _write_csv(
                parameter_sweeps_dir / f"{sweep_parameter}_sweep.csv",
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
        sweep_manifest = {
            "study_id": resolved_config["study"]["study_id"],
            "run_id": resolved_config["run"]["run_id"],
            "source_config_path": source_config_path,
            "baseline_nominal_values": baseline_nominal_values,
            "parameter_order": parameter_order,
            "parameter_specs": parameter_specs,
            "families": list(resolved_config["comparison"]["compared_families"]),
            "parameter_sweep_artifacts": artifact_paths,
            "canonical_artifacts": summary["artifacts"],
            "started_at_utc": manifest["lifecycle"]["started_at_utc"],
            "finished_at_utc": manifest["lifecycle"]["finished_at_utc"],
        }
        manifest_path = _write_json(run_root / "sweep_manifest.json", sweep_manifest)
        summary_path = _write_json(run_root / "single_factor_sweep_summary.json", sweep_manifest)
        send_run_completion_notification(
            study_id=str(resolved_config["study"]["study_id"]),
            run_id=str(summary["run_id"]),
            status="completed",
            trial_counts=summary.get("trial_counts"),
            extra_message=f"swept_parameters={len(parameter_order)}",
        )
        return {
            "study_id": resolved_config["study"]["study_id"],
            "run_id": resolved_config["run"]["run_id"],
            "parameter_sweep_artifacts": artifact_paths,
            "sweep_manifest_path": manifest_path,
            "single_factor_sweep_summary_path": summary_path,
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
        send_run_completion_notification(
            study_id=str(resolved_config["study"]["study_id"]),
            run_id=str(summary["run_id"]),
            status="failed",
            trial_counts=summary.get("trial_counts"),
            extra_message=f"{type(exc).__name__}: {exc}",
        )
        raise
