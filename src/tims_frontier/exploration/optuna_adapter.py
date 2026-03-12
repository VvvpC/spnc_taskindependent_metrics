from __future__ import annotations

import optuna
from typing import Any, Callable, Mapping

from tims_frontier.construction.builders import build_trial_spec
from tims_frontier.evaluation.tims import evaluate_trial_tims
from tims_frontier.orchestration.seeds import derive_seed_bundle


def create_optuna_study_kwargs(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return create-study kwargs derived from resolved config."""

    optuna_cfg = resolved_config["exploration"]["optuna"]
    return {
        "directions": [optuna_cfg["objective_directions"]["MC"], optuna_cfg["objective_directions"]["CQ"]],
        "sampler_name": optuna_cfg["sampler"],
        "sampler_seed": optuna_cfg["sampler_seed"],
        "study_direction_mode": optuna_cfg["study_direction_mode"],
    }


def create_optuna_sampler(resolved_config: Mapping[str, Any]) -> optuna.samplers.BaseSampler:
    """Instantiate the configured Optuna sampler for the study."""

    kwargs = create_optuna_study_kwargs(resolved_config)
    sampler_name = kwargs["sampler_name"]
    sampler_seed = int(kwargs["sampler_seed"])
    if sampler_name == "NSGAIISampler":
        return optuna.samplers.NSGAIISampler(seed=sampler_seed)
    raise ValueError(f"Unsupported Optuna sampler: {sampler_name}")


def create_family_study(
    resolved_config: Mapping[str, Any],
    *,
    family: str,
    study_name: str | None = None,
) -> optuna.study.Study:
    """Create an in-memory Optuna study for one reservoir family."""

    kwargs = create_optuna_study_kwargs(resolved_config)
    return optuna.create_study(
        study_name=study_name or f"{family}_mc_cq_study",
        directions=list(kwargs["directions"]),
        sampler=create_optuna_sampler(resolved_config),
    )


def suggest_search_params(trial: Any, resolved_config: Mapping[str, Any]) -> dict[str, float]:
    search_space = resolved_config["exploration"]["search_space"]
    sampled: dict[str, float] = {}
    for name in ("beta_prime", "theta", "gamma", "m0"):
        bounds = search_space[name]
        sampled[name] = float(trial.suggest_float(name, float(bounds["low"]), float(bounds["high"])))
    return sampled


def create_family_objective(
    resolved_config: Mapping[str, Any],
    *,
    family: str,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[Any], tuple[float, float]]:
    """Build an Optuna objective function for one family."""

    global_seed = int(resolved_config["execution"]["seed_policy"]["global_seed"])

    def objective(trial: Any) -> tuple[float, float]:
        family_trial_index = int(getattr(trial, "number", 0)) + 1
        sampled = suggest_search_params(trial, resolved_config)
        seeds = derive_seed_bundle(global_seed=global_seed, family=family, family_trial_index=family_trial_index)
        build_spec = build_trial_spec(
            resolved_config,
            family=family,
            sampled_params=sampled,
            seed_bundle=seeds,
        )
        metrics = evaluate_trial_tims(build_spec, resolved_config, seed_bundle=seeds)
        if record_callback is not None:
            record_callback(
                {
                    "family": family,
                    "family_trial_index": family_trial_index,
                    "sampled_params": sampled,
                    "seed_bundle": seeds,
                    "metrics": metrics,
                }
            )
        return float(metrics["MC"]), float(metrics["CQ"])

    return objective
