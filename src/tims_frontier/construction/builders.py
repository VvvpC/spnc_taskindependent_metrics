from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ReservoirParameterSet:
    h: float
    theta_H: float
    k_s_0: float
    phi: float
    beta_prime: float
    Nvirt: int
    m0: float
    bias: bool
    Nwarmup: int
    params: dict[str, Any]


@dataclass(frozen=True)
class MorphologySpec:
    geometry_mode: str
    scheme: str | None = None
    reference_beta_param: str | None = None
    n_instances: int | None = None
    beta_spread: float | None = None
    beta_sampling_rule: str | None = None
    clip_beta_to_positive: bool = True
    weights_mode: str | None = None
    morphology_seed: int | None = None


@dataclass(frozen=True)
class TrialBuildSpec:
    family: str
    family_role: str
    geometry_mode: str
    reservoir_params: ReservoirParameterSet
    morphology: MorphologySpec
    weights: list[float] = field(default_factory=list)


def build_trial_spec(
    resolved_config: Mapping[str, Any],
    *,
    family: str,
    sampled_params: Mapping[str, float],
    seed_bundle: Mapping[str, int],
) -> TrialBuildSpec:
    """Build a workflow-native trial specification from resolved config."""

    shared = resolved_config["reservoir"]["shared_defaults"]
    family_cfg = resolved_config["families"][family]
    family_role = "baseline" if family == resolved_config["comparison"]["baseline_family"] else "candidate"
    nvirt = int(resolved_config["exploration"]["search_space"]["Nvirt"]["value"])

    reservoir_params = ReservoirParameterSet(
        h=float(shared["physical"]["h"]),
        theta_H=float(shared["physical"]["theta_H"]),
        k_s_0=float(shared["physical"]["k_s_0"]),
        phi=float(shared["physical"]["phi"]),
        beta_prime=float(sampled_params["beta_prime"]),
        Nvirt=nvirt,
        m0=float(sampled_params["m0"]),
        bias=bool(shared["network"]["bias"]),
        Nwarmup=int(shared["network"]["Nwarmup"]),
        params={
            "theta": float(sampled_params["theta"]),
            "gamma": float(sampled_params["gamma"]),
            "delay_feedback": shared["simulation"]["delay_feedback"],
            "Nvirt": nvirt,
            "length_warmup": int(shared["network"]["Nwarmup"]),
            "warmup_sample": int(shared["network"]["Nwarmup"]) * nvirt,
            "voltage_noise": bool(shared["simulation"]["voltage_noise"]),
            "johnson_noise": bool(shared["simulation"]["johnson_noise"]),
            "thermal_noise": bool(shared["simulation"]["thermal_noise"]),
        },
    )

    resolved_construction = family_cfg["resolved_construction"]
    if family == "uniform":
        morphology = MorphologySpec(geometry_mode=resolved_construction["geometry_mode"])
        weights: list[float] = []
    else:
        morph = resolved_construction["morphology"]
        n_instances = int(morph["n_instances"])
        weights_mode = morph["weights_mode"]
        weights = [1.0 / n_instances] * n_instances if weights_mode == "equal" else []
        morphology = MorphologySpec(
            geometry_mode=resolved_construction["geometry_mode"],
            scheme=morph["scheme"],
            reference_beta_param=morph["reference_beta_param"],
            n_instances=n_instances,
            beta_spread=float(morph["beta_spread"]),
            beta_sampling_rule=morph["beta_sampling_rule"],
            clip_beta_to_positive=bool(morph["clip_beta_to_positive"]),
            weights_mode=weights_mode,
            morphology_seed=seed_bundle.get("morphology_seed"),
        )

    return TrialBuildSpec(
        family=family,
        family_role=family_role,
        geometry_mode=resolved_construction["geometry_mode"],
        reservoir_params=reservoir_params,
        morphology=morphology,
        weights=weights,
    )
