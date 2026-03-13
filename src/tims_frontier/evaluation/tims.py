from __future__ import annotations

from typing import Any, Mapping

from tims_frontier.construction.builders import MorphologySpec, ReservoirParameterSet, TrialBuildSpec


def _legacy_reservoir_params(params: ReservoirParameterSet) -> Any:
    from formal_Parameter_Dynamics_Preformance import ReservoirParams as LegacyReservoirParams

    return LegacyReservoirParams(
        h=params.h,
        theta_H=params.theta_H,
        k_s_0=params.k_s_0,
        phi=params.phi,
        beta_prime=params.beta_prime,
        Nvirt=params.Nvirt,
        m0=params.m0,
        bias=params.bias,
        Nwarmup=params.Nwarmup,
        params=params.params,
    )


def _legacy_morphology_config(params: ReservoirParameterSet, morphology: MorphologySpec) -> Any:
    from Reservoirs_morphology_creator import MorphologyConfig

    if morphology.geometry_mode == "uniform":
        return MorphologyConfig(morph_type="uniform")

    spread = float(morphology.beta_spread or 0.0)
    beta_low = params.beta_prime - spread
    beta_high = params.beta_prime + spread
    if morphology.clip_beta_to_positive:
        beta_low = max(beta_low, 1e-9)
        beta_high = max(beta_high, beta_low)

    return MorphologyConfig(
        morph_type=str(morphology.scheme),
        beta_range=(beta_low, beta_high),
        distribution_type=str(morphology.scheme),
        random_seed=morphology.morphology_seed,
        n_instances=morphology.n_instances,
    )


def evaluate_trial_tims(
    build_spec: TrialBuildSpec,
    resolved_config: Mapping[str, Any],
    *,
    seed_bundle: Mapping[str, int],
) -> dict[str, float | str]:
    """Evaluate TIMs for a single trial spec through legacy adapters."""

    from Reservoirs_morphology_evaluation import (
        evaluate_heterogeneous_KRandGR,
        evaluate_heterogeneous_MC,
    )

    tims_cfg = resolved_config["evaluation"]["tims"]
    legacy_params = _legacy_reservoir_params(build_spec.reservoir_params)
    legacy_morphology = _legacy_morphology_config(build_spec.reservoir_params, build_spec.morphology)

    mc_cfg = tims_cfg["mc"]
    kr_gr_cfg = tims_cfg["kr_gr"]

    mc_result = evaluate_heterogeneous_MC(
        legacy_params,
        legacy_morphology,
        build_spec.weights,
        signal_len=int(mc_cfg["signal_len"]),
        seed=int(seed_bundle["input_signal_seed"]),
        mask_seed=int(seed_bundle["mask_seed"]),
        splits=list(mc_cfg["splits"]),
        delays=int(mc_cfg["delays"]),
    )
    kr_gr_result = evaluate_heterogeneous_KRandGR(
        legacy_params,
        legacy_morphology,
        build_spec.weights,
        Nreadouts=int(kr_gr_cfg["n_readouts_value"]),
        Nwash=int(kr_gr_cfg["n_wash"]),
        threshold=float(kr_gr_cfg["threshold"]),
        seed=int(seed_bundle["input_signal_seed"]),
        mask_seed=int(seed_bundle["mask_seed"]),
    )

    kr = float(kr_gr_result["KR"])
    gr = float(kr_gr_result["GR"])
    cq = kr - gr
    return {
        "MC": float(mc_result["MC"]),
        "KR": kr,
        "GR": gr,
        "CQ": cq,
        "CQ_formula": tims_cfg["cq"]["formula"],
    }
