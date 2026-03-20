from __future__ import annotations

import statistics
from typing import Any, Mapping

from tims_frontier.construction.builders import ReservoirParameterSet

from .common import bootstrap_legacy_source_paths, derive_seed_bundle
from .models import SampledInstance


def _build_reservoir_params(sampled_parameters: Mapping[str, Any], runtime_config: Mapping[str, Any]) -> ReservoirParameterSet:
    shared = runtime_config["evaluator"]["shared_defaults"]
    nvirt = int(runtime_config["evaluator"]["nvirt"])
    nwarmup = int(shared["network"]["Nwarmup"])
    return ReservoirParameterSet(
        h=float(shared["physical"]["h"]),
        theta_H=float(shared["physical"]["theta_H"]),
        k_s_0=float(shared["physical"]["k_s_0"]),
        phi=float(shared["physical"]["phi"]),
        beta_prime=float(sampled_parameters["beta_prime"]),
        Nvirt=nvirt,
        m0=float(sampled_parameters["m0"]),
        bias=bool(shared["network"]["bias"]),
        Nwarmup=nwarmup,
        params={
            "theta": float(sampled_parameters["theta"]),
            "gamma": float(sampled_parameters["gamma"]),
            "delay_feedback": shared["simulation"]["delay_feedback"],
            "Nvirt": nvirt,
            "length_warmup": nwarmup,
            "warmup_sample": nwarmup * nvirt,
            "voltage_noise": bool(shared["simulation"]["voltage_noise"]),
            "johnson_noise": bool(shared["simulation"]["johnson_noise"]),
            "thermal_noise": bool(shared["simulation"]["thermal_noise"]),
        },
    )


def _evaluate_mock_instance(instance: SampledInstance) -> dict[str, float]:
    sampled = instance.sampled_parameters
    offsets = [float(item) for item in instance.beta_offsets]
    weights = [float(item) for item in instance.weights]
    weighted_offset = sum(offset * weight for offset, weight in zip(offsets, weights))
    dispersion = statistics.pstdev(offsets) if len(offsets) > 1 else 0.0
    mc = (
        1.25
        + 0.10 * float(sampled["beta_prime"])
        - 3.0 * abs(float(sampled["gamma"]) - 0.055)
        - 4.0 * abs(float(sampled["theta"]) - 0.21)
        + 8.0 * float(sampled["m0"])
        + 0.20 * weighted_offset
    )
    kr = 60.0 + 0.8 * float(sampled["beta_prime"]) + 2.5 * weighted_offset + 4.0 * dispersion
    gr = max(1.0, 6.0 - 2.0 * dispersion + abs(weighted_offset))
    cq = kr - gr
    return {"MC": float(mc), "KR": float(kr), "GR": float(gr), "CQ": float(cq), "CQ_formula": "KR_minus_GR"}


def _evaluate_real_instance(instance: SampledInstance, runtime_config: Mapping[str, Any], *, seed_bundle: Mapping[str, int]) -> dict[str, float | str]:
    bootstrap_legacy_source_paths()
    import numpy as np

    from Reservoirs_morphology_evaluation import RunSpnc_heterogenous
    from formal_Parameter_Dynamics_Preformance import Evaluate_KR_GR, gen_KR_GR_input, generate_signal, linear_MC

    reservoir_params = _build_reservoir_params(instance.sampled_parameters, runtime_config)
    tims_cfg = runtime_config["evaluator"]["tims"]

    temp_params = {"beta_prime": reservoir_params.beta_prime, "beta_ref": reservoir_params.beta_prime}
    res_params = {
        "m0": reservoir_params.m0,
        "h": reservoir_params.h,
        "deltabeta_list": list(instance.beta_offsets),
    }

    signal = generate_signal(int(tims_cfg["mc"]["signal_len"]), seed=int(seed_bundle["input_signal_seed"]))
    mc_output = RunSpnc_heterogenous(
        signal,
        1,
        reservoir_params.Nvirt,
        1,
        temp_params,
        res_params,
        reservoir_params.params,
        *instance.weights,
        fixed_mask=True,
        seed_mask=int(seed_bundle["mask_seed"]),
    )
    mc = linear_MC(
        signal,
        mc_output,
        splits=list(tims_cfg["mc"]["splits"]),
        delays=int(tims_cfg["mc"]["delays"]),
    )

    inputs = gen_KR_GR_input(
        reservoir_params.Nvirt,
        int(tims_cfg["kr_gr"]["n_wash"]),
        seed=int(seed_bundle["input_signal_seed"]),
    )
    outputs = []
    for input_row in inputs:
        output = RunSpnc_heterogenous(
            input_row.reshape(-1, 1),
            1,
            reservoir_params.Nvirt,
            1,
            temp_params,
            res_params,
            reservoir_params.params,
            *instance.weights,
            fixed_mask=True,
            seed_mask=int(seed_bundle["mask_seed"]),
        )
        outputs.append(output)
    states = np.stack(outputs, axis=0)
    max_value = float(np.max(np.abs(states)))
    if max_value > 0:
        states = states / max_value
    kr, gr = Evaluate_KR_GR(states, reservoir_params.Nvirt, threshold=float(tims_cfg["kr_gr"]["threshold"]))
    cq = float(kr - gr)
    return {"MC": float(mc), "KR": float(kr), "GR": float(gr), "CQ": cq, "CQ_formula": "KR_minus_GR"}


def evaluate_sampled_instances(
    instances: list[SampledInstance],
    runtime_config: Mapping[str, Any],
    *,
    family_seed: int,
) -> list[dict[str, Any]]:
    """Evaluate sampled instances through the fixed TIMs pipeline."""

    results: list[dict[str, Any]] = []
    for instance in instances:
        seed_bundle = derive_seed_bundle(
            global_seed=family_seed,
            family="autoresearch",
            family_trial_index=int(instance.sample_index),
            matched_across_families=False,
        )
        if runtime_config["evaluator"]["mode"] == "mock":
            metrics = _evaluate_mock_instance(instance)
        else:
            metrics = _evaluate_real_instance(instance, runtime_config, seed_bundle=seed_bundle)
        results.append(
            {
                "sample_index": instance.sample_index,
                "sampled_parameters": dict(instance.sampled_parameters),
                "subgroup_details": [dict(item) for item in instance.subgroup_details],
                "beta_offsets": list(instance.beta_offsets),
                "weights": list(instance.weights),
                "seed_bundle": dict(seed_bundle),
                **metrics,
            }
        )
    return results
