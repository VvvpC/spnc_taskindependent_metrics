from __future__ import annotations

import random
from typing import Any

from .models import CompiledFamilySpec, ParameterSpec, SampledInstance


def _lhs_values(n_samples: int, rng: random.Random) -> list[float]:
    base = [(index + rng.random()) / float(n_samples) for index in range(n_samples)]
    rng.shuffle(base)
    return base


def _sample_parameter_sequence(
    spec: ParameterSpec,
    *,
    n_samples: int,
    sampler_name: str,
    rng: random.Random,
) -> list[Any]:
    if spec.kind == "fixed":
        return [spec.value for _ in range(n_samples)]
    if spec.kind == "categorical":
        return [spec.choices[int(rng.randrange(len(spec.choices)))] for _ in range(n_samples)]

    if sampler_name == "latin_hypercube":
        unit = _lhs_values(n_samples, rng)
    else:
        unit = [rng.random() for _ in range(n_samples)]

    low = float(spec.low)
    high = float(spec.high)
    values = [low + (high - low) * float(item) for item in unit]
    if spec.kind == "uniform_int":
        return [int(round(item)) for item in values]
    return [float(item) for item in values]


def _distribution_offsets(
    *,
    form: str,
    count: int,
    center: float,
    spread: float,
    rng: random.Random,
) -> list[float]:
    if count <= 0:
        raise ValueError("Subgroup count must be positive")
    if count == 1:
        return [float(center)]
    if form == "gradient":
        if count == 1:
            return [float(center)]
        step = (2.0 * spread) / float(count - 1)
        return [float(center - spread + step * index) for index in range(count)]
    if form == "random":
        return [float(rng.uniform(center - spread, center + spread)) for _ in range(count)]
    if form == "normaldistribution":
        sigma = spread if spread > 0 else 1.0e-9
        return [float(rng.gauss(center, sigma)) for _ in range(count)]
    raise ValueError(f"Unsupported subgroup distribution form: {form}")


def _normalize(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [float(item / total) for item in weights]


def _locked_ratio_weights(
    subgroup_details: list[dict[str, Any]],
    *,
    mix_ratio: float | None,
) -> list[float]:
    explicit = [detail.get("weight_fraction") for detail in subgroup_details]
    if any(value is not None for value in explicit):
        totals = [float(value or 0.0) for value in explicit]
        normalized_totals = _normalize(totals)
    elif mix_ratio is not None and len(subgroup_details) == 2:
        normalized_totals = [float(1.0 - mix_ratio), float(mix_ratio)]
    else:
        normalized_totals = _normalize([float(detail["count"]) for detail in subgroup_details])

    weights: list[float] = []
    for subgroup_total, subgroup in zip(normalized_totals, subgroup_details):
        count = int(subgroup["count"])
        weights.extend([float(subgroup_total / count)] * count)
    return _normalize(weights)


def _coupled_weights(
    beta_offsets: list[float],
    subgroup_details: list[dict[str, Any]],
    *,
    rule: str,
    strength: float | None,
    mix_ratio: float | None,
) -> list[float]:
    total_count = len(beta_offsets)
    base = [1.0 / total_count] * total_count
    if rule == "independent":
        return base
    if rule == "locked_ratio":
        return _locked_ratio_weights(subgroup_details, mix_ratio=mix_ratio)

    offsets = [float(item) for item in beta_offsets]
    minimum = min(offsets)
    maximum = max(offsets)
    if maximum - minimum <= 1.0e-12:
        correlated = list(base)
    else:
        if rule == "positive_correlation":
            correlated = [item - minimum for item in offsets]
        elif rule == "negative_correlation":
            correlated = [maximum - item for item in offsets]
        else:
            raise ValueError(f"Unsupported coupling rule: {rule}")
        correlated = [item + 1.0e-6 for item in correlated]
        correlated = _normalize(correlated)
    blend = float(strength or 0.5)
    weights = [
        (1.0 - blend) * float(base_weight) + blend * float(correlated_weight)
        for base_weight, correlated_weight in zip(base, correlated)
    ]
    return _normalize(weights)


def sample_family(compiled: CompiledFamilySpec) -> list[SampledInstance]:
    """Sample concrete reservoir instances from a compiled family proposal."""

    plan = compiled.sampling_plan
    rng = random.Random(plan.seed)
    family = compiled.family_definition
    n_samples = int(plan.n_samples)
    continuous_sequences = {
        key: _sample_parameter_sequence(spec, n_samples=n_samples, sampler_name=plan.sampler_name, rng=rng)
        for key, spec in family.continuous_parameters.items()
    }
    subgroup_sequences: dict[str, dict[str, list[Any]]] = {}
    for subgroup in family.subgroups:
        subgroup_sequences[subgroup.name] = {
            "count": _sample_parameter_sequence(subgroup.count, n_samples=n_samples, sampler_name=plan.sampler_name, rng=rng),
            "offset_center": _sample_parameter_sequence(
                subgroup.offset_center,
                n_samples=n_samples,
                sampler_name=plan.sampler_name,
                rng=rng,
            ),
            "spread": _sample_parameter_sequence(subgroup.spread, n_samples=n_samples, sampler_name=plan.sampler_name, rng=rng),
        }
        if subgroup.weight_fraction is not None:
            subgroup_sequences[subgroup.name]["weight_fraction"] = _sample_parameter_sequence(
                subgroup.weight_fraction,
                n_samples=n_samples,
                sampler_name=plan.sampler_name,
                rng=rng,
            )

    coupling = family.coupling_rule
    strength_sequence = (
        _sample_parameter_sequence(coupling.strength, n_samples=n_samples, sampler_name=plan.sampler_name, rng=rng)
        if coupling.strength is not None
        else [None] * n_samples
    )
    mix_ratio_sequence = (
        _sample_parameter_sequence(coupling.mix_ratio, n_samples=n_samples, sampler_name=plan.sampler_name, rng=rng)
        if coupling.mix_ratio is not None
        else [None] * n_samples
    )

    sampled_instances: list[SampledInstance] = []
    for sample_index in range(n_samples):
        sample_rng = random.Random(plan.seed + (sample_index + 1) * 9973)
        sampled_parameters = {key: values[sample_index] for key, values in continuous_sequences.items()}
        subgroup_details: list[dict[str, Any]] = []
        beta_offsets: list[float] = []
        for subgroup in family.subgroups:
            subgroup_state = subgroup_sequences[subgroup.name]
            count = int(subgroup_state["count"][sample_index])
            center = float(subgroup_state["offset_center"][sample_index])
            spread = max(0.0, float(subgroup_state["spread"][sample_index]))
            distribution_form = subgroup.distribution_form or family.distribution_rule.form
            offsets = _distribution_offsets(
                form=distribution_form,
                count=count,
                center=center,
                spread=spread,
                rng=sample_rng,
            )
            detail = {
                "name": subgroup.name,
                "role": subgroup.role,
                "count": count,
                "offset_center": center,
                "spread": spread,
                "distribution_form": distribution_form,
                "offsets": list(offsets),
            }
            if subgroup.weight_fraction is not None:
                detail["weight_fraction"] = float(subgroup_state["weight_fraction"][sample_index])
            subgroup_details.append(detail)
            beta_offsets.extend(offsets)

        weights = _coupled_weights(
            beta_offsets,
            subgroup_details,
            rule=family.coupling_rule.rule,
            strength=(
                float(strength_sequence[sample_index]) if strength_sequence[sample_index] is not None else None
            ),
            mix_ratio=float(mix_ratio_sequence[sample_index]) if mix_ratio_sequence[sample_index] is not None else None,
        )
        sampled_instances.append(
            SampledInstance(
                sample_index=sample_index + 1,
                sampled_parameters=sampled_parameters,
                subgroup_details=subgroup_details,
                beta_offsets=[float(item) for item in beta_offsets],
                weights=weights,
            )
        )
    return sampled_instances
