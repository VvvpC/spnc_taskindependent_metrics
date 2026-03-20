from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .common import unique_preserve_order
from .models import (
    CompiledFamilySpec,
    CouplingRule,
    DistributionRule,
    FamilyDefinition,
    ParameterSpec,
    Proposal,
    SubgroupDefinition,
)


DEFAULT_BASE_PARAMETERS = {
    "beta_prime": ParameterSpec(kind="uniform_float", low=28.0, high=32.0),
    "theta": ParameterSpec(kind="uniform_float", low=0.15, high=0.25),
    "gamma": ParameterSpec(kind="uniform_float", low=0.04, high=0.08),
    "m0": ParameterSpec(kind="uniform_float", low=0.008, high=0.015),
}


def _default_single_subgroup() -> SubgroupDefinition:
    return SubgroupDefinition(
        name="core",
        role="core",
        count=ParameterSpec(kind="uniform_int", low=3, high=5),
        offset_center=ParameterSpec.fixed(0.0),
        spread=ParameterSpec(kind="uniform_float", low=2.0, high=4.0),
        distribution_form=None,
    )


def _default_tail_subgroup() -> SubgroupDefinition:
    return SubgroupDefinition(
        name="tail",
        role="tail",
        count=ParameterSpec(kind="uniform_int", low=1, high=2),
        offset_center=ParameterSpec(kind="uniform_float", low=1.5, high=3.0),
        spread=ParameterSpec(kind="uniform_float", low=0.25, high=1.0),
        distribution_form=None,
        weight_fraction=ParameterSpec(kind="uniform_float", low=0.2, high=0.4),
    )


def _ensure_distribution(distribution_rule: DistributionRule, auto_fields: list[str]) -> DistributionRule:
    if distribution_rule.form:
        return distribution_rule
    auto_fields.append("family_definition.distribution_rule.form")
    return replace(distribution_rule, form="random")


def _ensure_coupling(coupling_rule: CouplingRule, *, family_type: str, auto_fields: list[str]) -> CouplingRule:
    rule = coupling_rule.rule or "independent"
    result = replace(coupling_rule, rule=rule)
    if rule in {"positive_correlation", "negative_correlation"} and result.strength is None:
        auto_fields.append("family_definition.coupling_rule.strength")
        result = replace(result, strength=ParameterSpec(kind="uniform_float", low=0.3, high=0.7))
    if rule == "locked_ratio" and result.mix_ratio is None:
        auto_fields.append("family_definition.coupling_rule.mix_ratio")
        default_ratio = 0.3 if family_type == "bimodal_core_tail" else 1.0
        result = replace(result, mix_ratio=ParameterSpec.fixed(default_ratio))
    return result


def _ensure_base_parameters(
    continuous_parameters: Mapping[str, ParameterSpec],
    auto_fields: list[str],
) -> dict[str, ParameterSpec]:
    compiled = dict(continuous_parameters)
    for name, spec in DEFAULT_BASE_PARAMETERS.items():
        if name in compiled:
            continue
        compiled[name] = spec
        auto_fields.append(f"family_definition.continuous_parameters.{name}")
    return compiled


def _normalize_subgroups(
    family_type: str,
    subgroups: list[SubgroupDefinition],
    auto_fields: list[str],
) -> list[SubgroupDefinition]:
    if family_type == "single_distribution":
        if not subgroups:
            auto_fields.append("family_definition.subgroups[0]")
            return [_default_single_subgroup()]
        return subgroups

    if not subgroups:
        auto_fields.extend(["family_definition.subgroups[0]", "family_definition.subgroups[1]"])
        return [_default_single_subgroup(), _default_tail_subgroup()]

    names = {subgroup.name for subgroup in subgroups}
    normalized = list(subgroups)
    if "core" not in names:
        auto_fields.append("family_definition.subgroups.core")
        normalized.insert(0, _default_single_subgroup())
    if "tail" not in names:
        auto_fields.append("family_definition.subgroups.tail")
        normalized.append(_default_tail_subgroup())
    normalized.sort(key=lambda subgroup: (0 if subgroup.name == "core" else 1, subgroup.name))
    return normalized


def compile_proposal(proposal: Proposal, runtime_config: Mapping[str, Any]) -> CompiledFamilySpec:
    """Normalize proposal structure and fill fields that arise from one semantic edit."""

    auto_fields = list(proposal.auto_completed_fields)
    family = proposal.family_definition
    topology_name = family.topology_name
    if not topology_name:
        topology_name = family.family_type
        auto_fields.append("family_definition.topology_name")

    normalized_subgroups = _normalize_subgroups(family.family_type, family.subgroups, auto_fields)
    normalized_distribution = _ensure_distribution(family.distribution_rule, auto_fields)
    normalized_coupling = _ensure_coupling(
        family.coupling_rule,
        family_type=family.family_type,
        auto_fields=auto_fields,
    )
    normalized_parameters = _ensure_base_parameters(family.continuous_parameters, auto_fields)

    compiled_family = replace(
        family,
        topology_name=topology_name,
        subgroups=normalized_subgroups,
        distribution_rule=normalized_distribution,
        coupling_rule=normalized_coupling,
        continuous_parameters=normalized_parameters,
    )
    compiled_proposal = replace(proposal, family_definition=compiled_family)
    return CompiledFamilySpec.from_proposal(
        compiled_proposal,
        auto_completed_fields=unique_preserve_order(auto_fields),
    )
