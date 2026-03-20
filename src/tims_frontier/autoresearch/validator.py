from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .compiler import compile_proposal
from .models import (
    CompiledFamilySpec,
    ParameterSpec,
    Proposal,
    SUPPORTED_COUPLINGS,
    SUPPORTED_DISTRIBUTIONS,
    SUPPORTED_EDIT_TYPES,
    SUPPORTED_FAMILY_TYPES,
    SUPPORTED_PARAMETER_KINDS,
    SUPPORTED_SAMPLERS,
)


class ProposalValidationError(ValueError):
    """Raised when a proposal is structurally invalid or semantically too large."""


def _validate_parameter_spec(name: str, spec: ParameterSpec, constraints: Mapping[str, Any]) -> None:
    if spec.kind not in SUPPORTED_PARAMETER_KINDS:
        raise ProposalValidationError(f"Unsupported parameter kind for {name}: {spec.kind}")
    if spec.kind == "fixed":
        if spec.value is None:
            raise ProposalValidationError(f"Fixed parameter {name} must provide a value")
    elif spec.kind in {"uniform_float", "uniform_int"}:
        if spec.low is None or spec.high is None:
            raise ProposalValidationError(f"Range parameter {name} must provide low/high")
        if float(spec.low) > float(spec.high):
            raise ProposalValidationError(f"Parameter {name} has low > high")
    elif spec.kind == "categorical" and not spec.choices:
        raise ProposalValidationError(f"Categorical parameter {name} must provide choices")

    if "min" in constraints or "max" in constraints:
        bound_cfg = constraints
    else:
        bound_cfg = constraints.get(name)
    if not isinstance(bound_cfg, Mapping):
        return
    minimum = bound_cfg.get("min")
    maximum = bound_cfg.get("max")
    values_to_check: list[float] = []
    if spec.kind == "fixed" and isinstance(spec.value, (int, float)):
        values_to_check.append(float(spec.value))
    if spec.kind in {"uniform_float", "uniform_int"}:
        values_to_check.extend([float(spec.low), float(spec.high)])
    for candidate in values_to_check:
        if minimum is not None and candidate < float(minimum):
            raise ProposalValidationError(f"Parameter {name} violates minimum {minimum}: {candidate}")
        if maximum is not None and candidate > float(maximum):
            raise ProposalValidationError(f"Parameter {name} violates maximum {maximum}: {candidate}")


def _structure_signature(compiled: CompiledFamilySpec) -> tuple[Any, ...]:
    family = compiled.family_definition
    return (
        family.family_type,
        family.topology_name,
        tuple((item.name, item.role) for item in family.subgroups),
    )


def _distribution_signature(compiled: CompiledFamilySpec) -> tuple[Any, ...]:
    family = compiled.family_definition
    return (
        family.distribution_rule.form,
        tuple((item.name, item.distribution_form) for item in family.subgroups),
    )


def _coupling_signature(compiled: CompiledFamilySpec) -> tuple[Any, ...]:
    family = compiled.family_definition
    coupling = family.coupling_rule
    return (
        coupling.rule,
        asdict(coupling.strength) if coupling.strength is not None else None,
        asdict(coupling.mix_ratio) if coupling.mix_ratio is not None else None,
    )


def _collect_scalar_state(compiled: CompiledFamilySpec) -> dict[str, Any]:
    family = compiled.family_definition
    state: dict[str, Any] = {}
    for key, spec in family.continuous_parameters.items():
        state[f"continuous_parameters.{key}"] = asdict(spec)
    for index, subgroup in enumerate(family.subgroups):
        prefix = f"subgroups[{index}]"
        state[f"{prefix}.count"] = asdict(subgroup.count)
        state[f"{prefix}.offset_center"] = asdict(subgroup.offset_center)
        state[f"{prefix}.spread"] = asdict(subgroup.spread)
        if subgroup.weight_fraction is not None:
            state[f"{prefix}.weight_fraction"] = asdict(subgroup.weight_fraction)
    coupling = family.coupling_rule
    if coupling.strength is not None:
        state["coupling_rule.strength"] = asdict(coupling.strength)
    if coupling.mix_ratio is not None:
        state["coupling_rule.mix_ratio"] = asdict(coupling.mix_ratio)
    return state


def _collect_categorical_state(compiled: CompiledFamilySpec) -> dict[str, Any]:
    family = compiled.family_definition
    state = {
        "family_type": family.family_type,
        "topology_name": family.topology_name,
        "distribution_rule.form": family.distribution_rule.form,
        "coupling_rule.rule": family.coupling_rule.rule,
    }
    for index, subgroup in enumerate(family.subgroups):
        state[f"subgroups[{index}].distribution_form"] = subgroup.distribution_form
    return state


def _state_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    changed: dict[str, tuple[Any, Any]] = {}
    keys = set(before) | set(after)
    for key in keys:
        if before.get(key) == after.get(key):
            continue
        changed[key] = (before.get(key), after.get(key))
    return changed


def _classify_semantic_diff(parent: CompiledFamilySpec, child: CompiledFamilySpec) -> tuple[set[str], dict[str, Any]]:
    allowed_types: set[str] = set()
    diagnostics: dict[str, Any] = {}

    structure_changed = _structure_signature(parent) != _structure_signature(child)
    distribution_changed = _distribution_signature(parent) != _distribution_signature(child)
    coupling_changed = _coupling_signature(parent) != _coupling_signature(child)
    scalar_changes = _state_diff(_collect_scalar_state(parent), _collect_scalar_state(child))
    categorical_changes = _state_diff(_collect_categorical_state(parent), _collect_categorical_state(child))

    diagnostics["structure_changed"] = structure_changed
    diagnostics["distribution_changed"] = distribution_changed
    diagnostics["coupling_changed"] = coupling_changed
    diagnostics["scalar_changes"] = scalar_changes
    diagnostics["categorical_changes"] = categorical_changes

    if structure_changed and not distribution_changed and not coupling_changed and not scalar_changes and not categorical_changes:
        before_count = len(parent.family_definition.subgroups)
        after_count = len(child.family_definition.subgroups)
        allowed_types.add("structural_expand" if after_count >= before_count else "structural_reduce")

    if distribution_changed and not coupling_changed and not scalar_changes:
        allowed_types.add("distribution_change")
        if structure_changed:
            before_count = len(parent.family_definition.subgroups)
            after_count = len(child.family_definition.subgroups)
            allowed_types.add("structural_expand" if after_count >= before_count else "structural_reduce")

    if coupling_changed and not structure_changed and not distribution_changed and not scalar_changes:
        allowed_types.add("coupling_change")

    if len(scalar_changes) == 1 and not structure_changed and not distribution_changed and not coupling_changed:
        allowed_types.add("scalar_tune")

    if len(categorical_changes) == 1 and not structure_changed and not distribution_changed and not coupling_changed and not scalar_changes:
        allowed_types.add("categorical_swap")

    return allowed_types, diagnostics


def infer_allowed_edit_types(
    proposal: Proposal,
    runtime_config: Mapping[str, Any],
    *,
    parent_proposal: Proposal | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Infer which edit types are semantically valid for a proposal relative to its parent."""

    if parent_proposal is None:
        return ({"initial_seed"} if proposal.parent_proposal_id is None else set()), {"bootstrap": True}
    compiled_parent = compile_proposal(parent_proposal, runtime_config)
    compiled_child = compile_proposal(proposal, runtime_config)
    return _classify_semantic_diff(compiled_parent, compiled_child)


def validate_proposal(
    proposal: Proposal,
    runtime_config: Mapping[str, Any],
    *,
    parent_proposal: Proposal | None = None,
) -> CompiledFamilySpec:
    """Validate proposal completeness, physical ranges, and semantic edit sparsity."""

    if proposal.edit_type not in SUPPORTED_EDIT_TYPES:
        raise ProposalValidationError(f"Unsupported edit_type: {proposal.edit_type}")
    if proposal.family_definition.family_type not in SUPPORTED_FAMILY_TYPES:
        raise ProposalValidationError(f"Unsupported family_type: {proposal.family_definition.family_type}")
    if proposal.family_definition.distribution_rule.form not in SUPPORTED_DISTRIBUTIONS:
        raise ProposalValidationError(
            f"Unsupported distribution form: {proposal.family_definition.distribution_rule.form}"
        )
    if proposal.family_definition.coupling_rule.rule not in SUPPORTED_COUPLINGS:
        raise ProposalValidationError(f"Unsupported coupling rule: {proposal.family_definition.coupling_rule.rule}")
    if proposal.sampling_plan.sampler_name not in SUPPORTED_SAMPLERS:
        raise ProposalValidationError(f"Unsupported sampler: {proposal.sampling_plan.sampler_name}")
    if proposal.sampling_plan.n_samples <= 0:
        raise ProposalValidationError("sampling_plan.n_samples must be positive")

    constraints = dict(runtime_config.get("constraints", {}))
    for name, spec in proposal.family_definition.continuous_parameters.items():
        _validate_parameter_spec(name, spec, constraints)
    for subgroup in proposal.family_definition.subgroups:
        _validate_parameter_spec(f"{subgroup.name}.count", subgroup.count, {"min": 1, "max": 16})
        _validate_parameter_spec(f"{subgroup.name}.offset_center", subgroup.offset_center, {})
        _validate_parameter_spec(f"{subgroup.name}.spread", subgroup.spread, {"min": 0.0})
        if subgroup.weight_fraction is not None:
            _validate_parameter_spec(f"{subgroup.name}.weight_fraction", subgroup.weight_fraction, {"min": 0.0, "max": 1.0})

    compiled = compile_proposal(proposal, runtime_config)

    if parent_proposal is None:
        if proposal.edit_type != "initial_seed":
            raise ProposalValidationError("The first proposal must use edit_type='initial_seed'")
        if proposal.parent_proposal_id is not None:
            raise ProposalValidationError("The first proposal must not set parent_proposal_id")
        return compiled

    if proposal.parent_proposal_id != parent_proposal.proposal_id:
        raise ProposalValidationError(
            f"parent_proposal_id must reference the last attempted proposal: expected {parent_proposal.proposal_id}, got {proposal.parent_proposal_id}"
        )

    compiled_parent = compile_proposal(parent_proposal, runtime_config)
    allowed_types, diagnostics = _classify_semantic_diff(compiled_parent, compiled)
    if proposal.edit_type not in allowed_types:
        raise ProposalValidationError(
            f"Proposal edit_type={proposal.edit_type} does not match semantic diff. Allowed types: {sorted(allowed_types)}. Diagnostics: {diagnostics}"
        )

    before = proposal.primary_edit.get("before")
    after = proposal.primary_edit.get("after")
    if before == after:
        raise ProposalValidationError("primary_edit.before and primary_edit.after must differ")
    return compiled
