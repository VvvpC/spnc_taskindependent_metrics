from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


SUPPORTED_EDIT_TYPES = {
    "initial_seed",
    "scalar_tune",
    "categorical_swap",
    "structural_expand",
    "structural_reduce",
    "coupling_change",
    "distribution_change",
}
SUPPORTED_SAMPLERS = {"random", "latin_hypercube"}
SUPPORTED_FAMILY_TYPES = {"single_distribution", "bimodal_core_tail"}
SUPPORTED_DISTRIBUTIONS = {"gradient", "random", "normaldistribution"}
SUPPORTED_COUPLINGS = {"independent", "positive_correlation", "negative_correlation", "locked_ratio"}
SUPPORTED_PARAMETER_KINDS = {"fixed", "uniform_float", "uniform_int", "categorical"}


def _ensure_mapping(payload: Mapping[str, Any] | None, *, label: str) -> Mapping[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return payload


@dataclass(frozen=True)
class ParameterSpec:
    kind: str
    value: Any | None = None
    low: float | int | None = None
    high: float | int | None = None
    choices: list[Any] = field(default_factory=list)

    @classmethod
    def fixed(cls, value: Any) -> "ParameterSpec":
        return cls(kind="fixed", value=value)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | Any) -> "ParameterSpec":
        if not isinstance(payload, Mapping):
            return cls.fixed(payload)
        kind = str(payload.get("kind", "fixed"))
        if kind not in SUPPORTED_PARAMETER_KINDS:
            raise ValueError(f"Unsupported parameter kind: {kind}")
        return cls(
            kind=kind,
            value=payload.get("value"),
            low=payload.get("low"),
            high=payload.get("high"),
            choices=list(payload.get("choices", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.kind == "fixed":
            payload["value"] = self.value
        if self.low is not None:
            payload["low"] = self.low
        if self.high is not None:
            payload["high"] = self.high
        if self.choices:
            payload["choices"] = list(self.choices)
        return payload


@dataclass(frozen=True)
class SubgroupDefinition:
    name: str
    role: str
    count: ParameterSpec
    offset_center: ParameterSpec
    spread: ParameterSpec
    distribution_form: str | None = None
    weight_fraction: ParameterSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SubgroupDefinition":
        data = dict(_ensure_mapping(payload, label="subgroup"))
        return cls(
            name=str(data["name"]),
            role=str(data.get("role", data["name"])),
            count=ParameterSpec.from_mapping(data.get("count", 1)),
            offset_center=ParameterSpec.from_mapping(data.get("offset_center", 0.0)),
            spread=ParameterSpec.from_mapping(data.get("spread", 0.0)),
            distribution_form=str(data["distribution_form"]) if data.get("distribution_form") is not None else None,
            weight_fraction=(
                ParameterSpec.from_mapping(data["weight_fraction"]) if data.get("weight_fraction") is not None else None
            ),
            metadata=dict(_ensure_mapping(data.get("metadata"), label="subgroup.metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "role": self.role,
            "count": self.count.to_dict(),
            "offset_center": self.offset_center.to_dict(),
            "spread": self.spread.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.distribution_form is not None:
            payload["distribution_form"] = self.distribution_form
        if self.weight_fraction is not None:
            payload["weight_fraction"] = self.weight_fraction.to_dict()
        return payload


@dataclass(frozen=True)
class DistributionRule:
    form: str
    clip_beta_to_positive: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DistributionRule":
        data = dict(_ensure_mapping(payload, label="distribution_rule"))
        return cls(
            form=str(data.get("form", "random")),
            clip_beta_to_positive=bool(data.get("clip_beta_to_positive", True)),
            metadata=dict(_ensure_mapping(data.get("metadata"), label="distribution_rule.metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "clip_beta_to_positive": self.clip_beta_to_positive,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CouplingRule:
    rule: str
    strength: ParameterSpec | None = None
    mix_ratio: ParameterSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CouplingRule":
        data = dict(_ensure_mapping(payload, label="coupling_rule"))
        return cls(
            rule=str(data.get("rule", "independent")),
            strength=ParameterSpec.from_mapping(data["strength"]) if data.get("strength") is not None else None,
            mix_ratio=ParameterSpec.from_mapping(data["mix_ratio"]) if data.get("mix_ratio") is not None else None,
            metadata=dict(_ensure_mapping(data.get("metadata"), label="coupling_rule.metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "rule": self.rule,
            "metadata": dict(self.metadata),
        }
        if self.strength is not None:
            payload["strength"] = self.strength.to_dict()
        if self.mix_ratio is not None:
            payload["mix_ratio"] = self.mix_ratio.to_dict()
        return payload


@dataclass(frozen=True)
class FamilyDefinition:
    family_type: str
    topology_name: str
    subgroups: list[SubgroupDefinition]
    distribution_rule: DistributionRule
    coupling_rule: CouplingRule
    continuous_parameters: dict[str, ParameterSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FamilyDefinition":
        data = dict(_ensure_mapping(payload, label="family_definition"))
        return cls(
            family_type=str(data["family_type"]),
            topology_name=str(data.get("topology_name", data["family_type"])),
            subgroups=[SubgroupDefinition.from_mapping(item) for item in data.get("subgroups", [])],
            distribution_rule=DistributionRule.from_mapping(data.get("distribution_rule")),
            coupling_rule=CouplingRule.from_mapping(data.get("coupling_rule")),
            continuous_parameters={
                key: ParameterSpec.from_mapping(value)
                for key, value in dict(_ensure_mapping(data.get("continuous_parameters"), label="continuous_parameters")).items()
            },
            metadata=dict(_ensure_mapping(data.get("metadata"), label="family_definition.metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_type": self.family_type,
            "topology_name": self.topology_name,
            "subgroups": [item.to_dict() for item in self.subgroups],
            "distribution_rule": self.distribution_rule.to_dict(),
            "coupling_rule": self.coupling_rule.to_dict(),
            "continuous_parameters": {key: spec.to_dict() for key, spec in self.continuous_parameters.items()},
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SamplingPlan:
    sampler_name: str
    n_samples: int
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "SamplingPlan":
        data = dict(_ensure_mapping(payload, label="sampling_plan"))
        return cls(
            sampler_name=str(data.get("sampler_name", "latin_hypercube")),
            n_samples=int(data.get("n_samples", 4)),
            seed=int(data.get("seed", 1234)),
            metadata=dict(_ensure_mapping(data.get("metadata"), label="sampling_plan.metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler_name": self.sampler_name,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    parent_proposal_id: str | None
    edit_type: str
    primary_edit: dict[str, Any]
    rationale: str
    expected_effect: str
    family_definition: FamilyDefinition
    sampling_plan: SamplingPlan
    auto_completed_fields: list[str] = field(default_factory=list)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Proposal":
        data = dict(_ensure_mapping(payload, label="proposal"))
        return cls(
            proposal_id=str(data["proposal_id"]),
            parent_proposal_id=(
                str(data["parent_proposal_id"]) if data.get("parent_proposal_id") not in (None, "") else None
            ),
            edit_type=str(data["edit_type"]),
            primary_edit=dict(_ensure_mapping(data.get("primary_edit"), label="primary_edit")),
            rationale=str(data.get("rationale", "")),
            expected_effect=str(data.get("expected_effect", "")),
            family_definition=FamilyDefinition.from_mapping(
                dict(_ensure_mapping(data.get("family_definition"), label="family_definition"))
            ),
            sampling_plan=SamplingPlan.from_mapping(data.get("sampling_plan")),
            auto_completed_fields=[str(item) for item in list(data.get("auto_completed_fields", []))],
            notes=str(data.get("notes", "")),
            metadata=dict(_ensure_mapping(data.get("metadata"), label="proposal.metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "parent_proposal_id": self.parent_proposal_id,
            "edit_type": self.edit_type,
            "primary_edit": dict(self.primary_edit),
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "family_definition": self.family_definition.to_dict(),
            "sampling_plan": self.sampling_plan.to_dict(),
            "auto_completed_fields": list(self.auto_completed_fields),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CompiledFamilySpec:
    proposal_id: str
    parent_proposal_id: str | None
    edit_type: str
    primary_edit: dict[str, Any]
    rationale: str
    expected_effect: str
    family_definition: FamilyDefinition
    sampling_plan: SamplingPlan
    auto_completed_fields: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_proposal(cls, proposal: Proposal, *, auto_completed_fields: list[str]) -> "CompiledFamilySpec":
        return cls(
            proposal_id=proposal.proposal_id,
            parent_proposal_id=proposal.parent_proposal_id,
            edit_type=proposal.edit_type,
            primary_edit=dict(proposal.primary_edit),
            rationale=proposal.rationale,
            expected_effect=proposal.expected_effect,
            family_definition=proposal.family_definition,
            sampling_plan=proposal.sampling_plan,
            auto_completed_fields=list(auto_completed_fields),
            metadata=dict(proposal.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "parent_proposal_id": self.parent_proposal_id,
            "edit_type": self.edit_type,
            "primary_edit": dict(self.primary_edit),
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "family_definition": self.family_definition.to_dict(),
            "sampling_plan": self.sampling_plan.to_dict(),
            "auto_completed_fields": list(self.auto_completed_fields),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SampledInstance:
    sample_index: int
    sampled_parameters: dict[str, Any]
    subgroup_details: list[dict[str, Any]]
    beta_offsets: list[float]
    weights: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "sampled_parameters": dict(self.sampled_parameters),
            "subgroup_details": [dict(item) for item in self.subgroup_details],
            "beta_offsets": list(self.beta_offsets),
            "weights": list(self.weights),
        }


@dataclass(frozen=True)
class ScoreBundle:
    baseline_points: list[dict[str, float]]
    baseline_frontier: list[dict[str, float]]
    baseline_hv: float
    family_points: list[dict[str, float]]
    family_frontier: list[dict[str, float]]
    family_hv: float
    combined_frontier: list[dict[str, float]]
    combined_hv: float
    score_s_abs: float
    audit_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_points": [dict(item) for item in self.baseline_points],
            "baseline_frontier": [dict(item) for item in self.baseline_frontier],
            "baseline_hv": self.baseline_hv,
            "family_points": [dict(item) for item in self.family_points],
            "family_frontier": [dict(item) for item in self.family_frontier],
            "family_hv": self.family_hv,
            "combined_frontier": [dict(item) for item in self.combined_frontier],
            "combined_hv": self.combined_hv,
            "score_s_abs": self.score_s_abs,
            "audit_metrics": dict(self.audit_metrics),
        }
