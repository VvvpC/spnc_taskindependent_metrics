from __future__ import annotations

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in [REPO_ROOT / "src"]:
    entry_text = str(entry)
    if entry_text not in sys.path:
        sys.path.insert(0, entry_text)

from tims_frontier.autoresearch.compiler import compile_proposal
from tims_frontier.autoresearch.models import Proposal
from tims_frontier.autoresearch.validator import ProposalValidationError, validate_proposal


RUNTIME_CONFIG = {
    "constraints": {
        "beta_prime": {"min": 5.0, "max": 50.0},
        "theta": {"min": 0.01, "max": 0.8},
        "gamma": {"min": 0.001, "max": 0.3},
        "m0": {"min": 0.001, "max": 0.05},
    }
}


def base_proposal_dict() -> dict:
    return {
        "proposal_id": "proposal_0001",
        "parent_proposal_id": None,
        "edit_type": "initial_seed",
        "primary_edit": {"target": "initialization", "before": None, "after": "single_distribution"},
        "rationale": "seed",
        "expected_effect": "baseline",
        "family_definition": {
            "family_type": "single_distribution",
            "topology_name": "single_group",
            "subgroups": [
                {
                    "name": "core",
                    "role": "core",
                    "count": {"kind": "uniform_int", "low": 3, "high": 5},
                    "offset_center": {"kind": "fixed", "value": 0.0},
                    "spread": {"kind": "uniform_float", "low": 2.0, "high": 3.0},
                }
            ],
            "distribution_rule": {"form": "random", "clip_beta_to_positive": True},
            "coupling_rule": {"rule": "independent"},
            "continuous_parameters": {
                "beta_prime": {"kind": "uniform_float", "low": 28.0, "high": 32.0},
                "theta": {"kind": "uniform_float", "low": 0.16, "high": 0.24},
                "gamma": {"kind": "uniform_float", "low": 0.04, "high": 0.08},
                "m0": {"kind": "uniform_float", "low": 0.008, "high": 0.015},
            },
        },
        "sampling_plan": {"sampler_name": "latin_hypercube", "n_samples": 3, "seed": 1234},
    }


class ProposalValidatorTests(unittest.TestCase):
    def test_initial_seed_proposal_is_valid(self) -> None:
        proposal = Proposal.from_mapping(base_proposal_dict())
        compiled = validate_proposal(proposal, RUNTIME_CONFIG)
        self.assertEqual(compiled.proposal_id, "proposal_0001")

    def test_scalar_tune_allows_one_numeric_change(self) -> None:
        parent = Proposal.from_mapping(base_proposal_dict())
        child_payload = base_proposal_dict()
        child_payload["proposal_id"] = "proposal_0002"
        child_payload["parent_proposal_id"] = "proposal_0001"
        child_payload["edit_type"] = "scalar_tune"
        child_payload["primary_edit"] = {
            "target": "continuous_parameters.gamma.high",
            "before": 0.08,
            "after": 0.09,
        }
        child_payload["family_definition"]["continuous_parameters"]["gamma"]["high"] = 0.09
        child = Proposal.from_mapping(child_payload)
        compiled = validate_proposal(child, RUNTIME_CONFIG, parent_proposal=parent)
        self.assertEqual(compiled.edit_type, "scalar_tune")

    def test_multiple_scalar_changes_are_rejected(self) -> None:
        parent = Proposal.from_mapping(base_proposal_dict())
        child_payload = base_proposal_dict()
        child_payload["proposal_id"] = "proposal_0002"
        child_payload["parent_proposal_id"] = "proposal_0001"
        child_payload["edit_type"] = "scalar_tune"
        child_payload["primary_edit"] = {
            "target": "continuous_parameters.gamma.high",
            "before": 0.08,
            "after": 0.09,
        }
        child_payload["family_definition"]["continuous_parameters"]["gamma"]["high"] = 0.09
        child_payload["family_definition"]["continuous_parameters"]["theta"]["high"] = 0.25
        child = Proposal.from_mapping(child_payload)
        with self.assertRaises(ProposalValidationError):
            validate_proposal(child, RUNTIME_CONFIG, parent_proposal=parent)

    def test_bimodal_auto_completion_adds_tail(self) -> None:
        payload = base_proposal_dict()
        payload["family_definition"]["family_type"] = "bimodal_core_tail"
        payload["family_definition"]["topology_name"] = "core_tail"
        compiled = compile_proposal(Proposal.from_mapping(payload), RUNTIME_CONFIG)
        subgroup_names = [item.name for item in compiled.family_definition.subgroups]
        self.assertIn("core", subgroup_names)
        self.assertIn("tail", subgroup_names)
        self.assertTrue(any("tail" in item for item in compiled.auto_completed_fields))


if __name__ == "__main__":
    unittest.main()
