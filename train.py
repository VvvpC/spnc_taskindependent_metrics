from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from tims_frontier.autoresearch import run_train_file



# This is the only file the autoresearch agent is allowed to edit.
CURRENT_PROPOSAL = {'proposal_id': 'proposal_0001',
 'parent_proposal_id': 'None',
 'edit_type': 'initial_seed',
 'primary_edit': {'target': 'subgroup.core.spread.high', 'before': 3.5, 'after': 3.0},
 'rationale': 'Narrow the maximum spread in the core subgroup from 3.5 to 3.0 to reduce risk of '
              'numerical instability while maintaining heterogeneity. This conservative adjustment '
              'improves first-run success probability without sacrificing design diversity.',
 'expected_effect': 'Produces a slightly more constrained family that generates safer initial '
                    'configurations. The reduced upper bound on spread should prevent extreme '
                    'parameter configurations while preserving sufficient exploration space. '
                    'Hypervolume potential should remain comparable but with improved robustness.',
 'family_definition': {'family_type': 'single_distribution',
                       'topology_name': 'single_group',
                       'subgroups': [{'name': 'core',
                                      'role': 'core',
                                      'count': {'kind': 'uniform_int', 'low': 3, 'high': 5},
                                      'offset_center': {'kind': 'fixed', 'value': 0.0},
                                      'spread': {'kind': 'uniform_float', 'low': 2.0, 'high': 3.0},
                                      'metadata': {}}],
                       'distribution_rule': {'form': 'random',
                                             'clip_beta_to_positive': True,
                                             'metadata': {}},
                       'coupling_rule': {'rule': 'independent', 'metadata': {}},
                       'continuous_parameters': {'beta_prime': {'kind': 'uniform_float',
                                                                'low': 28.0,
                                                                'high': 32.0},
                                                 'theta': {'kind': 'uniform_float',
                                                           'low': 0.16,
                                                           'high': 0.24},
                                                 'gamma': {'kind': 'uniform_float',
                                                           'low': 0.04,
                                                           'high': 0.08},
                                                 'm0': {'kind': 'uniform_float',
                                                        'low': 0.008,
                                                        'high': 0.015}},
                       'metadata': {}},
 'sampling_plan': {'sampler_name': 'latin_hypercube', 'n_samples': 3, 'seed': 1234, 'metadata': {}},
 'notes': 'Subsequent rounds should keep proposal_id monotonic and apply only one minimal semantic '
          'edit.',
 'metadata': {}}




def main() -> None:
    run_train_file(CURRENT_PROPOSAL)


if __name__ == "__main__":
    main()


