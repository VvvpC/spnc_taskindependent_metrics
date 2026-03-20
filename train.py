from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
for relative_path in [
    "src",
    "src/Morphology_Research",
    "src/Project",
    "src/Optuna_TaskIndependent_Metrics",
    "src/ParetoFront_CQandMC",
    "src/Plot_Functions",
    "src/Test_Temporary",
]:
    source_dir = REPO_ROOT / relative_path
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

from tims_frontier.autoresearch import run_train_file



# This is the only file the autoresearch agent is allowed to edit.
CURRENT_PROPOSAL = {'proposal_id': 'proposal_0002',
 'parent_proposal_id': 'proposal_0001',
 'edit_type': 'scalar_tune',
 'primary_edit': {'target': 'continuous_parameters.theta.high', 'before': 0.24, 'after': 0.25},
 'rationale': 'Slightly expand the theta parameter upper bound from 0.24 to 0.25 to increase the '
              'exploration space for rotational coupling strength. This minimal expansion may '
              'allow discovery of designs with improved MC performance while maintaining the '
              'conservative spread constraints established in the baseline.',
 'expected_effect': 'The expanded theta range should enable sampling of configurations with '
                    'slightly stronger rotational coupling, potentially pushing the Pareto '
                    'frontier outward in the MC dimension. The hypervolume contribution S_abs is '
                    'expected to be positive but modest due to the minimal nature of this scalar '
                    'tune.',
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
                                                           'high': 0.25},
                                                 'gamma': {'kind': 'uniform_float',
                                                           'low': 0.04,
                                                           'high': 0.08},
                                                 'm0': {'kind': 'uniform_float',
                                                        'low': 0.008,
                                                        'high': 0.015}},
                       'metadata': {}},
 'sampling_plan': {'sampler_name': 'latin_hypercube', 'n_samples': 3, 'seed': 1234, 'metadata': {}},
 'auto_completed_fields': [],
 'notes': 'Subsequent rounds should keep proposal_id monotonic and apply only one minimal semantic '
          'edit.',
 'metadata': {}}




def main() -> None:
    run_train_file(CURRENT_PROPOSAL)


if __name__ == "__main__":
    main()


