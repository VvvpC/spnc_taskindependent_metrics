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
 'primary_edit': {'target': 'continuous_parameters.gamma.high', 'before': 0.08, 'after': 0.09},
 'rationale': 'Following the theta expansion, we slightly increase the upper bound of gamma '
              '(inter-dot coupling strength) from 0.08 to 0.09 to explore configurations with '
              'stronger coupling. This minimal expansion may enable designs with improved MC '
              'performance by allowing tighter dot clustering while maintaining conservative '
              'constraints.',
 'expected_effect': 'The expanded gamma range should enable sampling of configurations with '
                    'stronger inter-dot coupling, potentially pushing the Pareto frontier outward '
                    'in the MC dimension. The hypervolume contribution S_abs is expected to be '
                    'positive but modest due to the minimal nature of this scalar tune.',
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
                                                           'high': 0.09},
                                                 'm0': {'kind': 'uniform_float',
                                                        'low': 0.008,
                                                        'high': 0.015}},
                       'metadata': {}},
 'sampling_plan': {'sampler_name': 'latin_hypercube',
                   'n_samples': 10,
                   'seed': 1234,
                   'metadata': {}},
 'auto_completed_fields': [],
 'notes': 'Minimal scalar tune of gamma parameter following theta expansion strategy.',
 'metadata': {}}




def main() -> None:
    run_train_file(CURRENT_PROPOSAL)


if __name__ == "__main__":
    main()


