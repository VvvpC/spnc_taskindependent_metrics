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
CURRENT_PROPOSAL = {'proposal_id': 'proposal_0001',
 'parent_proposal_id': None,
 'edit_type': 'initial_seed',
 'primary_edit': {'target': 'continuous_parameters.m0.low', 'before': 0.008, 'after': 0.007},
 'rationale': 'Following the pattern where expanding upper bounds improved MC at CQ expense, we '
              'now explore reducing the lower bound of m0 to allow lighter mass configurations. '
              'This could improve dynamic responsiveness and potentially recover some CQ '
              'performance while maintaining MC gains from previous parameter expansions.',
 'expected_effect': 'The expanded m0 range at the lower end should sample configurations with '
                    'lighter effective masses, potentially improving sensitivity and pushing the '
                    'Pareto frontier to achieve better CQ-MC balance.',
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
                                                                'high': 33.0},
                                                 'theta': {'kind': 'uniform_float',
                                                           'low': 0.15,
                                                           'high': 0.25},
                                                 'gamma': {'kind': 'uniform_float',
                                                           'low': 0.04,
                                                           'high': 0.09},
                                                 'm0': {'kind': 'uniform_float',
                                                        'low': 0.007,
                                                        'high': 0.015}},
                       'metadata': {}},
 'sampling_plan': {'sampler_name': 'latin_hypercube',
                   'n_samples': 10,
                   'seed': 1234,
                   'metadata': {}},
 'auto_completed_fields': [],
 'notes': 'Minimal scalar tune of m0.low following theta.low adjustment, targeting CQ recovery.',
 'metadata': {}}




def main() -> None:
    run_train_file(CURRENT_PROPOSAL)


if __name__ == "__main__":
    main()


