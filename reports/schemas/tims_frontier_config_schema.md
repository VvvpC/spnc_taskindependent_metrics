# TIMs Frontier Config Schema

- Document type: Config Schema
- Schema version: `0.3.1`
- Status: Accepted baseline
- Scope: Phase 1 TIMs frontier workflow
- Aligned specification: `reports/specifications/tims_frontier_specification.md`
- Aligned result schema: `reports/schemas/tims_frontier_result_schema.md`

This document defines the official study-level configuration contract for the
current workflow. It is the source of truth for study configuration structure,
field ownership, and configuration semantics for the `uniform vs heterogeneous`
TIMs frontier comparison.

## 1. Scope

This schema is designed for the current first-phase workflow only:

- primary frontier: `MC-CQ`
- compared families: `uniform` and `heterogeneous`
- exploration method: Optuna
- optimized objectives: `MC` and `CQ`
- recorded TIMs: `MC`, `KR`, `GR`, `CQ`
- current heterogeneous scheme: `random`
- shared trial-level seed rule across families

## 2. Design Rules

- `comparison.primary_endpoint` is the single source of truth for the primary
  frontier definition.
- `analysis.frontier` references `comparison.primary_endpoint` rather than
  duplicating metric definitions.
- `Nvirt` is defined once under
  `exploration.shared_non_geometric_search.Nvirt`.
- Reservoir construction binds `Nvirt` from the exploration block through
  `reservoir.parameter_binding.Nvirt_source`.
- `CQ` is recorded explicitly but is derived using `CQ = KR - GR`.
- `GR` direction is `minimize`.
- `KR` and `GR` are recorded but are not primary optimization objectives.

## 3. Top-Level Keys

The current schema defines the following top-level blocks:

```text
schema_version
specification_ref
study
comparison
reservoir
families
evaluation
exploration
execution
storage
analysis
reporting
```

## 4. Canonical YAML

```yaml
schema_version: "0.3.1"

specification_ref:
  spec_id: "tims_frontier_specification"
  spec_status: "accepted-baseline"
  spec_path: "reports/specifications/tims_frontier_specification.md"

study:
  study_id: "uniform_vs_heterogeneous_tims_frontier_v1"
  title: "均匀与异质纳米点储层的 MC-CQ Frontier 比较"
  phase: "phase1_tims_frontier"
  objective: "compare_family_level_mc_cq_frontier"
  description: "在共享 non-geometric 搜索协议下比较 uniform 与 heterogeneous family 的 MC-CQ Pareto frontier。"
  tags:
    - "TIMs"
    - "frontier"
    - "uniform-vs-heterogeneous"
    - "optuna"
    - "phase1"

comparison:
  compared_families:
    - "uniform"
    - "heterogeneous"

  baseline_family: "uniform"
  candidate_family: "heterogeneous"

  primary_endpoint:
    kind: "pareto_frontier_difference"
    frontier_objectives:
      - "MC"
      - "CQ"
    directions:
      MC: "maximize"
      CQ: "maximize"
    output_artifact: "frontier_points_table"

  secondary_endpoints:
    - "KR_summary"
    - "GR_summary"
    - "point_cloud_summary"
    - "metric_distribution_summary"

  fairness_protocol:
    shared_non_geometric_search: true
    shared_trial_budget: true
    shared_exploration_strategy: true
    shared_evaluators: true
    shared_trial_seed_rule: true
    geometry_match_required: false

reservoir:
  backend:
    uniform_builder: "legacy_uniform_spnc"
    heterogeneous_builder: "legacy_morphology_reservoir"

  shared_defaults:
    physical:
      h: 0.4
      theta_H: 90
      k_s_0: 0
      phi: 45

    network:
      bias: true
      Nwarmup: 0

    simulation:
      delay_feedback: 0
      voltage_noise: false
      johnson_noise: false
      thermal_noise: false

    mask:
      fixed_mask: true
      seed_mask_mode: "derived_from_trial_seed"
      max_sequences_mask: false

  parameter_binding:
    reference_beta_param: "beta_prime"
    Nvirt_source: "exploration.shared_non_geometric_search.Nvirt"

families:
  uniform:
    enabled: true
    family_kind: "uniform"
    label: "Uniform reservoir family"

    construction:
      geometry_mode: "uniform"
      parameters: {}

  heterogeneous:
    enabled: true
    family_kind: "heterogeneous"
    label: "Heterogeneous reservoir family"

    construction:
      geometry_mode: "distributional_beta"

      morphology:
        scheme:
          type: "fixed"
          value: "random"

        reference_beta_source: "exploration.shared_non_geometric_search.beta_prime"

        n_instances:
          type: "fixed"
          value: 5

        beta_spread:
          type: "fixed"
          value: 5.0

        beta_sampling_rule: "uniform_in_reference_plus_minus_spread"
        clip_beta_to_positive: true

        weights_mode:
          type: "fixed"
          value: "equal"

        morphology_seed:
          mode: "derived_from_trial_seed"

evaluation:
  tims:
    recorded_metrics:
      - "MC"
      - "KR"
      - "GR"
      - "CQ"

    primary_metrics:
      - "MC"
      - "CQ"

    secondary_metrics:
      - "KR"
      - "GR"

    metric_directions:
      MC: "maximize"
      KR: "maximize"
      GR: "minimize"
      CQ: "maximize"

    mc:
      signal_len: 550
      delays: 10
      splits:
        - 0.2
        - 0.6
      signal_seed_mode: "derived_from_trial_seed"

    kr_gr:
      n_readouts: "inherit_from_Nvirt"
      n_wash: 10
      threshold: 0.001
      signal_seed_mode: "derived_from_trial_seed"
      state_normalization: "divide_by_max"

    cq:
      mode: "derived"
      formula: "KR_minus_GR"
      expression: "CQ = KR - GR"
      source_metrics:
        - "KR"
        - "GR"
      store_as_explicit_metric: true

  tasks:
    enabled: false

exploration:
  method: "optuna"
  study_mode: "per_family_independent_studies"

  shared_non_geometric_search:
    beta_prime:
      type: "float"
      low: 20.0
      high: 50.0

    theta:
      type: "float"
      low: 0.001
      high: 0.6

    gamma:
      type: "float"
      low: 0.01
      high: 0.3

    m0:
      type: "float"
      low: 0.003
      high: 0.06

    Nvirt:
      type: "fixed"
      value: 100

  optuna:
    objective_ref: "comparison.primary_endpoint"
    sampler: "NSGAIISampler"
    sampler_seed: 1234
    pruner: null
    n_trials_per_family: 500
    n_startup_trials: 50
    study_direction_mode: "multi_objective"

  coupling:
    shared_trial_seed_rule: true
    identical_trial_suggestions_across_families: false

  budget:
    equal_budget_across_families: true
    max_failures_per_family: 100
    timeout_seconds: null

execution:
  run_id_mode: "auto"

  seed_policy:
    global_seed: 1234
    trial_seed_mode: "deterministic_from_trial_index"
    matched_trial_seed_across_families: true
    derived_seed_targets:
      - "input_signal"
      - "mask"
      - "morphology"

  parallelism:
    max_workers: 1

  retry:
    max_attempts_per_trial: 2
    retry_on_exception: true

  resume:
    enabled: true
    resume_mode: "from_manifest"

  failure_policy:
    on_trial_failure: "record_and_continue"

  logging:
    level: "INFO"
    capture_stdout: true
    save_trial_logs: true

storage:
  root_dir: "results"
  raw_dir: "results/raw"
  processed_dir: "results/processed"
  report_dir: "reports"

  path_templates:
    study_root: "results/raw/{study_id}"
    run_root: "results/raw/{study_id}/{run_id}"
    processed_study_root: "results/processed/{study_id}"

  save:
    user_config_copy: true
    resolved_config: true
    trial_results_table_csv: true
    frontier_points_table_csv: true
    optuna_study_summary_json: true
    failure_log_json: true
    runtime_metadata_json: true
    intermediate_states_npz: false
    figures: true

  formats:
    table: "csv"
    metadata: "json"
    arrays: "npz"
    figures: "png"

  provenance:
    record_git_commit: true
    record_git_branch: true
    record_hostname: true
    record_python_version: true

analysis:
  primary_output: "frontier_points_table"

  frontier:
    endpoint_ref: "comparison.primary_endpoint"
    source_table: "trial_results"
    algorithm: "pareto_nondominated"
    dominance_epsilon: 0.0
    deduplicate_points: true
    deduplicate_tolerance: 1.0e-12

  comparison:
    baseline_family: "uniform"
    candidate_family: "heterogeneous"
    compute_frontier_difference: true
    compute_nondominated_overlap: true
    compute_point_cloud_summary: true

  secondary_metric_summaries:
    enabled: true
    metrics:
      KR: "maximize"
      GR: "minimize"

reporting:
  language: "zh-CN"

  outputs:
    markdown_summary: true
    json_summary: true
    figure_manifest: true
    frontier_points_table: true

  figures:
    generate_primary_frontier_plot: true
    primary_projection:
      - "MC"
      - "CQ"
    generate_secondary_metric_plots: true

  include:
    study_metadata: true
    comparison_protocol: true
    key_findings: true
    failure_summary: true
    caveats: true
```

## 5. Ownership Notes

- `comparison.primary_endpoint` owns the scientific definition of the primary
  frontier.
- `exploration.optuna.objective_ref` reuses that definition for optimization.
- `analysis.frontier.endpoint_ref` reuses that definition for processed
  frontier extraction.
- `evaluation.tims.metric_directions` defines evaluation semantics for all
  recorded TIMs, including secondary metrics.
- `storage` defines output expectations but does not redefine result schemas.

## 6. Companion Documents

This config schema is intended to work together with:

- `reports/specifications/tims_frontier_specification.md`
- `reports/schemas/tims_frontier_result_schema.md`
- `reports/schemas/tims_frontier_resolved_config_schema.md`
- `reports/schemas/tims_frontier_run_manifest_schema.md`

Future optional companions may include:

- example study config files
