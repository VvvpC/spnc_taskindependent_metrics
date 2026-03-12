# TIMs Frontier Resolved Config Schema

- Document type: Resolved Config Schema
- Schema version: `1.0`
- Status: Accepted baseline
- Scope: Phase 1 TIMs frontier workflow
- Aligned specification: `reports/specifications/tims_frontier_specification.md`
- Aligned study config schema: `reports/schemas/tims_frontier_config_schema.md`
- Aligned result schema: `reports/schemas/tims_frontier_result_schema.md`

This document defines the official resolved-config contract for the current
workflow. A resolved config is the immutable, fully materialized run-level
configuration snapshot produced from the study config before trial execution
starts.

## 1. Role

The resolved config exists to answer one question precisely:

> What exact configuration did this run use after defaults, references, and
> runtime paths were resolved?

It is not:

- the user-authored study config
- a per-trial parameter record
- a mutable runtime state file

## 2. Relationship to Other Documents

- `tims_frontier_specification.md`
  - defines the scientific objective and workflow intent
- `tims_frontier_config_schema.md`
  - defines the study-level configuration contract
- `tims_frontier_resolved_config_schema.md`
  - defines the fully materialized run-level configuration snapshot
- `tims_frontier_run_manifest_schema.md`
  - defines mutable orchestration state during execution
- `tims_frontier_result_schema.md`
  - defines output artifacts after execution

## 3. Core Rules

- The resolved config is written once per run.
- It must be immutable after creation.
- It must not contain unresolved references such as:
  - `objective_ref`
  - `endpoint_ref`
  - `Nvirt_source`
  - `reference_beta_source`
- It may preserve source provenance through explicit `source_trace` fields.
- It may include run-level concrete paths.
- It must not contain sampled per-trial values.

Recommended path:

```text
results/raw/{study_id}/{run_id}/resolved_config.yaml
```

## 4. Top-Level Structure

```text
schema_version
source_config_schema_version
source_config_path
specification_ref
result_schema_ref
run
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
source_trace
```

## 5. Canonical YAML

```yaml
schema_version: "1.0"
source_config_schema_version: "0.3.1"
source_config_path: "configs/studies/uniform_vs_heterogeneous_tims_frontier_v1.yaml"

specification_ref:
  spec_id: "tims_frontier_specification"
  spec_path: "reports/specifications/tims_frontier_specification.md"

result_schema_ref:
  schema_id: "tims_frontier_result_schema"
  schema_version: "1.0"
  schema_path: "reports/schemas/tims_frontier_result_schema.md"

run:
  run_id: "run_20260311_120000"
  created_at_utc: "2026-03-11T12:00:00Z"
  config_digest: "sha256:example_digest"
  mode: "study_run"
  status_at_creation: "created"

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
      Nvirt: 100
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
  bindings:
    reference_beta_param: "beta_prime"

families:
  uniform:
    enabled: true
    family_kind: "uniform"
    label: "Uniform reservoir family"
    resolved_construction:
      geometry_mode: "uniform"
      parameters: {}
  heterogeneous:
    enabled: true
    family_kind: "heterogeneous"
    label: "Heterogeneous reservoir family"
    resolved_construction:
      geometry_mode: "distributional_beta"
      morphology:
        scheme: "random"
        reference_beta_param: "beta_prime"
        n_instances: 5
        beta_spread: 5.0
        beta_sampling_rule: "uniform_in_reference_plus_minus_spread"
        clip_beta_to_positive: true
        weights_mode: "equal"
        morphology_seed_mode: "derived_from_trial_seed"

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
      n_readouts_from: "Nvirt"
      n_readouts_value: 100
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
  search_space:
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
    objective_metrics:
      - "MC"
      - "CQ"
    objective_directions:
      MC: "maximize"
      CQ: "maximize"
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
  paths:
    study_root: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1"
    run_root: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000"
    processed_study_root: "results/processed/uniform_vs_heterogeneous_tims_frontier_v1"
    resolved_config_path: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/resolved_config.yaml"
    trial_results_table_path: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/trial_results_table.csv"
    failure_log_path: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/failure_log.jsonl"
    run_summary_path: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/run_summary.json"
    run_manifest_path: "results/raw/uniform_vs_heterogeneous_tims_frontier_v1/run_20260311_120000/run_manifest.json"
    frontier_points_table_path: "results/processed/uniform_vs_heterogeneous_tims_frontier_v1/frontier_points_table.csv"
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
    metrics:
      - "MC"
      - "CQ"
    directions:
      MC: "maximize"
      CQ: "maximize"
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

source_trace:
  exploration.optuna.objectives_resolved_from: "comparison.primary_endpoint"
  analysis.frontier_resolved_from: "comparison.primary_endpoint"
  reservoir.shared_defaults.network.Nvirt_resolved_from: "exploration.search_space.Nvirt"
  families.heterogeneous.resolved_construction.morphology.reference_beta_param_resolved_from: "exploration.search_space.beta_prime"
```

## 6. Field Ownership Notes

- `run` captures immutable run-identity metadata.
- `storage.paths` stores concrete run-level output paths.
- `comparison.primary_endpoint` is expanded and no longer referenced indirectly.
- `exploration.optuna.objective_metrics` is concrete and no longer uses
  `objective_ref`.
- `analysis.frontier.metrics` is concrete and no longer uses `endpoint_ref`.
- `reservoir.shared_defaults.network.Nvirt` stores the resolved fixed value.

## 7. Non-Goals

This schema does not include:

- per-trial sampled parameters
- runtime progress counters
- mutable execution status transitions
- trial result rows

Those belong in the run manifest and result artifacts.
