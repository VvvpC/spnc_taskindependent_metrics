# TIMs Frontier Result Schema

- Document type: Result Schema
- Schema version: `1.0`
- Status: Accepted baseline
- Scope: Phase 1 TIMs frontier workflow
- Aligned specification: `reports/specifications/tims_frontier_specification.md`
- Aligned config schema: `reports/schemas/tims_frontier_config_schema.md`

This document defines the official result contract for the current workflow.
It is the source of truth for result storage, processed frontier export, and
run-level provenance capture for the `uniform vs heterogeneous` TIMs frontier
study.

## 1. Scope

This schema is designed for the current first-phase workflow only:

- primary frontier: `MC-CQ`
- compared families: `uniform` and `heterogeneous`
- recorded TIMs: `MC`, `KR`, `GR`, `CQ`
- primary processed artifact: frontier points table
- exploration mode: Optuna-based multi-objective search on `MC` and `CQ`

## 2. Artifact Set

The workflow defines four canonical result artifacts:

1. `trial_results_table.csv`
2. `frontier_points_table.csv`
3. `run_summary.json`
4. `failure_log.jsonl`

Recommended paths:

```text
results/raw/{study_id}/{run_id}/trial_results_table.csv
results/raw/{study_id}/{run_id}/run_summary.json
results/raw/{study_id}/{run_id}/failure_log.jsonl
results/processed/{study_id}/frontier_points_table.csv
```

## 3. Global Rules

- Timestamps use `UTC ISO 8601`.
- CSV empty values are left blank.
- `CQ` must be explicitly stored.
- `CQ` is derived with `CQ = KR - GR`.
- `GR` is stored as its raw value and is not sign-flipped.
- `failure_code` is currently an open string field.
- `completed`, `failed`, and `pruned` trials all enter `trial_results_table.csv`.
- `pruned` rows may retain partial metric outputs.
- `frontier_points_table.csv` deduplicates on `MC-CQ` objective values.

## 4. trial_results_table.csv

Semantics:

- one row = one actual trial record
- raw truth table
- includes successful and unsuccessful trials

Primary key:

- `trial_id`

Columns:

```text
result_schema_version
study_id
run_id
trial_id
family
family_role
family_trial_index
optuna_study_name
optuna_trial_number
status
failure_code
failure_message
global_seed
trial_seed
input_signal_seed
mask_seed
morphology_seed
beta_prime
theta
gamma
m0
Nvirt
geometry_mode
morphology_scheme
morphology_n_instances
morphology_beta_spread
morphology_weights_mode
MC
KR
GR
CQ
CQ_formula
runtime_seconds
started_at_utc
finished_at_utc
resolved_config_path
trial_log_path
```

Field rules:

- `family`: `uniform` or `heterogeneous`
- `family_role`: `baseline` or `candidate`
- `status`: `completed`, `failed`, or `pruned`
- `uniform` rows may leave `morphology_*` fields blank
- metric columns may be partially blank for `failed` or `pruned` rows
- `CQ_formula` is currently fixed to `KR_minus_GR`

## 5. frontier_points_table.csv

Semantics:

- one row = one nondominated point for one family under the primary endpoint
- processed table
- main study output for frontier comparison

Primary key:

- `frontier_point_id`

Foreign key:

- `source_trial_id -> trial_results_table.trial_id`

Columns:

```text
result_schema_version
study_id
run_id
frontier_point_id
frontier_scope
endpoint_ref
family
family_role
source_trial_id
source_optuna_trial_number
MC
CQ
KR
GR
beta_prime
theta
gamma
m0
Nvirt
geometry_mode
morphology_scheme
morphology_n_instances
morphology_beta_spread
morphology_weights_mode
pareto_rank
duplicate_count
frontier_algorithm
dominance_epsilon
generated_at_utc
```

Field rules:

- `frontier_scope` is currently expected to be `family_primary_endpoint`
- `endpoint_ref` points to `comparison.primary_endpoint`
- `duplicate_count` is computed by deduplicating on `MC-CQ`
- `pareto_rank` is expected to be `0` for all rows in this table
- `KR` and `GR` remain available for explanation and secondary analysis

## 6. run_summary.json

Semantics:

- run-level summary
- provenance and artifact manifest
- git metadata lives here, not in the trial table

Expected structure:

```json
{
  "result_schema_version": "1.0",
  "study_id": "string",
  "run_id": "string",
  "config_schema_version": "0.3.1",
  "specification_ref": "reports/specifications/tims_frontier_specification.md",
  "resolved_config_path": "string",
  "exploration_method": "optuna",
  "primary_endpoint": {
    "endpoint_ref": "comparison.primary_endpoint",
    "frontier_objectives": ["MC", "CQ"],
    "directions": {
      "MC": "maximize",
      "CQ": "maximize"
    }
  },
  "families": ["uniform", "heterogeneous"],
  "trial_counts": {
    "uniform": {
      "attempted": 0,
      "completed": 0,
      "failed": 0,
      "pruned": 0
    },
    "heterogeneous": {
      "attempted": 0,
      "completed": 0,
      "failed": 0,
      "pruned": 0
    }
  },
  "artifacts": {
    "trial_results_table": "string",
    "frontier_points_table": "string",
    "failure_log": "string"
  },
  "provenance": {
    "git_commit": "string",
    "git_branch": "string",
    "python_version": "string",
    "hostname": "string"
  },
  "started_at_utc": "string",
  "finished_at_utc": "string"
}
```

## 7. failure_log.jsonl

Semantics:

- one JSON object per failure event
- stores failure detail beyond the summary fields in the trial table

Expected record shape:

```json
{
  "result_schema_version": "1.0",
  "study_id": "string",
  "run_id": "string",
  "trial_id": "string",
  "family": "string",
  "optuna_trial_number": 0,
  "status": "failed",
  "failure_code": "string",
  "exception_type": "string",
  "message": "string",
  "traceback_path": "string",
  "trial_seed": 0,
  "timestamp_utc": "string"
}
```

## 8. Current Non-Goals

This version intentionally does not include:

- git metadata in `trial_results_table.csv`
- free-form `morphology_params_json` fields
- `is_frontier_point` flags in raw tables
- fixed global enumeration of `failure_code`

## 9. Companion Examples

The following companion examples are provided with this schema:

- `reports/schemas/examples/trial_results_table_example.csv`
- `reports/schemas/examples/frontier_points_table_example.csv`
