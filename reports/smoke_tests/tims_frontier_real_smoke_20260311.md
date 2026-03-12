# TIMs Frontier Real Smoke Test Record

## Purpose

This smoke test was used to verify that the TIMs frontier workflow can run end to end with:

- real study script entrypoint
- real reservoir builders
- real TIM evaluators
- real result persistence

This run is an engineering validation artifact, not a scientific comparison.

## Run Identity

- `study_id`: `uniform_vs_heterogeneous_tims_frontier_real_smoke_v1`
- `run_id`: `real_smoke_test_20260311_runtimefix`
- `branch`: `codex/heterogeneous-vs-uniform-tims-broadening`
- `git_commit`: `79bf8a0eef94b34fec1ad1f784475b61855371bb`
- `python_version`: `3.11.3`
- `hostname`: `DESKTOP-BHLB2ME`

## Input Config

- Study config: [real_smoke.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/real_smoke.yaml)
- Script entrypoint: [run_tims_frontier_study.py](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/scripts/run_tims_frontier_study.py)

## Runtime Window

- `started_at_utc`: `2026-03-11T22:25:37Z`
- `finished_at_utc`: `2026-03-11T22:29:24Z`
- Wall-clock duration: about `3m47s`

## Trial Budget

- `uniform`: `1` attempted, `1` completed, `0` failed, `0` pruned
- `heterogeneous`: `1` attempted, `1` completed, `0` failed, `0` pruned

## Trial Outputs

| family | trial_id | MC | KR | GR | CQ | runtime_seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| uniform | `uniform_trial_0001` | 5.552975752956628 | 79.0 | 4.0 | 75.0 | 95.0524080999894 |
| heterogeneous | `heterogeneous_trial_0001` | 4.962004338196499 | 69.0 | 5.0 | 64.0 | 131.10969990000012 |

## Frontier Output

The processed frontier table contains one nondominated point per family for the `MC-CQ` primary endpoint.

- `uniform` frontier point: `MC=5.552975752956628`, `CQ=75.0`
- `heterogeneous` frontier point: `MC=4.962004338196499`, `CQ=64.0`

## Artifacts

- Trial table: [trial_results_table.csv](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_real_smoke_v1/real_smoke_test_20260311_runtimefix/trial_results_table.csv)
- Frontier table: [frontier_points_table.csv](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/processed/uniform_vs_heterogeneous_tims_frontier_real_smoke_v1/frontier_points_table.csv)
- Run summary: [run_summary.json](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_real_smoke_v1/real_smoke_test_20260311_runtimefix/run_summary.json)
- Run manifest: [run_manifest.json](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_real_smoke_v1/real_smoke_test_20260311_runtimefix/run_manifest.json)
- Resolved config: [resolved_config.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_real_smoke_v1/real_smoke_test_20260311_runtimefix/resolved_config.yaml)
- Failure log: [failure_log.jsonl](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/results/raw/uniform_vs_heterogeneous_tims_frontier_real_smoke_v1/real_smoke_test_20260311_runtimefix/failure_log.jsonl)

## Validation Outcome

The workflow passed the intended smoke-test scope:

- script entrypoint executed successfully
- both families completed one real trial
- all canonical artifacts were created
- `runtime_seconds` is now persisted in `trial_results_table.csv`
- no runtime failure or persistence inconsistency was observed

## Interpretation Boundary

This record should not be used to draw any scientific conclusion about uniform vs heterogeneous reservoirs.
It only shows that the current workflow can execute a tiny real run and write schema-aligned outputs.
