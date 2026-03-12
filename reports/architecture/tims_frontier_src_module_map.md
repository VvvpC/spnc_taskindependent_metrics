# TIMs Frontier Source Module Map

This note maps the accepted workflow schemas onto the new source package
`src/tims_frontier/`.

## Package Layout

```text
src/tims_frontier/
  config/
    resolver.py
  construction/
    builders.py
  evaluation/
    tims.py
  exploration/
    optuna_adapter.py
  orchestration/
    runner.py
    run_manifest.py
    seeds.py
  storage/
    paths.py
    records.py
  analysis/
    frontier.py
  reporting/
    summary.py
```

## Schema-to-Module Mapping

- Specification document
  - `reports/specifications/tims_frontier_specification.md`
- Study config schema
  - `reports/schemas/tims_frontier_config_schema.md`
  - implemented by `src/tims_frontier/config/resolver.py`
- Resolved config schema
  - `reports/schemas/tims_frontier_resolved_config_schema.md`
  - implemented by `src/tims_frontier/config/resolver.py`
- Run manifest schema
  - `reports/schemas/tims_frontier_run_manifest_schema.md`
  - implemented by `src/tims_frontier/orchestration/run_manifest.py`
- Result schema
  - `reports/schemas/tims_frontier_result_schema.md`
  - implemented by:
    - `src/tims_frontier/storage/records.py`
    - `src/tims_frontier/analysis/frontier.py`
    - `src/tims_frontier/reporting/summary.py`

## Legacy Backend Adapters

- Uniform / heterogeneous TIM evaluation
  - adapter: `src/tims_frontier/evaluation/tims.py`
  - legacy backends:
    - `src/Morphology_Research/Reservoirs_morphology_evaluation.py`
    - `src/Morphology_Research/Reservoirs_morphology_creator.py`
    - `src/formal_Parameter_Dynamics_Preformance.py`
- Historical Optuna patterns
  - adapter boundary: `src/tims_frontier/exploration/optuna_adapter.py`
  - reference scripts:
    - `src/Optuna_TaskIndependent_Metrics/Optuna_CQ_MC_Pareto.py`

## Design Choice

The new package keeps schema-owned workflow logic inside `src/tims_frontier/`
and delays legacy imports until adapter boundaries. This avoids making the new
workflow package depend on fragile top-level legacy imports during simple config
or manifest operations.
