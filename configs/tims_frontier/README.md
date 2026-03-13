# TIMs Frontier Config Roles

This directory contains the workflow configs for the uniform-vs-heterogeneous TIMs frontier study.

## Current Roles

- [real_smoke.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/real_smoke.yaml)
  - Minimal real-chain smoke test
  - Purpose: verify that the full workflow still runs end to end
  - Current budget: `1` trial per family

- [small_dev.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/small_dev.yaml)
  - Current canonical development baseline
  - Purpose: daily development, regression checks, analysis/reporting iteration
  - Current budget: `5` trials per family

- [fixed_geometry_beta20.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/fixed_geometry_beta20.yaml)
  - Fixed-parameter controlled geometry comparison
  - Purpose: compare uniform vs heterogeneous under identical non-geometric parameters
  - Fixed settings: `beta_prime = 20`, `scheme = random`, `n_instances = 5`, `beta_spread = 3.0`, fixed morphology seed
  - Current budget: `100` matched repeated trials per family

- [fixed_geometry_beta20_v2.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/fixed_geometry_beta20_v2.yaml)
  - Fixed-parameter morphology-seed sweep meta-study
  - Purpose: repeat the same uniform-vs-heterogeneous comparison across `5` heterogeneous realizations
  - Fixed settings: same nominal `beta_prime = 20`, same `theta / gamma / m0 / Nvirt`, `scheme = random`, `n_instances = 5`, `beta_spread = 3.0`
  - Current budget: `100` matched repeated trials per family for each morphology seed

## Baseline Rule

Unless there is a strong reason not to, new workflow development should use [small_dev.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/small_dev.yaml) as the default runnable config.

If this development baseline changes in the future:

1. Update the header comment in [small_dev.yaml](/Users/Chen/Desktop/Repository/spnc_taskindependent_metrics_tims_broadening/configs/tims_frontier/small_dev.yaml), or replace it with a new canonical file.
2. Update this README to reflect the new baseline.
3. Record at least one successful real run against the new baseline before treating it as stable.
