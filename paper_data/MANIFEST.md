# Curated paper data

Collected on 2026-09-14 by `scripts/collect_paper_data.py` from the local, untracked `results/` tree. Sizes are of the copied files.

## `policy_seed2/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `paper_ppo_forcefix_nominalgains_seed2_3072_20260911` | 7 | 1.7 |

## `command_fidelity_seed2/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `validation_v4_seed2_v030_p048_trot_20260911` | 21 | 5.6 |
| `validation_v4_seed2_v030_p048_walk_20260911` | 20 | 3.2 |

## `extended_grid_seed2/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `old_policy_walk_df_seed2_20260911` | 2 | 0.0 |
| `old_policy_walk_df_seed2_20260911/grid` | 6 | 2.2 |

## `energy_and_return_map_seed2/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `stability_v4_seed2_descriptive_20260911` | 9 | 2.4 |
| `stability_v5_estimator_pilot_seed1100000_20260911` | 10 | 0.0 |

## `adaptive_selector_seed3/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `adaptive_duty_seed3_3072_20260911` | 8 | 1.7 |
| `adaptive_duty_grid_seed3_20260911` | 5 | 4.4 |
| `adaptive_duty_selector_seed3_20260911` | 10 | 0.5 |
| `adaptive_duty_selector_validation_seed3_20260911` | 11 | 0.9 |

## `high_duty_walk_seed4/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `high_duty_walk_seed4_3072_20260912` | 7 | 1.7 |
| `high_duty_walk_grid_seed4_20260912` | 5 | 1.6 |
| `high_duty_walk_selector_seed4_20260912` | 9 | 0.1 |
| `high_duty_walk_selector_validation_seed4_20260912` | 12 | 0.7 |
| `high_duty_walk_pipeline_20260912.json` | 1 | 0.0 |

## `narrow_stance_lineage/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `narrow_beam_center_from_seed2_3072_20260912` | 9 | 3.2 |
| `narrow_beam_tight2_from_center_3072_20260912` | 9 | 3.2 |
| `narrow_beam_tight3_center10_from_tight2_3072_20260912` | 9 | 3.2 |
| `narrow_beam_tight3_perturb_from_tight3_3072_20260913` | 9 | 3.2 |
| `validation_narrow_center_v030_p048_trot_20260912` | 8 | 0.1 |
| `validation_narrow_tight2_v030_p048_trot_20260912` | 8 | 0.1 |

## `push_study/`

| Source in `results/` | Files | Size (MB) |
|---|---:|---:|
| `validation_tight3_nominal_20260913` | 8 | 0.1 |
| `validation_tight3_perturbed_20260913` | 9 | 0.1 |
| `validation_tight3perturb_nominal_20260913` | 8 | 0.1 |
| `validation_tight3perturb_perturbed_20260913` | 9 | 0.1 |
| `perturbation_results_20260913` | 8 | 0.2 |
| `perturbation_smoke_20260913` | 4 | 0.0 |
| `perturbation_pipeline_20260913.sh` | 1 | 0.0 |

## `docs/`

Protocol and plan documents (9 files, 0.1 MB): `experiment_summary.md`, `paper_claim_coverage.md`, `paper_aligned_evaluation.md`, `perturbation.md`, `adaptive_duty_selector_plan.md`, `high_duty_walk_plan.md`, `old_policy_surface_plan.md`, `deployment_dr_plan.md`, `policy_quad_sdk_port.md`

**Total copied: 40.4 MB.**

## Left in `results/` (not tracked)

| Pattern | Why |
|---|---|
| `results/**/*.log, results/**/events.out.tfevents.*` | training and pipeline console logs and TensorBoard event files (about 50 MB); the numbers the section uses are in the CSV/JSON tables |
| `results/*/**.npz` | raw per-condition trial archives (about 1.3 GB for the paper runs); per-trial CSVs are exported instead |
| `results/deployment_gain_sweep*` | sim-to-sim motor-gain sweep (4.4 GB); not used in the section |
| `results/deployment_dr_*` | deployment domain-randomization smoke and solver runs |
| `results/flat_train_42*, results/train_42, results/validation_r[2-9]_*` | development lineage R1-R9 superseded by the seed-2 policy |
| `results/validation_seed0_*, results/paper_ppo_seed0_*, results/paper_ppo_phasefix_*, results/validation_phasefix_*, results/validation_forcefix_*` | earlier seed-0/seed-1 pilots and the pre-audit seed-2 evaluation |
| `results/narrow_beam_from_seed2*, results/narrow_beam_hdg5*, results/narrow_beam_tight_from_seed2*` | abandoned branches of the narrow-stance lineage |
| `results/*video*, results/*.mp4, results/port_*, results/sim2sim_*` | videos and ONNX exports for the hardware port |
| `results/benchmark_*, results/*smoke*, results/preflight, results/diagnostic_*` | throughput benchmarks and smoke tests |

Regenerate the section figures from the tracked tables with `python scripts/make_rl_section_figures.py --from-csv`.
