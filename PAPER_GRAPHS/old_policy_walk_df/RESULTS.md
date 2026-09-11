# Completed original-policy evaluation

The exact old seed-2 paper checkpoint completed 120 conditions and 3,840 trials.
The six-condition recorder smoke passed, all archive hashes were verified, and
all trial endpoints were recomputed from raw data before plotting. Five new
numerical/chronology tests pass. All three independent pre-run reviewers cleared
the execution; the science reviewer also independently audited the final data.

| Gait | Commanded DF | Mean achieved DF | Compliant trials | Periodic trials |
|---|---:|---:|---:|---:|
| Trot | 0.50 | 0.500 | 591/640 | 356/640 |
| Trot | 0.625 | 0.625 | 640/640 | 640/640 |
| Trot | 0.75 | 0.750 | 640/640 | 640/640 |
| Walk | 0.50 | 0.662 | 0/640 | 161/640 |
| Walk | 0.625 | 0.697 | 0/640 | 457/640 |
| Walk | 0.75 | 0.750 | 640/640 | 640/640 |

Walk below DF 0.75 was outside this policy's training support. Every such trial
failed command compliance. One physical failure occurred at walk DF 0.50,
speed 0.40 m/s, width 0.10 m, seed 1310016. Periodicity describes realized motion,
which can differ from the requested gait. It does not estimate chi or establish
the paper's convergence relationship.

64/120 physical cells passed the energy gate. The valid walking CoT comparison
still contains only DF 0.75, so its upper-panel trace must remain a line. The
lower panel now has measured walking periodicity across duty factors, explicitly
labeled as realized motion. No missing energy measurement was interpolated.

Artifacts: `figure_7_paper_style_rl_surfaces.png/.pdf`,
`rl_surfaces_by_width.png/.pdf`, and `walking_df_command_fidelity.png/.pdf`.
The original figures in the parent directory are preserved. All raw measurements
are in `results/old_policy_walk_df_seed2_20260911/grid/` in the repository.

After simulator completion, a CPU-only plot revision clarified the realized-motion
label and connected adjacent valid speed vertices; data and acceptance gates
were unchanged. The evaluation's original source snapshot remains archived, and
the final plot-script hash is recorded in `provenance.json`.
