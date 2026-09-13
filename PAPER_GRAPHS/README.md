# Paper-claim figure set

These figures are generated from the audited seed-2 flat-ground evaluations.
All PNG files are directly in this folder; the matching PDFs are vector exports.

| Figure | What it establishes | Scientific limit |
|---|---|---|
| `figure_1_command_fidelity` | The policy realizes commanded stance width and duty factor while tracking 0.30 m/s and staying straight. All 20/20 cells passed (1280 rollouts). | This validates the treatment variables; it is not a convergence result. |
| `figure_2_periodicity_diagnostic` | Low-DF trot is the only stratum with reduced phase-one periodicity (61.25% versus 100%). | Directionally agrees with the paper's duty-factor claim, but periodicity is not chi. |
| `figure_3_energy_validity_coverage` | Shows exactly where mechanical-CoT data pass the validity gate. | Only 65/80 cells pass and missingness is factor dependent, so efficiency correlations are suppressed. |
| `figure_4_return_map_validity` | Explains why the return-map result is withheld: zero-clone noise and translation symmetry fail at every tested radius. | 0 valid chi estimates; raw values must not be shown as policy evidence. |
| `figure_5_paper_claim_status` | Slide-ready map from each paper claim to the evidence currently available. | A single trained seed cannot establish population-level paper replication. |
| `figure_6_speed_df_cot_periodicity_3d` | Three-axis view of speed, duty factor, mechanical CoT, and the periodicity prerequisite. | CoT is descriptive with factor-dependent missingness; periodicity is not chi. |
| `figure_7_paper_style_rl_surfaces` | Paper-style response-surface layout using the RL measurements. | Trot forms a measured surface; walk has only one duty-factor row, and periodicity replaces unavailable chi only as a labeled diagnostic. |

## Direct quantitative result

- Maximum stance-width error across cell means: 0.0174 m.
- Maximum duty-factor error across cell means: 0.0007.
- Maximum speed error from 0.30 m/s: 0.0120 m/s.
- Maximum lateral RMSE: 0.0294 m, against the 0.10 m limit.
- Maximum heading RMSE: 0.0389 rad, against the 0.10 rad limit.

## Source data

- `results/validation_v4_seed2_v030_p048_trot_20260911/summary.csv`
- `results/validation_v4_seed2_v030_p048_walk_20260911/summary.csv`
- `results/stability_v4_seed2_descriptive_20260911/stability_references.csv`
- `results/stability_v4_seed2_descriptive_20260911/paper_energy_conditions.csv`
- `results/stability_v5_estimator_pilot_seed1100000_20260911/stability_references.csv`

Regenerate with `python scripts/make_paper_figures.py`. No simulator or GPU is used.
# Additional original-policy walking DF evaluation

The completed 120-cell, 3,840-trial extension is in
[old_policy_walk_df/RESULTS.md](old_policy_walk_df/RESULTS.md), with
[updated surfaces](old_policy_walk_df/figure_7_paper_style_rl_surfaces.png) and
[command-fidelity measurements](old_policy_walk_df/walking_df_command_fidelity.png).
It uses the same original seed-2 checkpoint. Walk DF 0.50/0.625 failed all command
gates, so valid walking CoT remains limited to DF 0.75. The new lower surface
shows periodicity of realized motion, including incorrectly executed commands.
The historical figures below remain unchanged.

## High-duty walk selector

The separate seed-4 high-duty walking-policy figures are in
[`high_duty_walk_selector/`](high_duty_walk_selector/). The final consolidated
figure shows that DF 0.80 was selected at every tested width and speed, with
median achieved DF 0.8229 and 100% fresh-validation compliance. This extension
is kept separate from the audited seed-2 paper-claim figure set above.
