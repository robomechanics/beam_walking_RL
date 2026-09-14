# Paper claim coverage after the V4 measurement audit

## Current controller and usable evidence

The controller is the fresh seed-2 checkpoint
`results/paper_ppo_forcefix_nominalgains_seed2_3072_20260911/model_1799.pt`.
It runs on flat ground with forward commands only, nominal Kp/Kd gains, and no
map, terrain feature, beam, or push. Its command inputs are speed, period, duty
factor, full body-frame stance width, and trot/walk gait.

The corrected V4 nominal evaluation completed 64 held-out rollouts in each of
20 cells at 0.30 m/s and period 0.48 s. All 1,280 rollouts reached the endpoint,
and all 20/20 cells passed the operational 5 N contact, command, stance-width,
and straightness screen. The policy therefore executes the factors needed by
the paper comparison, including 0.10 m tucked-leg locomotion.

The first 80-cell V4 stability/CoT collection completed, but its return-map
finite differences are invalid. It built each +/- and zero clone around that
clone's independently drifted settled state. Although command fidelity passed
every reference, no stability condition passed the numerical validity gates.
The raw chi values are suppressed and must not be interpreted as evidence about
the controller. V4 archives are permanently ineligible for paper-claim release.

The mechanical-CoT trace is independent of the stencil defect. It produced
65/80 valid cells, below the 90% coverage gate. All cells at DF>=0.625 and all
walk cells were valid, while only 5/20 DF-0.50 trot cells were valid. This
factor-dependent missingness makes correlation estimates selection-biased.

## Paper-aligned claim table

| Paper result | Evaluation evidence | Current status | Required next result |
|---|---|---|---|
| Higher duty factor improves convergence (lower largest singular value, chi). | V4 nominal confirms accurate DF commands. In the V4 full grid, DF-0.50 trot reached a phase-one periodic orbit in 61.25% of references versus 100% for the other gait/DF strata. | **Not established.** The periodicity pattern points in the paper's direction but is not valid chi evidence. | Pass the V5 numerical pilot, freeze valid perturbation radii, and rerun matched DF-0.75 versus DF-0.50 return maps. Diagnose whether low-DF cells need longer settling or are multi-period. |
| Narrower stance generally worsens convergence. | V4 nominal proves the policy reaches 0.10--0.50 m full stance widths while going straight. | **Not established.** No valid chi-by-width result exists. | Measure V5 chi across all five widths; the primary local contrast is 0.10 versus 0.30 m. |
| Speed has a weak relationship with convergence. | The completed full collection covered 0.25, 0.30, 0.35, and 0.40 m/s, but its chi is invalid. | **Not established.** This range is only a local low-speed analogue of the paper's 0.25--2.5 m/s sweep. | Measure the V5 0.40-versus-0.25 chi ratio under the frozen equivalence margin. |
| Matched walk and trot have similar convergence. | V4 nominal confirms both two-beat trot and four-beat walk execute at DF 0.75. | **Not established.** The V4 return maps cannot support the comparison. | Measure matched V5 walk/trot chi at DF 0.75. |
| Duty factor and speed affect efficiency; gait is weak. | Positive mechanical CoT was collected, but only 65/80 cells passed and failures concentrate at DF-0.50 trot. | **Suppressed.** Factor-dependent missing cells block an unbiased relationship. | Resolve low-DF phase-one periodicity, then recollect a complete CoT grid and run matched contrasts. |
| Width has little efficiency effect. | Width-conditioned CoT samples exist only inside the incomplete 65-cell valid subset. | **Suppressed.** | Obtain complete matched 0.10-versus-0.30 m CoT pairs under the frozen validity gate. |
| Exact energy includes positive mechanical and Ohmic loss. | This evaluator measures positive mechanical CoT from clipped torque and joint velocity at 200 Hz. | **Proxy only.** No audited motor resistance/loss coefficient exists in the stock model. | Report positive mechanical CoT without claiming numerical electromechanical replication. |
| External-impulse robustness and narrow-beam traversal. | No evaluation uses a push, beam, map, or terrain feature. | **Outside scope.** | None under the requested straight flat-ground scope. |

## V5 correction and next run

V5 forks every baseline, zero clone, and +/- member from one exact phase-zero
reference state, synchronizes its contact caches and previous action, archives
the requested stencil and immediate simulator readback, and aborts above the
unchanged 0.001 normalized write tolerance. Pre-restore clone drift is retained
as a diagnostic. Zero-clone one-cycle divergence continues to gate hidden
solver-history noise. The former pre-restore-spread threshold is a V4-only gate;
V5 replaces it with the direct post-fork readback and zero-clone checks.

Before another full grid, the guarded development pilot uses one reference and
one condition at seed 1,100,000 with h=0.005, 0.01, 0.025, 0.05, and 0.10. It
reports every adjacent-radius matrix and chi difference. The extra radii receive
no frozen-gate pass. If the original trio is unsuitable, the diagnostic may
inform a separately versioned and preregistered protocol before the untouched
seed block beginning at 2,000,000 is used. V5 claim eligibility is
machine-disabled. Any corrected seed-2 result remains descriptive; paper
inference still requires five fresh, independently trained PPO policies.

## Artifacts

- Combined labeled flat-ground video:
  `results/paper_eval_videos_seed2_20260911/paper_eval_comparison_seed2.mp4`
- V4 nominal trot:
  `results/validation_v4_seed2_v030_p048_trot_20260911`
- V4 nominal walk:
  `results/validation_v4_seed2_v030_p048_walk_20260911`
- Audited failed V4 stability/CoT measurement:
  `results/stability_v4_seed2_descriptive_20260911`

The videos and nominal tables establish straight command execution. They do not
establish the paper's chi correlations.

## V5 pilot result

The five-radius V5 development pilot fixed the exposed stencil geometry but
failed reproducibility. Identical zero clones separated by 0.01598 normalized
units after one period. Noise divided by h ranged from 3.196 at h=0.005 to 0.160
at h=0.10, always above 0.05, and h=0.10 changed hybrid topology. Translation
symmetry and adjacent-h matrix agreement also failed. No pilot chi is usable.

The next reviewed diagnostic uses exactly 64 identical clones with evaluation-
only PhysX enhanced determinism. It is development-only, emits no chi or CoT,
and cannot authorize a full grid. Its endpoint limit is 0.00125 normalized units,
derived from the existing 0.05 noise fraction at h=0.025.

## Publication figure set

The audited results are summarized in `PAPER_GRAPHS`. Each figure is available as a PNG for
slides and a vector PDF for the manuscript, with the plotted CSV data beside it.

| Figure | Use in the paper | Interpretation |
|---|---|---|
| `figure_1_command_fidelity` | Treatment and controller validation | Direct evidence that the policy realizes 0.10--0.50 m body-frame stance width and the commanded duty factors while moving straight at the core speed and period. |
| `figure_2_periodicity_diagnostic` | Motivation for the remaining low-DF work | Phase-one periodicity is 61.25% for trot DF 0.50 and 100% for the other gait/DF strata. This is directionally consistent with the duty-factor claim but is not chi evidence. |
| `figure_3_energy_validity_coverage` | Missing-data audit | The 65/80 valid mechanical-CoT cells are visibly concentrated away from low-DF trot, showing why efficiency correlations are suppressed. |
| `figure_4_return_map_validity` | Estimator audit | Every tested finite-difference radius fails the frozen zero-clone noise and translation checks, so no raw chi result is plotted. |
| `figure_5_paper_claim_status` | Overview slide or results roadmap | Maps every target paper claim to supported, diagnostic-only, suppressed, or missing evidence. |
| `figure_6_speed_df_cot_periodicity_3d` | Three-axis factor-space view | Shows descriptive mechanical CoT and phase-one periodicity over speed and duty factor. The right panel is a convergence prerequisite, not a valid chi measurement. |

The command-fidelity figure is affirmative controller evidence. The periodicity
figure is diagnostic evidence. The energy and return-map figures document why
the corresponding paper correlations cannot yet be claimed.
