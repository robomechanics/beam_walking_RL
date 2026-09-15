# Specialist training and evaluation pipeline

This is the single, current description of how the RL policies for the paper
are trained and evaluated. It is updated in place when the pipeline changes.
The runnable version is `scripts/pipelines/specialist_pipeline.sh`.

## What it produces

One trot and one walk policy for the Unitree Go2 on flat ground, each commanded
by forward speed, duty factor, full stance width, gait period and gait. Each
policy is evaluated for command fidelity, periodicity, energetic cost and
robustness across the full commanded range. A duty-factor selector network,
fit on success under disturbance, chooses the duty factor for every speed,
period and stance width.

## Commanded domain

| Command | Trot | Walk |
|---|---|---|
| Stance width | 0.05–0.30 m, anchors every 0.05 m | same |
| Forward speed | 0.25–0.40 m/s, anchors every 0.05 m/s | same |
| Gait period | 0.36–0.54 s | 0.40–0.54 s |
| Duty factor, training anchors | 0.50, 0.625, 0.75, with duty drawn continuously over 0.50–0.75 | 0.75, 0.80, 0.85, 0.90, with duty drawn continuously over 0.75–0.90 |
| Duty factor, evaluated | 0.50, 0.55, 0.60, 0.65, 0.70, 0.75 | 0.75, 0.80, 0.85, 0.90 |
| Minimum swing | 5 control ticks (0.10 s) | 2 control ticks (0.04 s) |

The minimum swing sets which duty factors exist at each period. Trot cannot
command 0.75 below 0.40 s, and walk starts at 0.40 s so that 0.90 keeps two
swing ticks. Commands change only at gait-cycle boundaries.

## Stages

| # | Stage | Script | Key settings |
|---|---|---|---|
| 1 | Training | `scripts/narrow_specialist_experiment.py train --gait G` | 3,072 envs, 1,800 PPO updates, seed 5, 10% grounded starts |
| 2 | Command fidelity | `scripts/evaluate_policy.py --task narrow_specialist` | trot 0.36/0.40/0.48/0.54 s, walk 0.40/0.48/0.54 s, all widths and evaluated duty factors, 64 held-out trials, 0.30 m/s |
| 3 | Duty-contrast check | `scripts/narrow_fidelity_gate.py --criterion contrast` | recorded for every fidelity run |
| 4 | Robustness | `scripts/evaluate_policy.py --task narrow_specialist --perturbation` | every speed (0.25/0.30/0.35/0.40 m/s) and every period of stage 2, all widths and evaluated duty factors, 64 held-out trials under random base disturbances up to 25 N |
| 4b | Robustness, stronger disturbance | same, `--perturbation_max_force 50 --perturbation_max_torque 6` | for a gait whose 0.20–0.30 m cells all reach 95% success at 25 N and 0.30 m/s, its 0.30 m/s runs repeated with disturbances up to 50 N |
| 5 | Period sweep | `scripts/collect_policy_surfaces.py --task narrow_specialist --period-sweep` | trot 0.36/0.40/0.48/0.54 s, walk 0.40/0.48/0.54 s, 4 speeds, 6 widths, feasible evaluated duty factors, 32 matched trials |
| 6 | Selector network | `scripts/fit_robust_duty_selector.py`; `scripts/evaluate_policy.py --perturbation --split test` on the selections; `scripts/validate_robust_duty_selector.py` | labels from stage 4, fit, fresh-seed validation under the same disturbances, promotion |
| 7 | Figures | `scripts/make_narrow_paper_figures.py` | five paper figures, each with its data table (below) |
| 8 | Export | shell step | per-trial CSVs, robustness summaries, policy provenance, duty-contrast report, selector fit, validation and promotion, and the paper figures into `paper_data/` |

Trainings run one at a time. Stages 2 and 5 run trot and walk side by side, and
the runs of stage 4 and of the selector validation share two workers. Each
evaluation is an independent process with its own seeds, so this changes only
wall-clock time.

### Paper figures

| Figure | Content |
|---|---|
| 1 | Periodicity (cycle RMS, median and interquartile band) against realized duty factor, trot and walk, one panel per stance width |
| 2 | Positive mechanical CoT against realized duty factor, trot and walk, one panel per stance width |
| 3 | Robustness: success rate against stance width at 0.30 m/s, one line per commanded duty factor, per gait |
| 4 | Realized duty factor by stance width and commanded duty factor, per gait |
| 5 | Duty factor chosen by the selector network against stance width, trot and walk: mean and interquartile range over the network's speed and period contexts |

Every figure pools the periods at which every duty factor of the gait was run.
CoT uses every trial that completed without a failure and with a finite positive
value.

### Training details

**Training budget per gait.** One PPO update is 3,072 simulated robots each
taking 48 control steps, 147,456 control steps in total, about 49 minutes of
simulated robot time. Training runs 1,800 updates: 265.4 million control steps,
about 1,475 hours of simulated robot time, 29 min per gait on one RTX 5070 Ti.

- **PPO.** 48-step rollouts, 5 epochs, 4 minibatches, learning rate 1e-3
  (adaptive), γ 0.995, λ 0.95, clip 0.2, entropy 0.001. Actor and critic MLPs
  (256, 128, 128), ELU.
- **Curriculum.** First 50 updates at speed 0.30 m/s, width 0.15 m, period
  0.48 s with duty anchors balanced. Updates 50–150 balance speed and width
  anchors. After update 150, 75% of commands stay on anchors and 25% are drawn
  continuously over speed, width, period and duty, capped by the swing bound.
- **Plant.** Stock Go2 PD gains (Kp 25, Kd 0.5), 50 Hz control, 200 Hz
  physics, **self-collision on**, no domain randomization.
- **Placement reward.** 3 cm lateral tolerance. Width observations are
  normalized over 0.10–0.50 m.

### Robustness disturbances

Random world-frame base force up to 25 N and torque up to 3 N m, held
0.10–0.40 s at a time, ramping from 10% to 100% of that bound over the first
10 s of each episode. The stronger level doubles both bounds.

### Selector network

**Labels.** A context is a gait, speed, period and stance width of the stage 4
runs. Its label is the lowest duty factor whose success rate under disturbance
is at least 90% of the best success rate any duty factor reaches in that
context. A context where no duty factor succeeds has no label, and the selector
abstains there. Each label carries a 95% interval from 100,000 binomial
resamples of the context's trials.

**Network.** A classifier over the evaluated duty factors with inputs stance
width, speed, period and gait, two hidden layers of 32 units (ELU). Its output
is masked to the gait's duty factors that are feasible at the period, and it
must reproduce every label exactly.

**Validation.** Every selection is run again under the same disturbances on the
held-out test split, 64 trials per context. A context fails when its fresh
success rate is significantly below 90% of the best success rate in its
labelling runs: one-sided exact binomial test at a family-wise 5%, Bonferroni
corrected over contexts. The selector is promoted when no context fails.

Taking the lowest duty factor that keeps nearly all of the achievable robustness
selects the least conservative gait that is still robust. The share is taken of
the best rate in each context, not a fixed success level, so contexts where
every duty factor succeeds only rarely still get a label. Energetic cost does
not enter the selection.

## Design decisions

- **No domain randomization.** Randomized actuator delay (up to 20 ms) and motor
  strength (down to 70%) make the policy touch down one to three control ticks
  early, which compresses the realized duty factor toward the middle of the
  range.
- **Self-collision on.** At 0.05 m the feet are millimetres apart. Without
  collision the legs could pass through each other and flatter narrow stances.
- **No disturbances during training.** The policies are trained without
  external disturbances, so each leg executes the commanded contact schedule;
  disturbances enter only in the robustness evaluation.
- **Duty-contrast check, not exact tracking.** The check records whether
  realized duty rises with the command and keeps at least 60% of the commanded
  spread at every period. Analyses use the **realized** duty factor and stance
  width, which every table includes.
- **Robustness is measured under disturbance.** Without disturbance every
  width and duty factor completes its trials, so success cannot separate duty
  factors. Random base disturbances make success report robustness against duty
  factor for each width, speed and period. Where the wider stances reach full
  success at every duty factor, a second level with twice the force and torque
  keeps them informative.
- **The selector is chosen by robustness.** Its labels and its validation both
  come from success under the same disturbances, over the full speed, period and
  width grid the network covers.
- **Duty factors every 0.05.** Trot and walk are evaluated on the same duty
  step, so trends and selector choices have the same resolution for both gaits.
- **No single fixed period.** Every stage spans the gait's periods. The period
  sweep already contains the 0.48 s slice, so no separate fixed-period grid is
  collected.

## Analyzing duty factor

- Use the realized duty factor, `achieved_df`, as a continuous variable
  instead of the command labels. Discrete swing ticks round the commanded
  schedule, so realized levels are unevenly spaced and change with period.
- Compare trot and walk at matched realized duty factor, around 0.75 where the
  two ranges meet, as Investigation 1 does.
- Before claiming a difference between neighbouring duty factors, check that
  their `achieved_df` distributions actually separate.

## Outputs

| Location | Content |
|---|---|
| `results/narrow_specialist_{trot,walk}_seed5_3072_<tag>/` | checkpoints used for every evaluation |
| `results/validation_narrow_<gait>_v030_p<period>_<tag>/` | fidelity archives, tables, plots |
| `results/<tag>_fidelity_gate.json` | duty-contrast report, including where in the gait phase contact deviates |
| `results/push_robustness_narrow_<gait>_v<speed>_p<period>[_f50]_<tag>/` | disturbance-trial archives and `perturbation_summary.json`; speed written as `v025`–`v040`; 25 N and 50 N levels |
| `results/narrow_surfaces_sweep_<gait>_<tag>/grid/` | period-sweep archives and `surface_trials.csv` |
| `results/narrow_robust_selector_<tag>/` | selector fit: success of every candidate, labels with intervals, contexts without success, predictions, checkpoint |
| `results/narrow_robust_selector_validation_<gait>_v<speed>_p<period>_<tag>/` | fresh-seed disturbance runs of the selections |
| `results/narrow_robust_selector_promoted_<tag>/` | per-context validation summary, report, validated checkpoint |
| `PAPER_GRAPHS/narrow_specialist/paper_figures/` | the five paper figures with their tables |
| `paper_data/specialist_<tag>/` | shareable export: per-trial CSVs, robustness summaries (`trials/robustness_<gait>_v<speed>_p<period>.json`), selector validation runs (`trials/selector_validation_<gait>_v<speed>_p<period>.json`), `selector_fit/`, `selector_promoted/`, policy provenance, duty-contrast report, paper figures |
| `results/<tag>.status` | one line per finished stage |

`results/` and `PAPER_GRAPHS/` are not tracked. `paper_data/` is.

In every per-trial CSV, one row is one rollout. Columns include the commands
(`gait, period, speed, step_width, command_df`), the realized response
(`achieved_df, achieved_stance_width, forward_speed`,
`positive_mechanical_cot`), and quality flags
(`any_failure, compliant, energy_trial_valid`).

## Running

```
setsid nohup bash scripts/pipelines/specialist_pipeline.sh > results/pipeline.log 2>&1 &
tail -f results/narrow_20260914.status
```

Set `TAG` to start a separate run, `PYTHON_BIN` if the Isaac Lab Python is
elsewhere, and `REPO_ROOT` when running a copy of the script from another
directory. The script is resumable: completed stages are skipped, and a
training stage without its final checkpoint is retrained from the start. A new
selector fit discards the previous selector validation runs.

Evaluation accepts a checkpoint only if the training source files
(`NARROW_TRAINING_SOURCE_FILES` in `narrow_specialist.py`) still match the hash
recorded at training time, so leave them unchanged while a run is in progress.

## Code map

| File | Role |
|---|---|
| `source/beam_walking/beam_walking/experiment/narrow_specialist.py` | command ranges, training anchors, sampler, lineage |
| `source/beam_walking/beam_walking/experiment/narrow_specialist_task.py` | environment config: self-collision, placement tolerance |
| `scripts/narrow_specialist_surface_data.py` | sweep definitions, evaluated duty factors, checkpoint checks for the collector |
| `source/beam_walking/beam_walking/experiment/unified_duty_selector.py` | selector classifier, candidate levels, feasibility mask |
| `scripts/fit_robust_duty_selector.py` | selector labels and intervals from the robustness runs, fit |
| `scripts/validate_robust_duty_selector.py` | validation run plan, fresh-seed test, promotion |
| `scripts/make_narrow_paper_figures.py` | the five paper figures and their tables |

## Limitations

- Stances near 0.05 m rely on simulated self-collision, which has not been
  validated against hardware.
- Energy is positive mechanical joint work per distance. It omits motor
  heating, so it understates the cost of high-torque postures such as wide stance.
- Fidelity runs use 0.30 m/s. The robustness runs and the period sweep cover all
  four speeds.
- Results are simulation on flat ground with commanded stance width; there is no
  physical narrow support.
