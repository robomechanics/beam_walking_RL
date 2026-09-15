# Specialist training and evaluation pipeline

This is the single, current description of how the RL policies for the paper
are trained and evaluated. It is updated in place when the pipeline changes.
The runnable version is `scripts/pipelines/specialist_pipeline.sh`.

## What it produces

One trot and one walk policy for the Unitree Go2 on flat ground, each commanded
by forward speed, duty factor, full stance width, gait period and gait. The
policies reported in the paper are **push-trained**: a clean policy is trained
first, then fine-tuned with random pushes so it does not exploit the noise-free
simulator. Every policy is evaluated for command fidelity, energetic cost and
periodicity across the full commanded range, and a duty-factor selector is fit
on those measurements.

## Commanded domain

| Command | Trot | Walk |
|---|---|---|
| Stance width | 0.05–0.30 m, anchors every 0.05 m | same |
| Forward speed | 0.25–0.40 m/s, anchors every 0.05 m/s | same |
| Gait period | 0.36–0.54 s | 0.40–0.54 s |
| Duty factor anchors | 0.50, 0.625, 0.75 | 0.75, 0.80, 0.85, 0.90 |
| Minimum swing | 5 control ticks (0.10 s) | 2 control ticks (0.04 s) |

The minimum swing sets which duty factors exist at each period. Trot cannot
command 0.75 below 0.40 s, and walk starts at 0.40 s so that 0.90 keeps two
swing ticks. Commands change only at gait-cycle boundaries.

## Stages

| # | Stage | Script | Key settings |
|---|---|---|---|
| 1 | Clean training | `scripts/narrow_specialist_experiment.py train --gait G` | 3,072 envs, 1,800 PPO updates, seed 5, 10% grounded starts |
| 2 | Push fine-tune | `scripts/narrow_specialist_push_experiment.py train --gait G --perturbation --initialize_from <clean>` | warm start, 1,800 updates, seed 3 |
| 3 | Command fidelity | `scripts/evaluate_policy.py --task narrow_specialist` | every period of the gait, all widths and duty anchors, 64 held-out trials, 0.30 m/s |
| 4 | Duty-contrast gate | `scripts/narrow_fidelity_gate.py --criterion contrast` | stops the pipeline if the duty contrast collapses |
| 5 | Period sweep | `scripts/collect_policy_surfaces.py --task narrow_specialist --period-sweep` | trot 0.36/0.40/0.48/0.54 s, walk 0.40/0.48/0.54 s, 4 speeds, 6 widths, feasible duties, 32 matched trials |
| 6 | Selector | `scripts/fit_unified_duty_selector.py`, then `collect_policy_surfaces.py --selector-checkpoint`, then `scripts/validate_unified_selector.py` | fit on both sweeps, fresh-seed validation per gait, promotion |
| 7 | Figures | `scripts/make_specialist_period_surfaces.py`, `scripts/make_specialist_trend_figures.py` | one panel or file per period |
| 8 | Export | shell step | CSVs, provenance and figures into `paper_data/` |

Stages run one simulator job at a time. Training waits for 14.8 GB of free
host memory, because the capacity check rejects a start below 14.3 GB.

### Training details

- **PPO.** 48-step rollouts, 5 epochs, 4 minibatches, learning rate 1e-3
  (adaptive), γ 0.995, λ 0.95, clip 0.2, entropy 0.001. Actor and critic MLPs
  (256, 128, 128), ELU.
- **Curriculum.** First 50 updates at speed 0.30 m/s, width 0.15 m, period
  0.48 s with duty anchors balanced. Updates 50–150 balance speed and width
  anchors. After update 150, 75% of commands stay on anchors and 25% are drawn
  continuously over speed, width, period and duty, capped by the swing bound.
- **Plant.** Stock Go2 PD gains (Kp 25, Kd 0.5), 50 Hz control, 200 Hz
  physics, **self-collision on**, no domain randomization.
- **Placement reward.** 3 cm lateral tolerance. Width observations keep the
  0.10–0.50 m normalization used by every earlier policy.
- **Pushes during fine-tuning.** Random world-frame base force up to 25 N and
  torque up to 3 N m, held 0.10–0.40 s at a time, ramping from 10% to 100% of
  that bound over the first 10 s of each episode.

### Selector

For each context of stance width, speed, period and gait, the label is the
candidate duty factor with the lowest median positive mechanical cost of
transport among candidates passing at least 90% of the compliance and
finite-energy checks. The classifier is masked to the gait's candidates that
are feasible at that period. A selector is promoted only if, on fresh seeds,
every context reaches at least 90% compliance and 90% finite-energy compliance,
with median realized duty within 0.05 of the selection.

The selector's width domain is read from the sweep, so it accepts 0.05 m. As a
side effect, selector checkpoints fitted before this change (0.10–0.50 m) load
only from the tag `pre-specialist-data`.

## Design decisions

- **No domain randomization.** Earlier policies trained with randomized
  actuator delay (up to 20 ms) and motor strength (down to 70%) learned to
  touch down one to three control ticks early. Commanded duty 0.50 then ran at
  0.58–0.67 and the duty contrast the paper depends on collapsed.
- **Self-collision on.** At 0.05 m the feet are millimetres apart. Without
  collision the legs could pass through each other and flatter narrow stances.
- **Push-trained policies are the reported controller.** This follows the
  mentor's direction to avoid behaviour that only works in a noise-free
  simulator. The cost is measurable: a push-trained trot policy realized duty
  0.56 for 0.50 commanded, and 0.145 m stance for 0.10 m commanded. Analyze
  results against **realized** duty factor and stance width, which every table
  includes.
- **Contrast gate, not exact tracking.** Because push training shifts duty
  somewhat by design, the gate requires realized duty to rise with the command
  and keep at least 60% of the commanded spread, at every period tested.
  Tested on existing data, it passes an earlier push-trained trot policy (75%)
  and rejects the collapsed narrow-stance policy (44%).
- **No push study.** Dropped at the mentor's direction. The paper's robustness
  claim rests on Investigations 1 and 3.
- **No single fixed period.** Every stage spans the gait's periods. The period
  sweep already contains the 0.48 s slice, so no separate fixed-period grid is
  collected.

## Outputs

| Location | Content |
|---|---|
| `results/narrow_specialist_{trot,walk}_seed5_3072_<tag>/` | clean checkpoints |
| `results/narrow_specialist_push_{trot,walk}_from_seed5_3072_<tag>/` | push-trained checkpoints |
| `results/validation_narrow_<gait>_v030_p<period>_<tag>/` | fidelity archives, tables, plots |
| `results/<tag>_fidelity_gate.json` | gate report, including where in the gait phase contact deviates |
| `results/narrow_surfaces_sweep_<gait>_<tag>/grid/` | period-sweep archives and `surface_trials.csv` |
| `results/narrow_selector_<tag>/`, `..._promoted_<tag>/` | selector fit and validated checkpoint |
| `PAPER_GRAPHS/narrow_specialist/` | figures |
| `paper_data/specialist_narrow_<tag>/` | shareable export: per-trial CSVs, policy provenance, selector tables, figures |
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

Set `TAG` to start a separate run, and `PYTHON_BIN` if the Isaac Lab Python is
elsewhere. The script is resumable: a stage whose output already has a
completion marker is skipped, so rerunning after an interruption continues
where it stopped.

## Code map

| File | Role |
|---|---|
| `source/beam_walking/beam_walking/experiment/narrow_specialist.py` | command ranges, anchors, sampler, lineage |
| `source/beam_walking/beam_walking/experiment/narrow_specialist_task.py` | environment config: self-collision, placement tolerance |
| `source/beam_walking/beam_walking/experiment/narrow_specialist_push.py` | push fine-tune identity and parent verification |
| `scripts/narrow_specialist_surface_data.py` | sweep definitions and checkpoint checks for the collector |
| `source/beam_walking/beam_walking/experiment/unified_duty_selector.py` | selector model, labels, validation evidence |
| `scripts/export_push_outcomes.py` | tidy push-outcome export, kept for earlier push data |

## Limitations

- Stances near 0.05 m rely on simulated self-collision, which has not been
  validated against hardware.
- Energy is positive mechanical joint work per distance. It omits motor
  heating, so it understates the cost of high-torque postures such as wide stance.
- Fidelity runs use 0.30 m/s. The period sweep covers the other speeds.
- Results are simulation on flat ground with commanded stance width; there is no
  physical narrow support.
