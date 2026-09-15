# Specialist training and evaluation pipeline

This is the single, current description of how the RL policies for the paper
are trained and evaluated. It is updated in place when the pipeline changes.
The runnable version is `scripts/pipelines/specialist_pipeline.sh`.

## What it produces

One trot and one walk policy for the Unitree Go2 on flat ground, each commanded
by forward speed, duty factor, full stance width, gait period and gait. The
policies reported in the paper are **push-trained**: a clean policy is trained
first, then fine-tuned with random pushes so it does not exploit the noise-free
simulator. Every policy is evaluated for command fidelity, energetic cost,
periodicity and success under pushes across the full commanded range, and a
duty-factor selector is fit on those measurements.

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
| 2 | Push fine-tune | `scripts/narrow_specialist_push_experiment.py train --gait G --perturbation --initialize_from <previous>` | three consecutive 1,200-update segments (3,600 updates), each warm-started from the previous checkpoint; seed 3 |
| 3 | Command fidelity | `scripts/evaluate_policy.py --task narrow_specialist` | every period of the gait, all widths and duty anchors, 64 held-out trials, 0.30 m/s |
| 4 | Duty-contrast gate | `scripts/narrow_fidelity_gate.py --criterion contrast` | stops the pipeline if the duty contrast collapses |
| 5 | Push robustness | `scripts/evaluate_policy.py --task narrow_specialist --perturbation` | every period of the gait, all widths and duty anchors, 64 held-out pushed trials, 0.30 m/s |
| 6 | Period sweep | `scripts/collect_policy_surfaces.py --task narrow_specialist --period-sweep` | trot 0.36/0.40/0.48/0.54 s, walk 0.40/0.48/0.54 s, 4 speeds, 6 widths, feasible duties, 32 matched trials |
| 7 | Selector | `scripts/fit_unified_duty_selector.py`, then `collect_policy_surfaces.py --selector-checkpoint`, then `scripts/validate_unified_selector.py` | fit on both sweeps, fresh-seed validation per gait, promotion |
| 8 | Figures | `scripts/make_specialist_period_surfaces.py`, `scripts/make_specialist_trend_figures.py`, `scripts/make_narrow_push_robustness_figures.py` | one panel or file per period |
| 9 | Export | shell step | CSVs, provenance and figures into `paper_data/` |

Clean trainings run one at a time. Push fine-tune segments and evaluations
(stages 3, 5, 6 and the selector validations) run trot and walk side by side.
Each is an independent process with its own seeds, so this changes only
wall-clock time.

### Training details

**Training budget per gait.** One PPO update is 3,072 simulated robots each
taking 48 control steps, 147,456 control steps in total, about 49 minutes of
simulated robot time.

| Phase | Updates | Control steps | Simulated robot time |
|---|---|---|---|
| Clean training | 1,800 | 265.4 million | 1,475 hours |
| Push fine-tune, 3 segments of 1,200 | 3,600 | 530.8 million | 2,949 hours |
| **Push-trained policy, total** | **5,400** | **796.3 million** | **4,424 hours** |

On one RTX 5070 Ti, clean training takes 29 min per gait, and each push segment
takes about 38 min for both gaits together, about 3 h of training in total.

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
- **Pushes during fine-tuning.** Random world-frame base force up to 25 N and
  torque up to 3 N m, held 0.10–0.40 s at a time, ramping from 10% to 100% of
  that bound over the first 10 s of each episode.
- **Fine-tune segments.** Segment 1 starts from the clean policy's weights with
  a new Adam optimizer. Segments 2 and 3 each continue the previous segment's
  weights, Adam state and adaptive learning rate. The curriculum step counter
  carries through, so fine-tuning stays on the full command distribution. Every
  segment uses seed 3. Update numbers restart in each segment's directory:
  segment 3's `model_1199.pt` is update 5,399 of the policy.

### Selector

For each context of stance width, speed, period and gait, the label is the
candidate duty factor with the lowest median positive mechanical cost of
transport among candidates passing at least 90% of the compliance and
finite-energy checks. The classifier is masked to the gait's candidates that
are feasible at that period. A selector is promoted only if, on fresh seeds,
every context reaches at least 90% compliance and 90% finite-energy compliance,
with median realized duty within 0.05 of the selection.

The selector's width domain is read from the sweep, so it covers 0.05–0.30 m.

## Design decisions

- **No domain randomization.** Earlier policies trained with randomized
  actuator delay (up to 20 ms) and motor strength (down to 70%) learned to
  touch down one to three control ticks early. Commanded duty 0.50 then ran at
  0.58–0.67 and the duty contrast the paper depends on collapsed.
- **Self-collision on.** At 0.05 m the feet are millimetres apart. Without
  collision the legs could pass through each other and flatter narrow stances.
- **3,600 fine-tune updates.** Narrow stance under pushes learns slowly. After
  2,400 updates reward was still rising for both gaits and trot's duty contrast
  at 0.48 and 0.54 s was below the gate, so the fine-tune runs for three
  1,200-update segments.
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
- **Push robustness measures duty factor's effect on stability.** Without
  disturbance every width and duty factor completes its trials, so success
  cannot separate duty factors. The pushed trials use the training push profile
  and report success against duty factor for each width and period.
- **No single fixed period.** Every stage spans the gait's periods. The period
  sweep already contains the 0.48 s slice, so no separate fixed-period grid is
  collected.

## Duty-factor anchor spacing

Trot and walk are trained and evaluated at differently spaced duty factors:

| | Trot | Walk |
|---|---|---|
| Anchors | 0.50, 0.625, 0.75 | 0.75, 0.80, 0.85, 0.90 |
| Step between anchors | 0.125 | 0.05 |
| Swing difference between neighbours at 0.48 s | 3 control ticks | about 1 control tick |

The only shared value is 0.75. At a 0.48 s period the walk schedules round to
0.750, 0.833, 0.875 and 0.917, so the realized walk levels are unevenly spaced
and change with period.

**Why it matters**

- **Neighbouring walk levels nearly merge once executed.** Their spacing is the
  size of the controller's typical timing error. In the earlier specialist
  data, walk 0.85 ran at 0.859 and walk 0.90 at 0.879. Push training shifts
  duty factor slightly, which narrows the gap further.
- **Resolution differs between gaits.** Trot trends and selector choices move
  in 0.125 steps, walk in 0.05 steps. Trot curves look step-like and walk
  curves smooth for that reason alone, and part of the U shape in the trot
  selector comes from having only three candidates.
- **Training exposure is not affected.** The gaits are separate policies, and
  every anchor receives millions of training steps.

**How to analyze with it**

- Use the realized duty factor, `achieved_df`, as a continuous variable
  instead of the anchor labels.
- Compare trot and walk at matched realized duty factor, around 0.75 where the
  two ranges meet, as Investigation 1 does.
- Before claiming a difference between neighbouring walk anchors, check that
  their `achieved_df` distributions actually separate.
- Do not interpret differences in selector step size between gaits as a result.

## Outputs

| Location | Content |
|---|---|
| `results/narrow_specialist_{trot,walk}_seed5_3072_<tag>/` | clean checkpoints |
| `results/narrow_specialist_push_{trot,walk}_from_seed5_3072_<tag>/`, `..._seg2_3072_<tag>/` | push fine-tune, segments 1 and 2 |
| `results/narrow_specialist_push_{trot,walk}_seg3_3072_<tag>/` | push-trained checkpoints used for every evaluation |
| `results/validation_narrow_<gait>_v030_p<period>_<tag>/` | fidelity archives, tables, plots |
| `results/<tag>_fidelity_gate.json` | gate report, including where in the gait phase contact deviates |
| `results/narrow_surfaces_sweep_<gait>_<tag>/grid/` | period-sweep archives and `surface_trials.csv` |
| `results/push_robustness_narrow_<gait>_v030_p<period>_<tag>/` | pushed-trial archives and `perturbation_summary.json` |
| `PAPER_GRAPHS/narrow_specialist/push_robustness/` | success against duty factor per width and period, `push_robustness_success.csv` |
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

Set `TAG` to start a separate run, `PYTHON_BIN` if the Isaac Lab Python is
elsewhere, and `REPO_ROOT` when running a copy of the script from another
directory. The script is resumable: completed stages are skipped, and a
training stage without its final checkpoint is retrained from the start.

Evaluation accepts a push-trained checkpoint only if the training source files
(`PUSH_TRAINING_SOURCE_FILES` in `narrow_specialist_push.py`) still match the
hash recorded at training time, so leave them unchanged while a run is in
progress.

## Code map

| File | Role |
|---|---|
| `source/beam_walking/beam_walking/experiment/narrow_specialist.py` | command ranges, anchors, sampler, lineage |
| `source/beam_walking/beam_walking/experiment/narrow_specialist_task.py` | environment config: self-collision, placement tolerance |
| `source/beam_walking/beam_walking/experiment/narrow_specialist_push.py` | push fine-tune identity and parent verification |
| `scripts/narrow_specialist_surface_data.py` | sweep definitions and checkpoint checks for the collector |
| `source/beam_walking/beam_walking/experiment/unified_duty_selector.py` | selector model, labels, validation evidence |
| `scripts/make_narrow_push_robustness_figures.py` | push-robustness figures and per-cell success table |
| `scripts/export_push_outcomes.py` | tidy push-outcome export for the earlier push data |

## Limitations

- Stances near 0.05 m rely on simulated self-collision, which has not been
  validated against hardware.
- Energy is positive mechanical joint work per distance. It omits motor
  heating, so it understates the cost of high-torque postures such as wide stance.
- Fidelity runs use 0.30 m/s. The period sweep covers the other speeds.
- Results are simulation on flat ground with commanded stance width; there is no
  physical narrow support.
