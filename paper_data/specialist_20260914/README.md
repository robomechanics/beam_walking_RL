# Gait-specialist data set (2026-09-14)

Three PPO policies trained under one protocol, evaluated on flat ground.
Everything here is measured, not derived from the earlier seed-2 or seed-4 work.

## Policies

| Directory | Gait | Training |
|---|---|---|
| `policies/unified_specialist_trot_seed5_3072_20260914` | trot | fresh, 3072 envs, 1800 updates, seed 5 |
| `policies/unified_specialist_walk_seed5_3072_20260914` | walk | fresh, same protocol |
| `policies/unified_specialist_push_trot_from_seed5_3072_20260914` | trot | warm start from the trot specialist, ramped pushes on, seed 3 |

Each holds provenance, the PPO and environment configs, and the checkpoint hash.
The checkpoints themselves stay in `results/` because of size.

Command ranges the policies were trained on: stance width 0.10-0.50 m, speed
0.25-0.40 m/s, period 0.36-0.54 s for trot and 0.40-0.54 s for walk, duty factor
0.50/0.625/0.75 for trot and 0.75/0.80/0.85/0.90 for walk. Trot needs five swing
control ticks and walk two, which is why trot drops 0.75 below 0.40 s.

## Per-trial tables (`trials/`)

One row per rollout. Columns include the commanded factors, the achieved duty
factor, achieved stance width, forward speed, lateral error, per-leg duty, the
compliance and contact-topology gate outcomes, and positive mechanical cost of
transport.

| File | Content |
|---|---|
| `grid_speed_width_duty_BY_PERIOD_{trot,walk}.csv` | **the main grid.** Speed x width x duty repeated at each period: trot 0.36/0.40/0.48/0.54 s, walk 0.40/0.48/0.54 s. Four speeds, five widths, every feasible duty anchor, 32 matched trials per cell. Trot has 40 cells at 0.36 s rather than 60 because duty 0.75 needs five swing control ticks, which 18 ticks cannot supply. |
| `grid_speed_width_duty_FIXED_p048_{trot,walk}.csv` | the same grid at 0.48 s only, redundant with the 0.48 slice above; kept because the paper-style surface figure reads it |
| `selector_validation_{trot,walk}.csv` | fresh-seed rollouts of the selector's choice per context |
| `fidelity_*_p0??.csv` and `*_summary.csv` | 64 held-out trials per condition at 0.30 m/s |
| `push_*_nominal_*.csv` | per-trial tables for the undisturbed push arms |

## Push study (`push/`)

`push_trials.csv` and `push_conditions.csv` cover every arm on the same footing,
including the pushed ones, which the standard analyzer refuses to process.
Condition rows carry success/failure/timeout counts, the success rate with a
Wilson 95% interval, mean travel, and mean end time. 32,000 trials, 500
conditions. The disturbance is a random world-frame base wrench up to 25 N and
3 N m, held 0.10-0.40 s at a time, ramping from a tenth of peak to full over 10 s.

## Selector (`selector_fit/`, `selector_promoted/`)

One selector for both gaits: inputs stance width, speed, period and gait; output
masked to that gait's duty anchors that are feasible at the requested period.
Labels are the lowest-energy candidate passing the compliance gates. All 133
contexts passed fresh-seed validation, worst selected-versus-achieved error 0.042.

**Reading the selected duty factor against width.** The curves are not monotone,
for example trot at 0.48 s selects 0.625 at 0.10 m, 0.50 from 0.20 to 0.40 m,
and 0.625 again at 0.50 m. The two ends have different causes.
At 0.10 m duty 0.50 is the cheapest but fails the execution gates (0.00 pass),
so it is excluded. At 0.50 m it narrowly fails the gate (0.84) and, separately,
its cost has risen faster with width than duty 0.625's, so 0.625 is now cheaper
(0.263 against 0.275). The U is specific to trot at 0.48 s; at other periods
duty 0.50 fails the gates at every width and the selection is flat.

The robustness trend the paper argues for is in the push study instead: under
push, trot survival at 0.10 m rises 0.07, 0.15, 0.29 across duty 0.50, 0.625,
0.75. Choosing the lowest duty factor whose push survival reaches 90% gives
0.75, 0.625, 0.50, 0.50, 0.50 across widths 0.10 to 0.50 m, which is monotone.
Full explanation, tables and limits: `docs/selector_objective.md`.

## Known gaps

- The walk push study has the base arms only. No walk push fine-tune was trained.
- The walk pushed arm covers 0.30 m/s at three periods, not the full speed set.
- Period coverage is four sampled periods per gait, not every 0.02 s step.
