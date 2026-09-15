# Narrow-stance specialist data set

Trot and walk policies commanded over stance widths of 0.05-0.30 m. How they
were trained and evaluated, the file layout and the column meanings are
described in `PIPELINE.md` at the repository root. The CSVs and figures in this
folder are written by the export step when the run completes.

## Summary for review

**Controller.** One trot and one walk policy (Unitree Go2, PPO, 1,800 updates,
seed 5, self-collision on, no domain randomization), trained **without external
disturbances**. Checkpoint hashes are in `policies/`.

**Why not disturbance-trained policies.** Fine-tuning under random pushes was tried
for the reported controller. After it, the trot policy held its front-right leg in stance for
about 70% of the cycle at every commanded duty factor (0.50, 0.625 and 0.75),
width, speed and period, so almost no trot trial met the per-leg duty-factor
and contact-sequence checks and the selector had no trot contexts. Before that
fine-tuning, all four legs track duty factors 0.55-0.75 within about 0.03 (at 0.50
the rear-right leg runs 0.05-0.08 long). Disturbances are therefore applied only
at evaluation, where they measure robustness.

**What was evaluated**

| Measurement | Scope | Result |
|---|---|---|
| Command fidelity | every period, 6 widths, trot duty 0.50-0.75 and walk 0.75-0.90 in 0.05 steps, 64 trials per cell | realized duty rises with the command: 92-100% of the commanded spread for trot, 58-76% for walk (walk at 0.40 s is below the 60% check, where two-tick swings round the schedule) |
| Robustness | same cells, random base disturbances up to 25 N | success rises with duty factor at narrow stance for both gaits (figure 3); the stronger 50 N level was not needed because wider stances did not reach full success; per-period counts in `trials/robustness_*.json` |
| Period sweep | 4 speeds, 6 widths, every period, 32 trials per cell | periodicity and cost trends in figures 1-2 |
| Selected duty factor | robustness cells above | the lowest duty factor keeping at least 90% of the best success rate at each width rises as stance narrows (figure 5) |
| Duty-factor selector network | 132 contexts (trot 96, walk 36), lowest-cost duty factor among reliably executed candidates | validated on fresh seeds in all 132 contexts; not used by any figure; files in `selector_fit/` and `selector_promoted/` |

**Caveats**

- One training seed per gait.
- Figure 5's intervals are widest for trot at 0.10–0.15 m (0.60–0.75) and walk
  at 0.10 m (0.80–0.90). At 0.10 m successes are few (trot at most 10 of 128
  trials per duty factor, walk at most 19 of 192). At trot 0.20 m, 0.55 is
  selected over 0.50 by half a trial: 0.50 has 112 of 128 successes against a
  bar of 112.5. At 0.05 m no duty factor succeeds under this disturbance, so
  there is no selection.
- Figure 5 depends on the 90% share. At 80% and 95% the selection still rises
  from 0.30 m to 0.10 m for both gaits. At each period on its own it never falls
  as stance narrows, except walk at 0.40 s, which selects 0.80 at 0.30 m and
  0.75 at 0.25 m. A fixed success level from 50% to 90% would give no selection
  at 0.10 m.
- The selection prefers the lower of nearly equally robust duty factors, which
  assumes a less conservative gait is better when robustness is equal.
- Walk has selector contexts at 36 of 72 combinations. At some periods swing-tick
  rounding moves walk's executed schedule more than 0.05 from the commanded
  duty factor, which the compliance check rejects.
- Periodicity (cycle RMS) is a diagnostic of gait execution, not the convergence
  metric χ of Investigation 1.
- Simulation on flat ground with commanded stance width; no physical narrow
  support.

## Duty factors

| | Trot | Walk |
|---|---|---|
| Training anchors | 0.50, 0.625, 0.75, with duty drawn continuously over 0.50-0.75 | 0.75, 0.80, 0.85, 0.90, with duty drawn continuously over 0.75-0.90 |
| Evaluated | 0.50, 0.55, 0.60, 0.65, 0.70, 0.75 (0.75 from 0.40 s) | 0.75, 0.80, 0.85, 0.90 |

The only shared value is 0.75. Discrete swing ticks round each commanded
schedule, so realized duty factors are unevenly spaced and change with period.

**How to analyze with it**

- Use the realized duty factor, `achieved_df`, as a continuous variable
  instead of the command labels.
- Compare trot and walk at matched realized duty factor, around 0.75 where the
  two ranges meet, as Investigation 1 does.
- Before claiming a difference between neighbouring duty factors, check that
  their `achieved_df` distributions actually separate.

## Figures and the claims they support

Figures are in `figures/`, each with the table it plots. Numbers below are read
from those tables.

| Figure | Paper claim | What it shows |
|---|---|---|
| 1. Periodicity error against realized duty factor | Higher duty factor improves convergence (Investigation 1) | Trot cycle RMS falls as duty factor rises at every stance width, most at narrow stance: 0.107 at duty 0.50 to 0.003 at 0.75 at 0.05 m, and 0.010 to 0.002 at 0.15 m. Walk stays between 0.004 and 0.019 across 0.75–0.90. Periodicity is a diagnostic of gait execution, not the convergence metric χ. |
| 2. Cost of transport against realized duty factor | Duty factor affects energetic cost | Cost of transport (positive mechanical) rises with duty factor for both gaits at every width: trot 0.16 to 0.28 and walk 0.31 to 0.38 at 0.05 m. A higher duty factor costs energy, so it is worth raising where stance is constrained. |
| 3. Robustness (success rate) against stance width | Higher duty factor gives more robust locomotion on narrow terrain (Investigation 3) | At 0.15 m, trot success rises from 23% at duty 0.50 to 74% at 0.75, and walk from 51% at 0.75 to 77% at 0.85. At 0.20 m, trot rises from 88% to 98% and walk from 88% to 99%. Every duty factor fails at 0.05 m and succeeds at 0.25–0.30 m, so duty factor matters where stance is constrained. |
| 4. Realized duty factor by width | The learned controller executes the commanded duty factor (Investigation 2, command fidelity) | Trot realizes 0.52–0.76 for commands 0.50–0.75 and walk 0.76–0.87 for 0.75–0.90, the same at every stance width. |
| 5. Selected duty factor against stance width | As stance width decreases, higher duty factors are selected: duty factor is the parameter to raise when stance is constrained, and not when it is not | At each width the lowest duty factor keeping at least 90% of the best success rate under disturbance is selected. Trot: 0.50 at 0.25–0.30 m, 0.55 at 0.20 m, 0.70 at 0.15 m and 0.75 at 0.10 m. Walk: 0.75 at 0.25–0.30 m, 0.80 at 0.15–0.20 m and 0.85 at 0.10 m. At 0.05 m no duty factor succeeds under this disturbance. |
