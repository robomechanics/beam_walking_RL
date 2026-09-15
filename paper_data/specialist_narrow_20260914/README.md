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
| Optimal duty factor | robustness cells above, with undisturbed CoT from the period sweep at 0.30 m/s | the duty factor with the lowest CoT divided by success rate rises as stance narrows (figure 5) |
| Duty-factor selector | 132 contexts (trot 96, walk 36), lowest-cost duty factor among reliably executed candidates | validated on fresh seeds in all 132 contexts; files in `selector_fit/` and `selector_promoted/` |

**Caveats**

- One training seed per gait.
- Figure 5's intervals are wide where successes are few or scores nearly tie. At
  0.10 m successes are few (trot at most 10 of 128 trials per duty factor, walk
  at most 19 of 192), giving trot 0.60–0.75 and walk 0.80–0.90. At 0.15 m trot
  duty factors 0.55–0.75 score within 7% of each other, so the trot optimum
  there (0.75, interval 0.55–0.75) is not settled; only 0.50 is clearly worse.
  At 0.05 m no duty factor succeeds under this disturbance, so there is no
  optimum.
- Figure 5 depends on how success is weighted against energy. Weighting success
  half or twice as strongly (success rate raised to 0.5 or 2) keeps the optimum
  at 0.10 m above the optimum at 0.30 m for both gaits. Ignoring success, the
  lowest duty factor is the cheapest at every width, and success thresholds from
  50% to 90% give no duty factor at 0.10 m. A subtracted score, success minus
  λ times the CoT increase relative to the cheapest duty factor at that width,
  has no weight that suits every width: above
  λ = 0.10 trot at 0.10 m picks 0.50, which never succeeds, and below λ = 0.49
  trot at 0.30 m picks 0.55 on a one-trial success difference.
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
| 5. Optimal duty factor against stance width | As stance width decreases, higher duty factors give optimal performance: duty factor is the parameter to raise when stance is constrained, and not when it is not | Each duty factor is scored by mean undisturbed CoT divided by success rate under disturbance, the expected energy per successful traversal. Trot: 0.50 at 0.20–0.30 m and 0.75 at 0.10–0.15 m. Walk: 0.75 at 0.25–0.30 m, 0.80 at 0.15–0.20 m and 0.85 at 0.10 m. At 0.10 m trot 0.50 never succeeds (0 of 128). At 0.05 m no duty factor succeeds under this disturbance. |
