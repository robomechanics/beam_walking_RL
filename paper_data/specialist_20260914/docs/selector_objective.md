# Duty-factor selector: what it optimizes and how to read its curves

## What the selector does

One selector covers both gaits. Its inputs are commanded stance width, forward
speed, gait period, and gait. For each input context it returns one duty
factor, chosen from that gait's trained anchors (trot 0.50, 0.625, 0.75; walk
0.75, 0.80, 0.85, 0.90) that are feasible at the requested period.

The label for each context is:

> the candidate with the lowest median positive mechanical cost of transport,
> among candidates whose trials pass the compliance and finite-energy gates in
> at least 90% of 32 matched trials.

The gates check that the commanded duty factor, stance width and speed were
executed, that every footfall and lift-off landed within one control tick of
the requested schedule, and that the gait settled into a repeating cycle.

Labels were built from the period sweep (trot at 0.36, 0.40, 0.48, 0.54 s; walk
at 0.40, 0.48, 0.54 s). All 133 labeled contexts passed fresh-seed validation,
and the worst gap between the selected and the realized duty factor was 0.042.

## Why the curves are not monotone in stance width

In `selected_vs_achieved_duty_factor.png` the trot panel at 0.48 s selects 0.625
at 0.10 m, 0.50 from 0.20 to 0.40 m, and 0.625 again at 0.50 m. The walk panel
at 0.54 s selects 0.80 at 0.10 m and 0.85 elsewhere. Neither is a stability
trend. Both follow from the objective.

**Energy mostly prefers the lowest duty factor, except at wide stance.** Median positive mechanical
cost of transport for trot at 0.48 s:

| Duty factor | 0.10 m | 0.20 m | 0.30 m | 0.40 m | 0.50 m |
|---|---|---|---|---|---|
| 0.50 | 0.163 | 0.179 | 0.193 | 0.233 | 0.275 |
| 0.625 | 0.199 | 0.210 | 0.223 | 0.242 | 0.263 |
| 0.75 | 0.309 | 0.310 | 0.309 | 0.324 | 0.337 |

Up to 0.40 m the lowest duty factor is the cheapest. Its cost also grows fastest
with width (duty 0.50 rises 69% from 0.10 to 0.50 m, duty 0.625 rises 32%, duty
0.75 rises 9%), so the curves cross: at 0.50 m duty 0.625 is cheaper than 0.50
(0.263 against 0.275). The same crossover appears at 0.40 s and is a near tie at
0.36 s. Walk is not monotone in duty factor either: duty 0.90 is often cheaper
than 0.85.

**The gates decide which duty factors are allowed to compete.** Fraction of
trot trials at 0.48 s passing every gate:

| Duty factor | 0.10 m | 0.20 m | 0.30 m | 0.40 m | 0.50 m |
|---|---|---|---|---|---|
| 0.50 | 0.00 | 1.00 | 1.00 | 1.00 | 0.84 |
| 0.625 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Duty 0.50 clears the 90% threshold only between 0.20 and 0.40 m. The two ends
of the U have different causes. At 0.10 m duty 0.50 is cheaper but cannot be
executed cleanly, so it is excluded. At 0.50 m it narrowly fails the gate and is
also no longer the cheapest, so 0.625 would be selected on energy alone. At 0.50 m the failing gate is contact timing, not
command tracking: realized width stays within about a centimetre and the
command gate passes, but footfalls drift outside the one-tick tolerance more
often at the widest stance.

The selected-duty curves therefore show where each duty factor can be executed
cleanly. They should not be read as the duty factor the controller needs for
stability at that width.

## The robustness trend lives in the push study

The paper's claim is that higher duty factor improves robustness in narrow
conditions. That is measured by the push study
(table `push/push_conditions.csv`).
Fraction of trot trials completing the course under the ramped base push, base
policy, pooled over the collected speeds and periods:

| Duty factor | 0.10 m | 0.20 m | 0.30 m | 0.40 m | 0.50 m |
|---|---|---|---|---|---|
| 0.50 | 0.07 | 0.87 | 1.00 | 1.00 | 1.00 |
| 0.625 | 0.15 | 0.93 | 1.00 | 1.00 | 1.00 |
| 0.75 | 0.29 | 0.98 | 1.00 | 1.00 | 1.00 |

Survival rises with stance width, and within the narrow band it rises
monotonically with duty factor.

A selector built on robustness instead of energy recovers the paper's
direction. Taking the lowest duty factor whose push survival reaches a
threshold, and using energy only as the tiebreak:

| Survival threshold | 0.10 m | 0.20 m | 0.30 m | 0.40 m | 0.50 m |
|---|---|---|---|---|---|
| 90% | 0.75 | 0.625 | 0.50 | 0.50 | 0.50 |
| 95% | 0.75 | 0.75 | 0.50 | 0.50 | 0.50 |

Narrow stance selects high duty factor, wide stance selects low duty factor,
with no reversal.

## Limits of the robustness version

- Push data exists for trot only. A walk robustness selector needs the walk
  push runs, which were not collected.
- The push grid is coarser than the energy selector's contexts. The tables
  above pool over speed and period, so a per-context robustness selector would
  either need push trials at every context or be defined at that coarser
  resolution.
- The energy selector in this folder is unchanged and remains a valid answer to
  its own question: the cheapest duty factor that the controller executes
  cleanly.
