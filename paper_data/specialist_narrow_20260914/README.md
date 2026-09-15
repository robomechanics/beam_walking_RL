# Narrow-stance specialist data set

Push-trained trot and walk policies commanded over stance widths of 0.05-0.30 m.
How they were trained and evaluated, the file layout and the column meanings
are described in `PIPELINE.md` at the repository root. The CSVs and figures in
this folder are written by the export step when the run completes.

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
