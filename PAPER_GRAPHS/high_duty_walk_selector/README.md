# High-duty walk selector figures

These figures summarize the separate seed-4 high-duty walking policy at period
0.48 s. The selector was evaluated at commanded full step widths 0.10--0.50 m,
speeds 0.25--0.40 m/s, and candidate duty factors 0.75, 0.80, 0.85, and 0.90.

- `high_duty_walk_policy.png` / `.pdf`: final consolidated validation figure.
  All four speed traces coincide. DF 0.80 was selected at every width and speed,
  and the median achieved duty factor was 0.8229.
- `combined_fresh_selector_validation_p048.png` / `.pdf`: period-0.48 s
  comparison with the original adaptive selector validation. Solid, filled
  traces are the original seed-3 controller; dashed, open traces are the
  separately trained seed-4 high-duty controller. The displayed width range is
  0.10--0.40 m; the CSV retains the measured 0.50 m contexts.
- `combined_selector_validation_summary.csv`: source rows for the combined
  figure, including a controller label so the two validations remain distinct.
- `all_candidate_duty_response_p048.png` / `.pdf`: every measured duty-factor
  candidate for trot and walk, split by speed. Colored curves show achieved DF
  across width, crosses mark candidates that failed the 90% compliance gate,
  and black curves show the fresh selector outputs. The displayed width range
  is 0.10--0.40 m; the CSV retains the measured 0.50 m contexts.
- `all_candidate_duty_response_summary.csv`: candidate-level source data for
  that figure, summarized from 10,240 held-out grid trials.
- `high_duty_walk_policy_by_speed.png` / `.pdf`: original validation rendering
  with a separate trace for each commanded speed.
- `selector_fit.png` / `.pdf`: grid-label fit at the exact supported contexts.
- `candidate_energy_and_compliance.pdf`: condition-level candidate diagnostics.
- `selector_validation_summary.csv` and `candidate_summary.csv`: plotted data.

All 20 selected contexts passed fresh validation with 32/32 compliant trials.
The experiment found no duty-factor change with step width or speed in this
domain. These are controller-specific flat-ground results, not a universal
biomechanical optimum.

The complete hashed evidence remains in
`results/high_duty_walk_selector_validation_seed4_20260912` and
`results/high_duty_walk_selector_seed4_20260912`.
