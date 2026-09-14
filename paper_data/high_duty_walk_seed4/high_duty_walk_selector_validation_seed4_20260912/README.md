# Validated high-duty walk selector — seed 4

The independent flat-ground PPO controller was trained from scratch for 1,800
updates with 3,072 environments. It covers walk at period 0.48 s and commanded
duty factors 0.75, 0.80, 0.85, and 0.90 across full step widths 0.10–0.50 m
and speeds 0.25–0.40 m/s.

The grid used 2,560 matched held-out trials. The selector maps commanded step
width and speed to one of the four tested duty factors by choosing the
lowest-energy candidate that passed the registered compliance gates. It fitted
20 contexts and rejected 0.
Fresh rollout validation contains 20 context summaries.

The final figure consolidates the four coincident speed traces so the result is
readable: the selector commanded DF 0.80 and achieved median DF 0.8229 at every
width and speed. Its compliance panel shows that DF 0.80 passed the eligibility
gate in all 20 contexts; DF 0.85 passed in 3, while DF 0.75 and 0.90 passed in
none. The original per-speed rendering is retained as
`selected_vs_achieved_duty_factor_by_speed.png`.

Main artifacts:

- `selected_vs_achieved_duty_factor.png`: selected and achieved DF versus step width.
- `selector_validation_summary.csv`: one row per validated context.
- `selector_validation_trials.csv`: fresh held-out validation trials.
- `validated_high_duty_walk_selector.pt`: validated selector policy.
- `SELECTOR_VALIDATION_COMPLETE`: hashed validation completion record.

The selector outputs a command optimum for this trained controller and protocol;
it is not a universal biomechanical optimum.
