# Validated adaptive duty-factor selector — seed 3

The fresh 3,072-environment PPO run completed all 1,800 updates. A fixed
flat-ground grid evaluated six candidate duty factors at five commanded step
widths, four speeds, and both gaits (7,680 trials total). The selector fitted
39 eligible contexts and abstained for walk at width 0.10 m, speed 0.25 m/s,
period 0.48 s because no candidate passed the registered gate.

Fresh validation used 32 new trials per supported context beginning at seed
4,000,000. All 39 contexts achieved 100% compliance and finite mechanical CoT,
so `validated_duty_selector.pt` is deployment-ready for those exact contexts.
The maximum difference between selected and discretely achieved duty factor was
0.0335, inside the 0.05 gate.

At period 0.48 s, trot generally selects 0.55–0.60; the narrowest 0.10 m,
0.25 m/s context selects 0.65. Walk selects 0.75 for all supported contexts.
These values are the lowest-energy compliant choices observed for this trained
controller and protocol, not universal biomechanical optima.

Main artifacts:

- `selected_vs_achieved_duty_factor.png`: fresh rollout validation graph.
- `selector_validation_summary.csv`: one row per validated context.
- `selector_validation_trials.csv`: all 1,248 fresh validation trials.
- `validated_duty_selector.pt`: queryable selector checkpoint.
- `SELECTOR_VALIDATION_COMPLETE`: hashed completion record.

The grid-derived fitted curve and candidate diagnostics are in
`../adaptive_duty_selector_seed3_20260911/`.
