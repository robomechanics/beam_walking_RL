# High-duty walk selector protocol

This is an independent flat-ground extension. It does not modify or replace the
validated adaptive selector in `results/adaptive_duty_selector_validation_seed3_20260911`.

## Registered domain

- Gait: walk only.
- Commanded full step width: 0.10, 0.20, 0.30, 0.40, and 0.50 m.
- Commanded speed: 0.25, 0.30, 0.35, and 0.40 m/s.
- Period: fixed at 0.48 s (24 control ticks at 50 Hz).
- Candidate duty factors: 0.75, 0.80, 0.85, and 0.90.
- Reset distribution: 10% independently sampled grounded stance starts.
- Terrain: flat ground. Width is a foot-separation command, not terrain width.

At 0.48 s, the four candidates request 6.0, 4.8, 3.6, and 2.4 swing
control ticks. The discrete schedules contain 6, 4, 3, and 2 swing ticks,
respectively, so their schedule duty factors are 0.7500, 0.8333, 0.8750, and
0.9167. Both commanded and achieved duty factors are retained in the evidence.

## Low-level controller

Train a fresh PPO controller with 3,072 environments for 1,800 updates at seed
4. The task, observations, actions, reward, plant, and flat-ground setup remain
the repository defaults; only the training command sampler is replaced. The
sampler is balanced across the four registered duty factors during foundation
and core training, then combines anchor commands with continuous commands in
the same width and speed ranges. It always emits walk at period 0.48 s.

The run is fresh because the old checkpoint was trained against a different,
hash-bound task source. Mixing it with the current task would weaken source
identity and make the extension harder to interpret.

## Grid, labels, and selector

Evaluate every 5 widths x 4 speeds x 4 duty factors with 32 matched held-out
trials starting at seed 5,000,000. Each condition settles for 12 gait cycles and
measures the final 4. A candidate is eligible only when at least 90% of trials
pass the frozen command-compliance and finite-energy gates. The label for each
width-speed context is the eligible candidate with the lowest median positive
mechanical cost of transport.

Fit a small supervised network from `[step_width, speed]` to duty factor. Its
output is bounded to `[0.75, 0.90]`, and it abstains outside the exact labeled
width-speed contexts. The main graph has step width on the x axis, selected
duty factor on the y axis, and one legend entry per speed.

Validate the fitted selector with 32 fresh trials per supported context starting
at seed 6,000,000. Promotion requires at least 90% compliance, at least 90%
finite-energy compliant trials, and median achieved duty factor within 0.05 of
the selected value in every context. Failed contexts remain explicit
abstentions.

## Run order

1. Offline tests and exact-source council review.
2. Simulator smoke test.
3. Fresh PPO training.
4. Full candidate grid.
5. Offline selector fit and graph generation.
6. Fresh selector rollout validation and final README.

GPU, VRAM, RAM, and process capacity are checked immediately before every
simulator launch.
