# Adaptive duty-factor selector on flat ground

This is a separate experiment. It does not change the fixed-duty paper study,
its checkpoints, or its evaluation protocol. “Width” means commanded full
left-to-right foot separation in the robot body frame, not physical terrain or
beam width. All simulation remains on open flat ground without pushes.

## Scientific question and decision rule

The experiment asks how commanded step width changes the energy-efficient duty
factor for walk and trot. For each `(step width, speed, period, gait)` context,
the target is the candidate duty factor with the lowest median positive
mechanical cost of transport among candidates satisfying all of these gates:

- at least 90% of held-out trials pass;
- achieved duty factor is within 0.05 for every leg;
- stance-only achieved width is within 0.03 m;
- forward speed is within 0.04 m/s;
- stance and swing recall are at least 0.90;
- lateral position, heading, lateral velocity, and yaw rate pass the existing
  flat-ground limits;
- the operational 5 N contact topology score is at least 0.90;
- no physical termination occurs during the measured cycles;
- positive mechanical CoT is finite and based on positive forward progress.

Contexts with no eligible duty factor receive no label. The unresolved
finite-difference chi estimator is not used by this experiment.

## Controller and selector

The low-level controller retains the existing 68 policy observations, 12 joint
targets, PPO architecture, reward, nominal motor gains, flat terrain, gait
phase semantics, and 10% grounded starts. Its separate command sampler balances
both walk and trot over the same six primary candidates—0.50 through 0.75 in
0.05 increments—before adding uniform continuous interpolation. Every sampled
period/duty pair preserves at least
0.10 s of requested swing.

The selector is a 32–32 ELU feedforward network. It receives commanded full
step width, forward speed, period, and one-hot gait. Its scalar sigmoid output
is mapped to `[0.50, min(0.75, 1 - 0.10/period)]`, so it cannot request an
infeasible duty factor. Robot state is deliberately excluded from version 1:
the requested graph is then an identifiable command-to-duty relationship rather
than an average over an unspecified state distribution.

The primary grid uses widths 0.10–0.50 m, speeds 0.25–0.40 m/s, period 0.48 s,
both gaits, candidate duty factors 0.50–0.75 in increments of 0.05, and 32
matched held-out trials per candidate. The collector measures four cycles after
eight prior settling cycles. Optional period sweeps use the same scripts and
automatically omit duty factors that violate the swing-duration bound.

## Execution sequence

1. Run CPU tests and independent exact-tree council review.
2. Check every GPU compute process, VRAM, and host RAM using the existing
   bounded RustDesk rule.
3. Run the adaptive chronology/command-distribution smoke test.
4. Train one fresh 1,800-update adaptive controller with 3,072 environments.
5. Inspect training diagnostics and use only the final requested checkpoint.
6. Run a one-condition recorder smoke, then collect the held-out fixed
   duty-factor grid with identical seeded reset plans and root/joint readbacks
   matched within 1e-6.
7. Fit the selector and bootstrap the selected grid labels. Initial curves are
   explicitly marked as fitted and unvalidated.
8. Evaluate each labeled selector context on 32 new matched seeds. The query
   CLI remains disabled unless every context passes the 90% compliance gate
   with finite energy. Generate:
   `duty_factor_vs_step_width.png`, its PDF version, and the multipage
   `candidate_energy_and_compliance.pdf` diagnostic, plus selected-versus-
   achieved duty-factor figures from fresh validation.
9. Treat one-controller results as descriptive. Train independent low-level
controller seeds before making population-level claims.

An old compatible 68-input checkpoint may be passed with `--warm-start` for a
development run. That path resets optimizer, iteration, and curriculum state
and records the parent hash, but it is ineligible for the primary grid because
the old walk policy was trained almost entirely at duty factor 0.75.

Commands are intentionally recorded here only as templates. Choose fresh result
directories and run them only after the pre-run gate passes.

```bash
python scripts/adaptive_duty_experiment.py smoke --num_envs 64 --steps 48 \
  --output results/adaptive_duty_smoke --headless

python scripts/adaptive_duty_experiment.py train --num_envs 3072 \
  --iterations 1800 --seed 3 --output results/adaptive_duty_seed3 --headless

python scripts/collect_duty_grid.py \
  --checkpoint results/adaptive_duty_seed3/model_1799.pt \
  --output results/adaptive_duty_grid_smoke_seed3 --num_envs 8 --smoke --headless

python scripts/collect_duty_grid.py \
  --checkpoint results/adaptive_duty_seed3/model_1799.pt \
  --output results/adaptive_duty_grid_seed3 --headless

python scripts/fit_duty_selector.py \
  results/adaptive_duty_grid_seed3/duty_grid_trials.csv \
  --output results/adaptive_duty_selector_seed3

python scripts/collect_duty_grid.py \
  --checkpoint results/adaptive_duty_seed3/model_1799.pt \
  --selector-checkpoint results/adaptive_duty_selector_seed3/duty_selector.pt \
  --output results/adaptive_duty_selector_validation_seed3 --headless

python scripts/select_duty_factor.py \
  results/adaptive_duty_selector_validation_seed3/validated_duty_selector.pt \
  --step-width .20 --speed .30 --period .48 --gait trot
```
