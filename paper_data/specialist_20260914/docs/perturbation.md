# Ramped base perturbation (opt-in)

Added 2026-09-12 at the user's request. Off by default; the paper protocol
stays push-free unless `--perturbation` is passed.

## What it does

A random external force and torque act on the Go2 `base` link, in the world
frame. Each environment samples a direction and a magnitude fraction, holds
that wrench for a random duration (`--perturbation_hold MIN MAX`, default
0.10 to 0.40 s), then resamples. The wrench is scaled by an envelope that
starts at `--perturbation_start_fraction` (default 0.10) of the peak when an
episode starts and rises linearly until it reaches the peak:

    envelope = start + (1 - start) * clip(progress, 0, 1)
    progress = episode_time / ramp_time            (--perturbation_ramp_mode time, default)
             = forward_travel / ramp_distance      (--perturbation_ramp_mode distance)

Peak magnitudes are `--perturbation_max_force` (default 25 N) and
`--perturbation_max_torque` (default 3 N m); `--perturbation_ramp_time`
defaults to 10 s and `--perturbation_ramp_distance` to 3 m. The sampled
magnitude fraction is uniform in [0, 1], so the peak is a bound, not a
constant. The force heading is uniform in the horizontal plane with a vertical
component of at most 25% of the horizontal one; torque directions are uniform.

## Where it lives

- `source/beam_walking/beam_walking/experiment/perturbation.py`: profile,
  envelope, sampler, flags, outcome summary. Pure torch/numpy.
- `source/beam_walking/beam_walking/experiment/perturbation_env.py`: a mixin
  that writes the wrench to Isaac Lab's permanent wrench composer once per
  control step, before the physics substeps, and records the applied wrench in
  captured transitions. `perturbed_env_class(BeamEnv | DeploymentBeamEnv, cfg)`
  builds the subclass.
- `task.py` and `protocol.py` are untouched, so `task_sha256` and every
  existing checkpoint's evaluation eligibility are unchanged.

## Training

    python scripts/beam_experiment.py train ... --perturbation \
        [--perturbation_max_force 25 --perturbation_max_torque 3 \
         --perturbation_ramp_time 10 --perturbation_start_fraction 0.1 \
         --perturbation_hold 0.1 0.4]

Provenance records `external_pushes: true` and the full `perturbation` profile.
The smoke mode verifies the composer is active, the applied wrench is finite,
and the envelope and force stay within their bounds (`smoke_checks.json`,
key `perturbation`).

## Evaluation

Both evaluators accept the same flags:

    python scripts/beam_experiment.py evaluate --checkpoint ... --perturbation ...
    python scripts/evaluate_policy.py --checkpoint ... --output ... --perturbation ...

When enabled, trial archives end in `_perturbed` instead of `_nominal`, carry
`disturbed=True`, `external_pushes=True`, the profile JSON, and per-step traces
`perturbation_force_w`, `perturbation_torque_w` and `perturbation_envelope`.
The same disturbance sequence is replayed for every width/DF condition (the
sampler is reseeded with the evaluation seed before each condition).
`evaluate_policy.py` labels the archives with schema `<nominal>_perturbed`, so
the versioned nominal analysis will not accept them as protocol data. Each run
also writes `perturbation_summary.json` with success/failure/timeout counts,
applied-wrench statistics, and the mean envelope at failure steps per condition.

## Not changed

`high_duty_walk_experiment.py` and the adaptive/duty pipelines do not expose
the perturbation. The frozen deployment DR profile still declares
`external_pushes: false`; passing `--perturbation` with `--deployment_dr` is
allowed and recorded in provenance, but it is then not the frozen profile.
