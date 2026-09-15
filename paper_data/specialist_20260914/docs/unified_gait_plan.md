# Unified trot/walk controller protocol

Registered 2026-09-14. One fresh PPO policy is trained on both gaits under one
period distribution, replacing the split between the paper policy (seed 2:
trot DF 0.50--0.75, walk DF 0.75, period 0.36--0.54 s) and the high-duty walk
policy (seed 4: walk only, DF 0.75--0.90, period fixed at 0.48 s). Existing
checkpoints, selectors, and push-study results are untouched and remain bound
to their own sources.

## Registered domain

| Quantity | Trot | Walk |
|---|---|---|
| Gait share | 50% | 50% |
| Duty-factor anchors | 0.50, 0.625, 0.75 | 0.75, 0.80, 0.85, 0.90 |
| Continuous duty range | 0.50--0.75 | 0.75--0.90 |
| Minimum requested swing | 5 ticks (0.10 s) | 2 ticks (0.04 s) |
| Period, anchor stage | 0.48 s | 0.48 s |
| Period, continuous stage | 18--27 ticks (0.36--0.54 s) | 20--27 ticks (0.40--0.54 s) |
| Step width | 0.10--0.50 m, anchors every 0.10 m | same |
| Forward speed | 0.25--0.40 m/s, anchors every 0.05 m/s | same |
| Grounded stance resets | 10% | same |
| Terrain | flat ground; width is a foot-separation command | same |

The swing bounds are the ones each gait was previously registered with. The
walk period floor of 20 ticks is the smallest period at which DF 0.90 keeps two
swing ticks; it is the same floor the seed-2 sampler already applied to walk.
At every walk period from 20 to 27 ticks the four walk anchors map to distinct
discrete swing-tick schedules; at every trot period from 18 to 27 ticks the
three trot anchors map to distinct schedules (checked in `tests/test_unified_gait.py`).

## Curriculum

Same staging as the paper sampler in `protocol.py`:

1. Foundation (first 50 updates): speed 0.30, width 0.30, period 0.48; gait and
   duty strata balanced.
2. Core (updates 50--150): balanced speed/width anchors within every stratum at
   period 0.48.
3. Expansion (after update 150): 75% anchor commands as in the core stage, 25%
   continuous speed, width, period, and duty. Continuous duty is drawn from
   equal-probability Voronoi bins around each gait's anchors and capped by
   1 - min_swing / period_ticks for that gait. Note that the seed-4 high-duty
   sampler drew continuous walk duty uniformly over 0.75--0.90; the unified
   sampler uses the paper's Voronoi-bin rule for walk as well as trot, so the
   outer walk bins (0.75--0.775 and 0.875--0.90) carry twice the density they
   had in the seed-4 run.

## Interface

Observations, actions, reward, plant, and normalization are the repository
defaults from `task.py`. The duty command is still normalized against
0.50--0.75, so walk duty above 0.75 sits above +1, exactly as in the high-duty
walk policy. The 68-dimensional observation is unchanged. Motor gains are
nominal (no deployment domain randomization), matching both prior policies.

## Training

`scripts/unified_gait_experiment.py train` with 3,072 environments, 1,800
updates, seed 5, and 0.10 grounded-start probability. Task identity is the
hash of `task.py`, `protocol.py`, `unified_gait.py`, and `unified_gait_task.py`.
The checkpoint records that hash and a lineage id bound to the training source
hash and seed.

## Run order

1. Offline tests (`tests/test_unified_gait.py`) and council review.
2. Simulator smoke test (64 environments): every gait/DF stratum present at
   reset, finite observations and rewards, phase-clock alignment, grounded
   resets begin supported.
3. Fresh PPO training.
4. Evaluation, every stage on the same seed-5 checkpoint and all with the
   unified factor levels (trot DF 0.50/0.625/0.75, walk DF
   0.75/0.80/0.85/0.90, widths 0.10-0.50 m, speeds 0.25-0.40 m/s):
   - Command fidelity (`evaluate_policy.py --task unified`, 64 trials per
     cell, speed 0.30 m/s): trot and walk at 0.48 s, then trot at 0.36 and
     0.54 s (0.36 s carries only DF 0.50/0.625, the trot swing bound) and
     walk at 0.40 and 0.54 s.
   - Speed x width x DF surfaces at 0.48 s
     (`collect_policy_surfaces.py --task unified`, 140 conditions x 32
     matched trials): cost of transport, periodicity, and compliance for
     both gaits; figures via `make_policy_surfaces.py --task unified`.
   - Period sweep (`--period-sweep`, 460 conditions x 32 trials): trot at
     0.36/0.40/0.48/0.54 s and walk at 0.40/0.48/0.54 s with the feasible
     duty anchors at each period. This is the selector's label source.
   - Unified selector (`fit_unified_duty_selector.py`): input
     [step width, speed, period, gait], output masked to the gait's feasible
     candidates at that period; fresh-seed rollout validation
     (`collect_policy_surfaces.py --task unified --selector-checkpoint`) and
     promotion (`validate_unified_selector.py`).
   - The walk-only fixed-period grid/selector of the seed-4 protocol is also
     re-run on seed 5 as a like-for-like cross-check only; it is not the
     unified selector.

## Selector objective and the shape of its curves

The selector labels each context with the lowest-energy duty factor among
candidates passing the compliance gates. Positive mechanical cost of transport
usually rises with duty factor, but the low-duty cost grows fastest with width,
so trot duty 0.625 becomes cheaper than 0.50 at 0.50 m (0.40 and 0.48 s). The
selected duty factor is therefore set by execution limits at narrow stance and
by that energy crossover, reinforced by the gates, at wide stance. Its
dependence on width therefore traces gate eligibility, which fails at both
extremes of stance width, and can reverse direction (trot at 0.48 s: 0.625,
0.50, 0.50, 0.50, 0.625 across 0.10 to 0.50 m). It is not a stability trend.

The paper's robustness claim is tested by the push study. Under the ramped base
push, trot survival at 0.10 m increases monotonically with duty factor, and a
rule selecting the lowest duty factor with push survival of at least 90% gives
a monotone decrease from 0.75 at 0.10 m to 0.50 at 0.30 m and wider. See
`PAPER_GRAPHS/specialist_selector/README.md` for the tables and limits.

## What this does and does not replace

Nothing already collected is invalidated. The adaptive selector (seed 3), the
high-duty selector (seed 4), and the perturbation studies (seed-2 lineage) are
each hash-bound to their own checkpoints. If the paper is to report a single
policy for all of them, those grids need to be recollected on this policy;
that is a separate decision after its command fidelity is measured.
