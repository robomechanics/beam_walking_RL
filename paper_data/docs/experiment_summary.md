# PPO gait-command experiment summary

## Research scope actually tested

The current project is a flat-ground controller-development experiment. One
standard PPO MLP directly outputs 12 Go2 joint-position targets through stock PD
control. The policy receives robot proprioception, foot contacts, gait phase,
forward-speed command, duty-factor command, full left-right step-width command,
period command, and gait identity. Training samples speed 0.25-0.40 m/s, width
0.10-0.50 m, period 0.36-0.54 s, feasible duty factor 0.50-0.75, and four-beat
walk or two-beat trot. Ten percent of resets use a calibrated four-foot grounded
stance. There is no terrain map, beam input, lateral command, or yaw command.

All completed training and evaluation reported here used an open flat plane.
Step width is a commanded body-frame foot-placement quantity; it is not physical
beam width. The runs therefore test whether PPO can execute gait parameters while
walking straight. They cannot test the original paper's claim that narrower
physical beams reduce traversal success.

## Exact policy inputs

The timing-fixed actor/critic input is a 68-value vector in this order:

| Signal | Values | Hardware source |
|---|---:|---|
| Body-frame base linear velocity | 3 | State estimate using IMU, leg kinematics/contact, or visual-inertial odometry; not directly measured by an IMU alone |
| Body-frame base angular velocity | 3 | IMU gyroscope |
| Gravity direction projected into the body frame | 3 | IMU attitude estimate; represents roll and pitch |
| Joint positions relative to the default pose | 12 | Joint encoders |
| Joint velocities relative to default | 12 | Encoder-derived or drive-estimated joint velocity |
| Previous policy action | 12 | Controller memory; known exactly |
| Sine of duty-warped phase, one per leg | 4 | Internal gait clock; known exactly |
| Cosine of duty-warped phase, one per leg | 4 | Internal gait clock; known exactly |
| Forward-speed command | 1 | External command |
| Duty-factor command | 1 | External command |
| Full left-right step-width command | 1 | External command |
| Gait-period command | 1 | External command |
| Gait identity, one-hot trot/walk | 2 | External command |
| Requested stance flags, FL/FR/RL/RR | 4 | Computed from the commanded gait clock and duty factor |
| Heading error from the fixed forward axis | 1 | Globally referenced yaw estimate or start-referenced integrated gyro |
| Binary foot-contact flags, FL/FR/RL/RR | 4 | Foot-force sensors or a torque/contact observer |
| **Total** | **68** | |

The evaluated timing-fixed checkpoint is
`paper_ppo_phasefix_pilot_seed1_3072_20260910/model_1799.pt`; runtime inspection
confirmed its 68-input actor and 12 joint-target outputs.

No policy input contains world position, lateral distance from the yellow line,
foot world position, terrain height scan, map, beam geometry, physical beam width,
or the applied disturbance force. In simulation, base motion/yaw are ground-truth
states and contacts come from exact simulated forces thresholded at 5 N. Observation
corruption is disabled. The signals are physically estimable, but hardware claims
would require evaluating the policy with a real estimator, latency, noise, bias,
and contact-classification errors.

## Experiment progression

| Revision | Main question or correction | Measured outcome | Decision |
|---|---|---|---|
| Grounded smoke and throughput | Does the task reset safely, and how many environments are fastest? | Grounded 8-environment smoke had four-foot support and no immediate failures. Sequential benchmarks measured about 35k, 65k, and 114k simulation steps/s at 1,024, 2,048, and 4,096 environments. | Use 4,096 environments for training. |
| Initial PPO | Can the first reward learn the requested locomotion? | Stopped at update 574 after held-out regression. Model 200 ignored width and mostly stood; model 500 failed all 24 checked trials in about 0.9 s. | Rejected. |
| R2 | Recover survival and strengthen speed/contact/width shaping. | Model 99 completed 48/48 small-grid trials across both gaits, but width response was weak, drift was large, and 0/48 were fully compliant. | Survival recovered; command execution failed. |
| R3 | Add bounded lateral-velocity/yaw-rate costs and world-forward progress. | Model 199 completed 48/48 but 0/48 were compliant. After continuation to model 499, the broad grid completed 288/288, still with 0 compliant; widths remained near 0.25-0.30 m for commands spanning 0.10-0.50 m and speed was too high. | Stable locomotion, weak conditioning. |
| R4 | Sustain locomotion longer and raise speed/contact/width weights. | Midpoint model 700 still had 0 compliant trials, poor narrow-width behavior, and severe drift. Analysis exposed that world-frame width scoring could be improved by turning the body. | Replaced by a body-frame formulation. |
| R5 | Define and score step width in the robot body frame. | Model 1499 reached 288/288 endpoints and accurately changed body-frame width: nominal achieved widths were roughly 0.12/0.28/0.47 m for trot and 0.12/0.29/0.47 m for commands 0.10/0.30/0.50 m. Speed stayed near 0.30 m/s. Duty factor remained nearly fixed, lateral drift was excessive, and 0/288 passed every criterion. | Width control and foot tucking demonstrated; timing and straightness failed. |
| R6 model 1799 | Make phase encoding duty-factor aware and strengthen bounded lateral/yaw-rate costs. | 283/288 endpoints and 21/288 fully compliant. Width/speed response remained good, straightness improved sharply, and achieved duty factor began responding to its command, though unevenly. | First meaningful full-compliance result. |
| R6 model 2099 | Continue the unchanged R6 controller for 300 updates. | Best broad result: 285/288 endpoints and 41/288 compliant. Trot achieved 142/144 endpoints and 40 compliant; walk achieved 143/144 and only 1 compliant. All three failures were narrow pushed trials. Width and speed passed 288/288; duty-factor error, swing recall, and drift caused most failures. | Strongest checkpoint so far, but insufficient overall. |
| R7 | Add one inertial heading-error observation and a bounded heading cost. | 288/288 endpoints but only 13/288 compliant. Trot heading/lateral RMSE worsened; walk heading improved slightly while lateral drift worsened. | Rejected as the straightness solution. |
| R8 | Track world-forward speed and penalize world-lateral velocity more strongly. | The 150-update midpoint had 286/288 endpoints, 0/288 compliant, only 75/288 width passes, two falls, and much worse heading/lateral errors. | Stopped at the midpoint gate and rejected. |
| R9 | Return to R6 model 2099, preserve learned action noise, and try a conservative 25-update planar-velocity/heading transfer. | On the matched nominal DF 0.625 slice, R9 reached 48/48 endpoints and passed width/speed 48/48, but achieved 16/48 full passes versus 23/48 for R6. Trot was 16/24; walk was 0/24. Wide walk remained especially curved. | Rejected; no continuation. |

## What the experiments establish

1. **Direct PPO locomotion works.** A standard MLP, without MPC or a gait
   parameter selector, can drive the Go2 through joint targets and complete long
   flat-ground courses reliably.
2. **Step-width conditioning works.** From R5 onward, the policy changes measured
   body-frame foot separation across 0.10-0.50 m. The narrow command visibly and
   quantitatively tucks the legs while the commanded direction remains forward.
3. **Forward-speed conditioning works at the evaluated core setting.** The later
   policies track the 0.30 m/s evaluation command closely, including R6's 288/288
   speed-compliance checks.
4. **Duty-factor conditioning is only partial.** R6 produces a measurable DF
   response, especially for trot, but requested low/high values remain biased and
   leg dependent. Four-beat walk frequently misses the requested swing schedule.
5. **Trot is substantially more mature than walk.** R6 model 2099 produced 40
   fully compliant trot trials and only one walk trial on the same broad grid.
6. **Endpoint success is not enough.** Several policies reached almost every
   endpoint while turning, drifting, or missing the commanded contact schedule.
   The saved compliance metrics and reference-line videos exposed this difference.
7. **Simple heading penalties did not solve the path problem.** R7-R9 show that
   adding heading feedback or stronger world-frame velocity costs can disturb a
   useful gait and reduce overall compliance. R6 remains the best measured model.

## What the experiments do not establish

- They do not show that narrow physical beams are harder, because every run is
  on flat ground and step width is not beam width.
- They do not show that higher duty factor improves robustness. Actual DF is not
  yet controlled accurately enough across both gaits, and endpoint success is near
  ceiling in many cells.
- They do not support a claim that gait type has no effect. The learned trot is
  currently much more compliant than the learned four-beat walk.
- They do not yet establish period generalization. Period is sampled and supplied
  to the policy, but the reported broad evaluations use 0.48 s; the required held-
  out 0.36 s and 0.54 s checks are unfinished.
- They are development results from one evolving training lineage and eight
  validation seeds per condition. Final test seeds and the planned 64-trial cells
  have not been used, so this is not yet a final publication result.

## Scope-level conclusion

The strongest result is a useful gait-conditioned flat-ground PPO controller that
tracks forward speed and step width, including 0.10 m tucked-foot locomotion, and
executes two-beat trot reasonably well. The unresolved scientific bottlenecks are
four-beat walk contact timing and maintaining the original straight line across
gait/width combinations. Until those are solved, the experiment cannot isolate
the effect of duty factor on robustness. Even after command fidelity is solved,
physical beam-width claims would require restoring a beam evaluation because the
current flat-ground scope contains no beam-width treatment.

Videos for the latest diagnostic are indexed in
`results/r9_nominal_videos/README.md`, including the labeled six-condition grid.
Raw trajectories, manifests, plots, checkpoint provenance, and per-trial metrics
remain in the corresponding `results/validation_*` directories.

## Timing-fixed held-out result (2026-09-11)

The fresh seed-1 timing-fixed PPO checkpoint was evaluated on flat ground with
64 held-out seeds per cell at 0.30 m/s and period 0.48 s. The grid contains five
commanded widths (0.10--0.50 m), trot duty factors 0.50/0.625/0.75, and the
four-beat walk at duty factor 0.75: 20 cells and 1,280 rollouts in total. All
1,280 reached the endpoint, all 20 cells passed the frozen 50 Hz command,
straightness, speed, and width gate, and achieved widths followed the commands.

The new force-robust gate passed 14/20 cells. Trot passed 13/15: duty factor
0.50 at widths 0.10 and 0.50 failed because the 200 Hz force traces contained
extra touchdown/liftoff transitions even though their 50 Hz duty factor and
straight locomotion passed. Walk passed 1/5: width 0.10 passed, while widths
0.20--0.50 failed only at the 10 N contact threshold; their 2 N and 5 N gates
passed. This shows reliable locomotion and command tracking, but the full study
is not ready because contact-event topology is not robust to the preregistered
force thresholds.

These results establish command fidelity at the core speed/period setting; they
do not estimate the paper's closed-loop stability/convergence quantity. The
strict study marker is in the trot result directory as `study_readiness.json`
and reports `ready=false`, `passed_cells=14`.

## Force-fixed seed-2 held-out result (2026-09-11)

The replacement seed-2 policy completed 1,800 updates and 265.4 million samples
with nominal motor gains. Its new held-out evaluation used the same 20-cell,
1,280-rollout protocol. All 1,280 rollouts reached the endpoint and 19/20 cells
passed the complete force-robust command gate, improving on the seed-1 result's
14/20. The only rejected cell was trot at DF 0.50 and 0.10 m stance width: it
tracked speed, width, duty factor, and the straight line, but its touchdown and
liftoff sequence was not robust at the 2/5/10 N thresholds.

The observed gate counts are directionally consistent with the paper for duty
factor and narrow stance: trot passed 4/5 widths at DF 0.50 and 5/5 at DF 0.625
and 0.75, while the 0.10 m width passed 3/4 gait/DF combinations and every wider
width passed 4/4. Matched DF-0.75 trot and walk each passed all five widths.
These remain nominal contact-fidelity comparisons, not the paper's convergence
metric. The speed effect and mechanical cost of transport remain unmeasured
because the preregistered stability/energy grid requires 20/20 gate passage.

The synchronized labeled video is
`results/paper_eval_videos_seed2_20260911/paper_eval_comparison_seed2.mp4`, and
the current claim table is `docs/paper_claim_coverage.md`.
