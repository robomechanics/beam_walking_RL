# Go2 deployment-domain-randomization run

This is a separate deployment candidate. It does not replace or retroactively
change the nominal-gain paper experiment. The policy remains the same 68-input,
12-output feedforward PPO and uses only open flat ground, walk and trot, forward
speed 0.25--0.40 m/s, full stance width 0.10--0.50 m, period 0.36--0.54 s,
and 10% grounded stance starts. There are no pushes.

Training fine-tunes the user-selected trained seed-2 `model_1799.pt` for 1,800
additional PPO updates with 3,072 environments and the existing reward and balanced
command sampler. The exact parent SHA-256 is
`ef5dae663b9650a3197e589b406101d4a450282e1e358c2af895d0505912127a`.
Use `--initialize_from` with that checkpoint. The complete actor, critic and action
noise weights are loaded strictly and checked bitwise before the first update.
The optimizer is fresh for the changed dynamics distribution; DR update numbering
starts at zero, while the parent command curriculum counter is retained. RNG seed
3 controls DR sampling; it is separate from parent training seed 2. Parent identity
is saved in provenance and each checkpoint. The pre-update weights are archived
as `initialized_from_seed2.pt`. The final DR checkpoint is `model_1799.pt` in the
new run directory. The previous from-scratch seed-3 run was stopped at the user's
correction and is not this candidate. Startup randomization is
independent across simulated robots and, for gains, across joints:

- Kp multiplier 0.60--1.40 (nominal 25, hence 15--35);
- Kd multiplier 0.50--1.50 (nominal 0.5, hence 0.25--0.75);
- base-body mass multiplier 0.80--1.20 with inertia recomputed;
- base COM offset x ±0.04 m, y ±0.03 m, z ±0.02 m;
- static friction 0.40--1.50 and dynamic friction 0.30--1.20, constrained so
  dynamic friction does not exceed static friction;
- motor torque-capacity multiplier 0.70--1.10, applied consistently to the
  continuous and stall-torque limits of the DC motor envelope;
- joint friction coefficient 0--0.25 and armature 0--0.02;
- actuator command delay 0--4 physics steps (0--20 ms, or 0--1 policy step),
  sampled independently per simulated robot at reset;
- the inherited bounded Go2 observation noise on base velocity, angular
  velocity, projected gravity, joint position, and joint velocity.

The chronology smoke must verify 68 observations, 12 actions, finite values,
phase/action/reward alignment, reset phase zero, high-rate contact capture,
Kp/Kd samples inside their respective bounds with nonzero spread, observation
corruption enabled, torque strength, joint friction, armature and delay in
bounds with nonzero spread, PhysX readback of mass/inertia/COM/materials, and
no external-force or push event. Grounded calibration and reset poses are
stored separately for every randomized robot instance.

Reset preparation holds the stock joint targets using Kp 80 and Kd 3 for up to 20
seconds, with each instance's randomized mass, COM, friction and motor strength.
After five seconds, each robot must independently satisfy all support, contact,
velocity and tilt checks for 20 consecutive physics steps (100 ms); its pose is
captured at that point. Calibration ends once every robot passes.
The sampled Kp/Kd tensors are restored in a finally block before reset or any
policy step. This prepares supported starts rather than requiring an untrained
zero-action policy to stand indefinitely. The smoke verifies the restored gain
distribution and all four feet loaded on the first policy transition. Per-instance
poses and calibration diagnostics are saved. The policy default pose, action
offsets, DR ranges and calibration acceptance thresholds remain fixed. Deployment
simulation uses two articulation velocity iterations and applies forces every
TGS position iteration to improve contact velocity accuracy under the broad DR.
This solver setting does not add any external disturbance.

After training, use validation seeds beginning at 20,000 with 64 trials per
cell. Evaluate nominal dynamics, held-out randomized dynamics, and the full
cross product Kp scales [0.60, 0.80, 1.00, 1.20, 1.40] and Kd scales
[0.50, 0.75, 1.00, 1.25, 1.50]. Each profile covers all 20 period-.48,
speed-.30 walk/trot/width/duty cells with the existing command, force-topology,
and straightness gates.

Choose the deployment gain candidate lexicographically by: most passed cells;
highest worst-cell compliant-trial fraction; highest mean operational 5 N
topology fraction; lowest mean normalized speed/width/heading error; then
closest-to-nominal tie break. Confirm the selected pair on the untouched test
block beginning at seed 1,500,000. A candidate is ready for port validation
only if nominal validation, randomized validation, selected-gain validation,
and all 20 selected-gain test cells pass. This selection is simulation evidence,
not authorization to command physical hardware. The port must next reproduce
observations and joint targets in quad-sdk/MuJoCo before a restrained hardware
bring-up.
