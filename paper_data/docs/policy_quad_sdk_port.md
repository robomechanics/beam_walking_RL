# Porting the seed-2 policy to quad-sdk

Inspected 2026-09-11. This guide describes the evaluated checkpoint
`results/paper_ppo_forcefix_nominalgains_seed2_3072_20260911/model_1799.pt`.
Its actor weights were inspected on CPU: 68 → 256 → 128 → 128 → 12,
with ELU hidden activations and a linear output. Observation normalization is
disabled. No ONNX export of this policy was found in this repository.

The requested upstream target is
<https://github.com/robomechanics/quad-sdk/tree/devel_ros2>.
Implementation details below were checked against the available local checkout
at `/home/rml2/ros2_ws/src/quad-sdk`, branch `devel_ros2_sys_id_thomas`;
that branch can differ from upstream.

## Integration point

Add a dedicated `GaitPolicy` derived from `LegController`, using
`robot_driver/src/controllers/learned_velocity_policy.cpp` as the ONNX/session
and motor-message template. Register a distinct controller ID such as
`gait_policy` in `robot_driver.cpp` and add its source to the ONNX-enabled
section of `robot_driver/CMakeLists.txt`.

The existing learned controller constructs 45 observations, multiplies angular
velocity by 0.2 and joint velocity by 0.05, and builds a different nominal pose.
Changing only `robot_driver.model_path` will not work for this checkpoint.
Audit controller-specific command forwarding, initialization and estimator
selection when adding the new ID; updating only the constructor switch is
insufficient.

Data flow:

```text
RobotState + measured foot contacts + gait command + phase/previous-action state
    -> float32 observation [1,68]
    -> ONNX actor
    -> action [1,12]
    -> clip, scale, add nominal pose, reorder
    -> LegCommandArray -> existing driver PD/motor path
```

Run inference at exactly 50 Hz in simulation time. Hold the resulting joint
targets between inference ticks while the driver's faster PD loop continues.
Training used four 5 ms physics/actuator steps per 20 ms policy step. A different
simulator/driver actuator update rate is a plant difference to validate.

## Exact input contract

Indices below use Python's half-open slice convention. Every value is float32;
there is no observation history or recurrent hidden-state tensor.

| Slice | Contents |
|---|---|
| `0:3` | Body-frame base linear velocity, m/s |
| `3:6` | Body-frame base angular velocity, rad/s |
| `6:9` | Unit gravity vector in body frame: `R_world_body.T @ [0,0,-1]` |
| `9:21` | Joint positions minus `q_default`, rad, in policy order |
| `21:33` | Joint velocities, rad/s, in policy order; default velocities are zero |
| `33:45` | Previous clipped action, before the 0.25 scale and pose offset |
| `45:49` | Sine of four duty-warped leg phases |
| `49:53` | Cosine of four duty-warped leg phases |
| `53:57` | Normalized speed, duty factor, full stance width, period |
| `57:59` | Gait one-hot: trot `[1,0]`, walk `[0,1]` |
| `59:63` | Four desired contacts from raw phase `< duty_factor` |
| `63` | Signed yaw relative to the fixed course heading, wrapped to ±pi |
| `64:68` | Four measured contacts: latest foot-force norm `> 5 N` in training |

All four-leg vectors use FL, FR, RL, RR. Gravity is a unit vector, not raw IMU
acceleration and not a vector of magnitude 9.81. Do not apply the existing
controller's 0.2/0.05 velocity scales or a running observation normalizer.

The five physical commands are speed (m/s), duty factor, full left-right foot
separation (m), period (s), and gait. The four scalar normalized commands are:

```python
normalized = [
    2 * (speed - 0.25) / 0.15 - 1,
    2 * (duty - 0.50) / 0.25 - 1,
    2 * (width - 0.10) / 0.40 - 1,
    2 * (period - 0.36) / 0.18 - 1,
]
```

Supported training commands: speed 0.25–0.40, width 0.10–0.50,
period 0.36–0.54 in 0.02 increments; trot duty 0.50–0.75, walk duty 0.75.
Require at least five requested swing samples per leg. In particular, walk
needs period >=0.40. No lateral or yaw command input exists. Use a separate
stand/stop controller for zero velocity; zero is outside this policy's training
speed range. For initial port comparison use trot, speed .30, duty .625,
width .30, period .48.

## Gait clock

Maintain an integer tick `k`, initially zero, and `N = round(period / .02)`.
The following NumPy expresses the exact integer phase arithmetic to translate
to C++:

```python
quarters = np.array([0, 2, 2, 0] if gait == "trot" else [0, 3, 2, 1])
phase = (((4 * k + quarters * N) % (4 * N)) / (4 * N)).astype(np.float32)
desired = (phase < duty).astype(np.float32)
warped = np.where(phase < duty,
                  0.5 * phase / duty,
                  0.5 + 0.5 * (phase - duty) / (1 - duty))
sin_phase = np.sin(2 * np.pi * warped)
cos_phase = np.cos(2 * np.pi * warped)
```

Build the observation for tick k and apply its action for the following 20 ms.
Only then advance k to `(k+1) % N`. Commit gait/period/command changes at master
cycle boundaries, matching training. Reset previous action and phase on a new
episode/controller activation, not on every driver iteration. Measured contacts
and desired contacts are distinct inputs. The training reset temporarily clears
the measured contact cache until the first physics update; account for this
when constructing exact replay comparisons.

## Joint order and action conversion

The policy order is:

```text
FL_hip FR_hip RL_hip RR_hip
FL_thigh FR_thigh RL_thigh RR_thigh
FL_calf FR_calf RL_calf RR_calf
```

The nominal pose in that order is:

```python
q_default = np.array([.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5])
```

Use this default pose for BOTH relative position observations and action
offsets. The grounded-reset calibration in `settled_stance.json` is not the
action offset. The SDK's single three-angle stand parameter cannot represent
this full pose.

The inspected SDK uses leg-major order FL, RL, FR, RR, each containing
abduction, hip, knee. With that exact ordering:

```python
# SDK -> policy, for both q and dq:
policy_from_sdk = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
# Policy -> SDK, for output joint targets:
sdk_from_policy = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

action = np.clip(onnx_output.reshape(12), -5., 5.)
q_target_policy = q_default + 0.25 * action
q_target_sdk = q_target_policy[sdk_from_policy]
previous_action = action.copy()
```

Verify actual joint names, axes and signs for the selected Go2 model before
relying on positional indices. The output is neither torque nor absolute angle.
Populate `MotorCommand.pos_setpoint` with `q_target_sdk`, `vel_setpoint=0`,
`torque_ff=0`, `kp=25`, `kd=.5`. Preserve the driver's torque limiting path;
training additionally used the Isaac Lab DC motor torque-speed envelope with
23.5 Nm effort/saturation and 30 rad/s velocity parameters. Compare that envelope
to the destination plant instead of assuming equal PD gains make them identical.

## State estimation and contacts: required plumbing

The local driver's learned-policy hardware fallback without mocap fills IMU
orientation/gyro and joint state but does not supply base linear velocity.
This 68D actor requires a real velocity estimate. Route a valid estimator or
mocap state to the new controller; do not silently fill velocity with zeros.

Convert world linear velocity to body coordinates using the orientation.
Check the angular-velocity convention of the selected state source: gyro data
is already body-frame. Do not rotate it twice. For quaternion construction,
ROS fields are x/y/z/w while Eigen's scalar constructor takes w/x/y/z.
Keep the heading reference fixed after activation; continuously re-zeroing yaw
would erase the policy's heading feedback.

Use measured simulator `state/grfs` or the hardware `state/foot_contact`
pipeline, reordered `[0,2,1,3]` from SDK legs to policy legs. Do not substitute
planner contact schedules or commanded GRFs. Confirm freshness and provenance
before using `RobotState.feet[].contact`.

The local simulator contact threshold defaults to 5 N. Hardware `FootContact`
contains raw int16 Unitree readings and defaults to threshold 30 raw units;
these are not automatically the same as a 5 N simulated force norm. Hardware
contact calibration remains necessary for a matched observation interface.

## Export and validation sequence

1. Export the deterministic actor from the named checkpoint on CPU, preserving
   its ELU layers and linear output. Export one float32 input `obs` shaped
   `[1,68]` and output `actions` shaped `[1,12]`. Do not include the critic or
   exploration noise. The existing generic `scripts/rsl_rl/play.py` launches a
   simulator, so a CPU-only exporter is preferable for this step.
2. Check ONNX against the PyTorch actor on identical saved observations; record
   maximum absolute error. Separately compare the C++ observation builder and
   motor-target conversion with this repository's Python implementation,
   including cycle wraps, both gaits, reset, and boundary command changes.
3. Configure the new controller's model path, inference rate 50 Hz, complete
   12-joint default pose, gains and gait command parameters. Check ONNX shapes
   and finite values at load/inference. Reuse the existing state-machine and
   command freshness checks. Ensure only the selected controller writes motors.
   The existing implementation requires fresh velocity commands within 0.1 s;
   parameters alone do not satisfy that watchdog. Give the new gait command
   interface an explicit freshness/enable path.
4. After the required independent council review and GPU/RAM preflight,
   compare flat-ground Go2 rollouts using matched commands and logged
   observations/actions. Confirm speed, width, contact duty, heading and joint
   tracking before expanding the command range. No simulator was launched for
   this guide.

CPU ONNX Runtime is a reasonable first backend for this small network; measure
latency. The existing local loader unconditionally requests CUDA, so a CPU
configuration needs to omit that provider request. Do not allocate/reload the
model on each control iteration.

The seed-2 policy passed 20/20 nominal command/contact cells in Isaac Lab at
speed .30 and period .48. This does not establish portability to another plant
or hardware. Its stability/chi estimates remain invalid/suppressed due to the
documented estimator noise; those values should not guide this port.
