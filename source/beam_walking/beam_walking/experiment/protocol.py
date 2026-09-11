"""Simulator-independent command definitions, also exercised by offline tests."""
import torch

PERIOD = .48
CONTROL_DT = .02
CYCLE_STEPS = round(PERIOD / CONTROL_DT)
OFFSETS = (0., .5, .5, 0.)
GAITS = ("trot", "walk")
# The supplied paper draft reports walk (0,.75,.50,.25) and trot
# (0,.50,.50,0). It does not state a different leg order, so this experiment
# freezes those tuples in the repository order FL,FR,RL,RR. Walk touchdown
# order is FL -> FR -> RL -> RR; trot groups FL+RR -> FR+RL.
GAIT_OFFSETS = (OFFSETS, (0., .75, .50, .25))
WALK_TOUCHDOWN_ORDER = ("FL", "FR", "RL", "RR")
PERIOD_TICKS = tuple(range(18, 28))  # 0.36 through 0.54 s at 50 Hz.
STEP_WIDTH_RANGE = (.10, .50)
SPEED_RANGE = (.25, .40)
DUTY_RANGE = (.50, .75)
SPEED_ANCHORS = (.25, .30, .35, .40)
STEP_WIDTH_ANCHORS = (.10, .20, .30, .40, .50)
DUTY_ANCHORS = (.50, .625, .75)
COMMAND_STRATA = ((0, .50), (0, .625), (0, .75), (1, .75))
STEP_WIDTH_FRAME = "body"
MIN_SWING_STEPS = 5  # >=0.10 s of requested swing on every leg; validate physically.
WIDTHS = (.8, .45, .30, .20)
FOUNDATION_CONTROL_STEPS = 2400  # 50 PPO updates with the 48-step rollout horizon.
CORE_CONTROL_STEPS = 7200        # Expand period/interpolation after 150 updates.
ANCHOR_FRACTION = .75


def advance_phase_ticks(ticks, period_ticks, reset_mask):
    """Advance continuing environments and keep freshly reset ones at phase zero."""
    if ticks.shape != period_ticks.shape or ticks.shape != reset_mask.shape:
        raise ValueError("Phase ticks, periods, and reset mask must have identical shapes")
    # Periods are validated when commands are constructed. Avoid a GPU-to-CPU
    # synchronization in this per-control-step helper.
    advanced = (ticks + 1) % period_ticks
    return torch.where(reset_mask, torch.zeros_like(advanced), advanced)


def validate_scientific_gait_duties(gait, duties):
    """Reject unsupported four-beat walk cells in confirmatory evaluation."""
    if gait not in GAITS:
        raise ValueError(f"Unknown gait: {gait}")
    if gait == "walk" and any(abs(float(duty) - .75) > 1e-7 for duty in duties):
        raise ValueError("Four-beat walk is confirmatory only at duty factor 0.75")


def leg_phase(ticks, period_ticks=None, gait=None):
    if period_ticks is None:
        period_ticks = torch.full_like(ticks, CYCLE_STEPS)
    if gait is None:
        gait = torch.zeros_like(ticks)
    quarters = torch.tensor(GAIT_OFFSETS, device=ticks.device).mul(4).long()[gait]
    # Keep modulo arithmetic integer even where a quarter cycle is between ticks.
    numerator = (4 * ticks[:, None] + quarters * period_ticks[:, None]) % (4 * period_ticks[:, None])
    return numerator / (4 * period_ticks[:, None])


def discrete_stance_fraction(duty, period_ticks, gait):
    """Actual requested fraction at control samples, per leg; includes tick rounding."""
    quarters = torch.tensor(GAIT_OFFSETS, device=duty.device).mul(4).long()[gait]
    ticks = torch.arange(max(PERIOD_TICKS), device=duty.device)
    phase = ((4 * ticks[None, :, None] + quarters[:, None] * period_ticks[:, None, None])
             % (4 * period_ticks[:, None, None])) / (4 * period_ticks[:, None, None])
    valid = ticks[None, :] < period_ticks[:, None]
    return ((phase < duty[:, None, None]) & valid[:, :, None]).sum(1) / period_ticks[:, None]


def _balanced_indices(count, levels, device):
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    start = torch.randint(levels, (1,), device=device)
    values = (torch.arange(count, device=device) + start) % levels
    return values[torch.randperm(count, device=device)]


def sample_training_commands(count, common_control_step, device="cpu"):
    """Sample staged, edge-covered commands from physically meaningful gait regimes.

    Trot is trained across duty factors 0.50--0.75. Four-beat walk is trained at
    duty factor 0.75, its minimum no-flight value and the only duty factor shared
    with trot in this study. Stage 2 adds continuous speed, width, and period
    coverage without encoding any desired stability outcome.
    """
    values = torch.empty((count, 5), device=device)
    period_ticks = torch.full(
        (count,), CYCLE_STEPS, device=device, dtype=torch.long)
    if common_control_step < FOUNDATION_CONTROL_STEPS:
        stratum = _balanced_indices(count, len(COMMAND_STRATA), device)
        strata = torch.tensor(COMMAND_STRATA, device=device)
        values[:] = torch.tensor([.30, .625, .30, PERIOD, 0.], device=device)
        values[:, 4] = strata[stratum, 0]
        values[:, 1] = strata[stratum, 1]
        return values, period_ticks, 0

    core_count = (
        count if common_control_step < CORE_CONTROL_STEPS
        else round(ANCHOR_FRACTION * count)
    )
    # Balance the four paper-facing gait/DF strata exactly. This gives matched
    # walk and trot at DF=.75 equal exposure without favoring an outcome sign.
    stratum = _balanced_indices(core_count, len(COMMAND_STRATA), device)
    strata = torch.tensor(COMMAND_STRATA, device=device)
    if core_count:
        values[:core_count, 3] = PERIOD
        values[:core_count, 4] = strata[stratum, 0]
        values[:core_count, 1] = strata[stratum, 1]
        for stratum_index in range(len(COMMAND_STRATA)):
            ids = (stratum == stratum_index).nonzero().flatten()
            combinations = len(SPEED_ANCHORS) * len(STEP_WIDTH_ANCHORS)
            combo = _balanced_indices(len(ids), combinations, device)
            width = combo % len(STEP_WIDTH_ANCHORS)
            speed = combo // len(STEP_WIDTH_ANCHORS)
            values[ids, 0] = torch.tensor(SPEED_ANCHORS, device=device)[speed]
            values[ids, 2] = torch.tensor(
                STEP_WIDTH_ANCHORS, device=device)[width]

    remaining = count - core_count
    if remaining:
        target = slice(core_count, count)
        stratum = _balanced_indices(remaining, len(COMMAND_STRATA), device)
        strata = torch.tensor(COMMAND_STRATA, device=device)
        gait = strata[stratum, 0].long()
        values[target, 0] = (
            SPEED_RANGE[0]
            + (SPEED_RANGE[1] - SPEED_RANGE[0])
            * torch.rand(remaining, device=device)
        )
        # DF=.75 walk needs at least five swing samples, hence >=20 ticks.
        minimum_ticks = torch.where(
            gait == GAITS.index("walk"),
            torch.full_like(gait, 20),
            torch.full_like(gait, min(PERIOD_TICKS)),
        )
        span = max(PERIOD_TICKS) - minimum_ticks + 1
        period_ticks[target] = (
            minimum_ticks
            + torch.floor(torch.rand(remaining, device=device) * span).long()
        )
        values[target, 3] = period_ticks[target] * CONTROL_DT
        values[target, 4] = gait.to(values.dtype)
        max_df = torch.minimum(
            torch.full((remaining,), DUTY_RANGE[1], device=device),
            1. - MIN_SWING_STEPS / period_ticks[target],
        )
        # Equal-probability Voronoi bins around the three trot anchors preserve
        # continuous DF coverage without changing stratum exposure.
        lower = torch.tensor([.50, .5625, .6875, .75], device=device)[stratum]
        upper = torch.tensor([.5625, .6875, .75, .75], device=device)[stratum]
        upper = torch.minimum(upper, max_df)
        sampled_df = lower + (upper - lower) * torch.rand(remaining, device=device)
        values[target, 1] = torch.where(
            gait == GAITS.index("walk"),
            torch.full_like(sampled_df, .75), sampled_df)
        values[target, 2] = (
            STEP_WIDTH_RANGE[0]
            + (STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0])
            * torch.rand(remaining, device=device)
        )
        order = torch.randperm(count, device=device)
        values, period_ticks = values[order], period_ticks[order]
    return (
        values, period_ticks,
        1 if common_control_step < CORE_CONTROL_STEPS else 2,
    )


def contact_score(actual, desired, duty):
    """Duration-balanced contact score with no duty-factor preference."""
    # Pass realized per-leg schedule duty to account for finite control ticks.
    if duty.ndim == 1:
        duty = duty[:, None]
    return .5 * ((actual & desired).float() / duty
                 + (~actual & ~desired).float() / (1 - duty)).mean(dim=1)


def robust_contact_score(force_norm, desired, duty, contact_threshold=5.,
                         stance_floor=15., swing_ceiling=2.):
    """Score scheduled contact at physics rate with bounded force margins.

    ``force_norm`` is ``[environment, substep, leg]``.  Stance and swing are
    duration balanced by the realized per-leg schedule duty, so an ideal gait
    receives the same cycle-average score for every supported command.
    """
    if force_norm.ndim != 3 or force_norm.shape[-1] != 4:
        raise ValueError("Contact forces must have shape [environment, substep, 4]")
    if desired.shape != (force_norm.shape[0], force_norm.shape[2]):
        raise ValueError("Desired contacts must have shape [environment, 4]")
    if duty.ndim == 1:
        duty = duty[:, None]
    if duty.shape != desired.shape:
        raise ValueError("Realized schedule duty must match desired contacts")
    if not (swing_ceiling < contact_threshold < stance_floor):
        raise ValueError("Force margins must satisfy swing < contact < stance")

    desired = desired[:, None, :]
    duty = duty[:, None, :]
    contact = force_norm > contact_threshold
    hard = .5 * (
        (contact & desired).to(force_norm.dtype) / duty
        + (~contact & ~desired).to(force_norm.dtype) / (1 - duty)
    )
    stance_quality = ((force_norm - contact_threshold)
                       / (stance_floor - contact_threshold)).clamp(0., 1.)
    swing_quality = ((contact_threshold - force_norm)
                      / (contact_threshold - swing_ceiling)).clamp(0., 1.)
    margin = .5 * (
        desired.to(force_norm.dtype) * stance_quality / duty
        + (~desired).to(force_norm.dtype) * swing_quality / (1 - duty)
    )
    return .5 * hard.mean(dim=(1, 2)) + .5 * margin.mean(dim=(1, 2))


def normalized_gait_command(values):
    """Normalize measurable commands and one-hot encode the categorical gait."""
    if values.shape[-1] != 5:
        raise ValueError("Commands must be [speed, duty factor, width, period, gait]")
    speed = 2 * (values[..., 0:1] - SPEED_RANGE[0]) / (SPEED_RANGE[1] - SPEED_RANGE[0]) - 1
    duty = 2 * (values[..., 1:2] - DUTY_RANGE[0]) / (DUTY_RANGE[1] - DUTY_RANGE[0]) - 1
    width = 2 * (values[..., 2:3] - STEP_WIDTH_RANGE[0]) / (STEP_WIDTH_RANGE[1] - STEP_WIDTH_RANGE[0]) - 1
    period_range = (min(PERIOD_TICKS) * CONTROL_DT, max(PERIOD_TICKS) * CONTROL_DT)
    period = 2 * (values[..., 3:4] - period_range[0]) / (period_range[1] - period_range[0]) - 1
    gait = values[..., 4].long()
    if torch.any((gait < 0) | (gait >= len(GAITS))):
        raise ValueError("Gait ID must select trot or walk")
    one_hot = torch.nn.functional.one_hot(gait, num_classes=len(GAITS)).to(values.dtype)
    return torch.cat([speed, duty, width, period, one_hot], dim=-1)


def duty_warped_phase(phase, duty):
    """Encode commanded stance and swing into equal observation half-cycles.

    Contact schedules and rewards continue to use ``phase``.  This deterministic,
    invertible observation transform makes commanded liftoff occur at encoded
    phase 0.5 for every duty factor without adding policy inputs.
    """
    if duty.ndim == phase.ndim - 1:
        duty = duty.unsqueeze(-1)
    if duty.shape != phase.shape and duty.shape != phase.shape[:-1] + (1,):
        raise ValueError("Duty factor must broadcast across the leg phase dimension")
    # Command construction already enforces 0 < duty < 1.  Avoid a per-control-step
    # GPU-to-CPU synchronization here; this function is part of every observation.
    return torch.where(phase < duty,
                       .5 * phase / duty,
                       .5 + .5 * (phase - duty) / (1 - duty))


def width_score(lateral_error):
    """All-foot squared error before the Gaussian prevents sacrificing two feet."""
    return torch.exp(-lateral_error.square().mean(dim=1) / .01)


def world_to_body(displacements, root_quaternion):
    """Inverse unit-wxyz rotation for batched [environment, foot, xyz] vectors."""
    qvec = root_quaternion[:, None, 1:].expand_as(displacements)
    cross = torch.linalg.cross(qvec, displacements, dim=-1)
    return (displacements - 2 * root_quaternion[:, None, :1] * cross
            + 2 * torch.linalg.cross(qvec, cross, dim=-1))


def clearance_score(height_error, desired, duty):
    """Positive swing tracking, normalized by each leg's sampled swing duration."""
    return (torch.exp(-height_error.square() / .0016)
            * ~desired / (1 - duty)).mean(dim=1)


def fore_aft_target(phase, duty, speed, period):
    """Body-frame foot-x target for stance sweep and swing recovery."""
    if duty.ndim == phase.ndim - 1:
        duty = duty.unsqueeze(-1)
    # A no-slip stance foot sweeps relative to the body for D*T seconds.
    step_length = (speed * duty.squeeze(-1) * period).unsqueeze(-1)
    stance_progress = phase / duty
    swing_progress = (phase - duty) / (1 - duty)
    offset = torch.where(phase < duty, .5 - stance_progress, swing_progress - .5)
    hip_x = torch.tensor([.1934, .1934, -.1934, -.1934],
                         device=phase.device, dtype=phase.dtype)
    return hip_x + step_length * offset


def fore_aft_score(error):
    """All-foot body-frame fore-aft placement score."""
    return torch.exp(-error.square().mean(dim=1) / .0025)


def straight_motion_cost(yaw_rate):
    """Bounded body turning cost; course velocity is scored separately."""
    return 1.0 - torch.exp(-(yaw_rate / .20).square())


def heading_stabilization_cost(heading_error):
    """Bounded rational cost that retains slope beyond small heading errors."""
    squared = heading_error.square()
    return 2.0 * squared / (squared + .20 ** 2)


def speed_score(error):
    """Commanded-speed tracking with a 0.10 m/s Gaussian scale."""
    return torch.exp(-(error / .10).square())


def planar_speed_score(forward_error, world_lateral_velocity):
    """Track the fixed-course velocity vector using one bounded score."""
    return torch.exp(-(forward_error.square() + world_lateral_velocity.square()) / .01)
