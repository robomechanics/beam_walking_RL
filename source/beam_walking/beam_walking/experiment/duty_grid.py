"""Summaries for the fixed duty-factor grid used to label the selector."""

import numpy as np

from .analysis import schedule_centered_topology_fraction
from .stability import command_fidelity, mechanical_cot


def summarize_condition(payload, *, step_width, speed, period, gait,
                        command_df, seed_start, topology_min=.90):
    """Return one selector-training row per independently reset trial."""
    contacts = np.asarray(payload["contacts"], dtype=bool)
    desired = np.asarray(payload["desired"], dtype=bool)
    feet = np.asarray(payload["feet_body"], dtype=float)
    trials, cycles, steps, legs = contacts.shape
    if legs != 4 or desired.shape != contacts.shape:
        raise ValueError("Grid contacts and desired schedules have invalid shapes")

    flattened = {
        "nominal_contacts": contacts.reshape(trials, cycles * steps, 4),
        "nominal_desired": desired.reshape(trials, cycles * steps, 4),
        "nominal_feet_body": feet.reshape(trials, cycles * steps, 4, 3),
    }
    for output, source in (
            ("nominal_forward_velocity", "forward_velocity"),
            ("nominal_lateral_position", "lateral_position"),
            ("nominal_heading", "heading"),
            ("nominal_world_lateral_velocity", "world_lateral_velocity"),
            ("nominal_body_yaw_rate", "body_yaw_rate"),
            ("nominal_failure", "failure")):
        flattened[output] = np.asarray(payload[source]).reshape(
            trials, cycles * steps)

    rows = []
    sample_dt = float(payload["sample_dt"])
    for trial in range(trials):
        fidelity = command_fidelity(
            flattened, trial, speed, command_df, step_width)
        topology = []
        for cycle in range(cycles):
            actual = np.asarray(
                payload["substep_contacts"][trial, cycle], dtype=bool
            ).reshape(-1, 4)
            target = np.repeat(desired[trial, cycle],
                               actual.shape[0] // steps, axis=0)
            times = np.arange(1, len(actual) + 1) * sample_dt
            try:
                fraction, _, _ = schedule_centered_topology_fraction(
                    times, actual, target, gait, period=period,
                    sample_dt=sample_dt, tolerance=.02,
                    initial_contacts=payload["initial_contacts"][trial, cycle],
                    initial_desired=payload["initial_desired"][trial, cycle],
                    circular=True,
                )
            except ValueError:
                # A reset or malformed event sequence makes this trial
                # ineligible; retain it as a failed trial instead of aborting
                # the other independent trials in the condition.
                fraction = float("nan")
            topology.append(fraction)
        topology_fraction = (
            float(np.mean(topology))
            if np.isfinite(topology).all() else float("nan"))
        done = bool(np.asarray(payload["done"])[trial].any())
        any_failure = bool(np.asarray(
            payload.get("any_failure", flattened["nominal_failure"].any(axis=1))
        )[trial])
        energy_valid = True
        try:
            energy = mechanical_cot(
                payload["applied_torque"][trial:trial + 1],
                payload["joint_velocity"][trial:trial + 1],
                payload["x_boundaries"][trial:trial + 1],
                float(payload["robot_mass_kg"]),
                float(payload["gravity_mps2"]), sample_dt=sample_dt,
            )
            positive_cot = float(np.median(energy["positive_mechanical_cot"]))
            absolute_cot = float(np.median(energy["absolute_mechanical_cot"]))
        except ValueError:
            energy_valid = False
            positive_cot = absolute_cot = float("nan")
        compliant = bool(
            fidelity["command_gate_pass"] and not done and not any_failure
            and np.isfinite(topology_fraction)
            and topology_fraction >= topology_min and energy_valid)
        rows.append({
            "seed": seed_start + trial,
            "step_width": step_width, "speed": speed,
            "period": period, "gait": gait, "command_df": command_df,
            "compliant": int(compliant),
            "any_failure": int(any_failure),
            "command_gate_pass": fidelity["command_gate_pass"],
            "topology_fraction": topology_fraction,
            "topology_gate_pass": int(topology_fraction >= topology_min),
            "energy_valid": int(energy_valid),
            "positive_mechanical_cot": positive_cot,
            "absolute_mechanical_cot": absolute_cot,
            **{key: value for key, value in fidelity.items()
               if key != "command_gate_pass"},
        })
    return rows
