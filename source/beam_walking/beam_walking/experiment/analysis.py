"""Offline analysis of measured flat-ground gait-command trials."""
import json
from pathlib import Path
import numpy as np

LEGS = ("FL", "FR", "RL", "RR")
OFFSETS = (0., .5, .5, 0.)
PERIOD = .48
DT = .02


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., center - radius), min(1., center + radius)


def complete_cycle_df(times, contacts, offset, eligible=None, period=PERIOD):
    """Integrate end-of-step contact over fully observed cycles after settling.

    Each sample at tick k represents (k-1,k]. Fractional walk phase offsets
    receive interval-overlap weights; the closing boundary must be observed.
    """
    ticks = np.rint(np.asarray(times) / DT).astype(int)
    count = round(period / DT)
    if count < 1 or not np.isclose(count * DT, period):
        raise ValueError("Period must be a positive integer number of control steps")
    if not len(ticks):
        return []
    if not np.all(np.diff(ticks) > 0):
        raise ValueError("Contact times must be strictly increasing")
    values = []
    for cycle in range(int(np.floor(ticks[0] / count + offset)) - 1,
                       int(np.floor(ticks[-1] / count + offset)) + 1):
        lower, upper = (cycle - offset) * count, (cycle + 1 - offset) * count
        if lower < count or lower < ticks[0] - 1 or upper > ticks[-1]:
            continue
        required = np.arange(int(np.floor(lower)) + 1, int(np.ceil(upper)) + 1)
        selected = np.isin(ticks, required)
        if not np.array_equal(ticks[selected], required):
            continue
        if eligible is not None and not np.asarray(eligible)[selected].all():
            continue
        weights = np.minimum(ticks[selected], upper) - np.maximum(ticks[selected] - 1, lower)
        values.append(float(np.sum(np.asarray(contacts)[selected] * weights) / count))
    return values


def validate_manifest(directory):
    """Reject missing cells, mixed checkpoints, and unmatched trial plans."""
    directory = Path(directory)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    frame = manifest.get("step_width_frame", "world")
    if frame not in ("world", "body"):
        raise ValueError("Invalid manifest step_width_frame")
    required = ["expected_conditions", "seeds", "period", "gait", "source_sha256", "checkpoint_sha256"]
    if any(key not in manifest for key in required):
        raise ValueError("Evaluation manifest is missing required run identity")
    cells = manifest["expected_conditions"]
    conditions = [(cell["step_width"], cell["df"], cell["disturbed"]) for cell in cells]
    if len(conditions) != len(set(conditions)):
        raise ValueError("Duplicate evaluation condition in manifest")
    names = [cell["filename"] for cell in cells]
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected filenames must be nonempty and unique")
    if set(names) != {p.name for p in directory.glob("trial_*.npz")}:
        raise ValueError("Measured files do not exactly match the evaluation manifest")
    seeds = np.asarray(manifest["seeds"])
    if not len(seeds) or len(seeds) != len(set(seeds.tolist())):
        raise ValueError("Trial seeds must be nonempty and unique")
    reference, push_reference = None, None
    for cell in cells:
        path = directory / cell["filename"]
        if path.name != cell["filename"]:
            raise ValueError("Manifest filenames must be local basenames")
        with np.load(path) as data:
            data_frame, _, _ = validate_width_frame(data)
            if data_frame != frame:
                raise ValueError(f"Width frame mismatch: {path.name}")
            for key in ["period", "gait", "source_sha256", "checkpoint_sha256"]:
                if key not in data or data[key].item() != manifest[key]:
                    raise ValueError(f"Run identity mismatch: {path.name}, {key}")
            if "speed" in manifest and ("speed" not in data or not np.isclose(data["speed"].item(), manifest["speed"])):
                raise ValueError(f"Run identity mismatch: {path.name}, speed")
            for key in ["step_width", "df", "disturbed"]:
                if key not in data or data[key].item() != cell[key]:
                    raise ValueError(f"Condition mismatch: {path.name}, {key}")
            if not np.array_equal(data["seeds"], seeds):
                raise ValueError(f"Trial seed mismatch: {path.name}")
            if not np.isclose(float(data["control_dt"]), DT):
                raise ValueError("Unsupported contact sampling interval")
            plan_key = "reset_plan" if "reset_plan" in data else "push_plan"
            paired = {key: data[key].copy() for key in ["initial_root", "initial_joints", plan_key]}
            if reference is None:
                reference = paired
            elif any(not np.allclose(paired[key], reference[key], atol=1e-6) for key in paired):
                raise ValueError(f"Matched reset state or push plan mismatch: {path.name}")
            if not cell["disturbed"] and "planned_force" in data and np.any(data["planned_force"]):
                raise ValueError("Nominal condition contains a planned push")
            if cell["disturbed"]:
                if "planned_force" not in data:
                    raise ValueError("Legacy disturbed condition is missing its planned force")
                if push_reference is None:
                    push_reference = data["planned_force"].copy()
                elif not np.allclose(data["planned_force"], push_reference):
                    raise ValueError("Disturbed comparisons require identical planned forces")
    return [directory / name for name in names]


def validate_width_frame(data):
    """Verify declared axes and return independently recomputed body coordinates.

    Missing frame metadata denotes legacy world-axis measurements. Optional
    orientation is wxyz and maps body vectors into the world frame.
    """
    frame = str(np.asarray(data.get("step_width_frame", "world")).item())
    if frame not in ("world", "body"):
        raise ValueError("Invalid step_width_frame")
    if frame == "body" and any(key not in data for key in ("root_quat", "feet_body")):
        raise ValueError("Body width requires root_quat and feet_body")
    if "root_quat" not in data:
        if "feet_body" in data:
            raise ValueError("Cannot verify feet_body without root_quat")
        return frame, None, None
    valid = np.asarray(data["valid"], dtype=bool)
    quat = np.asarray(data["root_quat"])
    feet, body = np.asarray(data["feet"]), np.asarray(data["body"])
    if (quat.shape != valid.shape + (4,) or feet.shape != valid.shape + (4, 3)
            or body.shape != valid.shape + (3,)):
        raise ValueError("Malformed root_quat, feet, or body shape")
    if (not np.isfinite(quat[valid]).all()
            or not np.allclose(np.linalg.norm(quat[valid], axis=-1), 1., atol=1e-4)):
        raise ValueError("root_quat must contain finite unit wxyz quaternions")
    relative = feet - body[..., None, :]
    # Inverse quaternion rotation, using the conjugated vector component.
    vector = -quat[..., None, 1:]
    cross = 2 * np.cross(vector, relative)
    recomputed = relative + quat[..., None, :1] * cross + np.cross(vector, cross)
    if not np.isfinite(recomputed[valid]).all():
        raise ValueError("Nonfinite reconstructed body foot coordinates")
    if "feet_body" in data:
        recorded = np.asarray(data["feet_body"])
        if recorded.shape != feet.shape or not np.allclose(recorded[valid], recomputed[valid], atol=2e-5, rtol=1e-4):
            raise ValueError("feet_body does not match world feet/root_quat rotation")
    w, x, y, z = np.moveaxis(quat, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return frame, recomputed, yaw


def trial_rows(path):
    with np.load(path) as archive:
        d = {key: archive[key] for key in archive.files}
    frame, body_feet, yaw = validate_width_frame(d)
    period, gait = float(d["period"]), str(d["gait"])
    offsets_by_gait = {"trot": (0., .5, .5, 0.), "walk": (0., .5, .25, .75)}
    if gait not in offsets_by_gait or not np.allclose(d["phase_offsets"], offsets_by_gait[gait]):
        raise ValueError(f"Invalid gait or phase offsets: {path}")
    rows = []
    for trial in range(d["valid"].shape[1]):
        mask = d["valid"][:, trial].astype(bool)
        count = int(mask.sum())
        if count == 0 or not mask[:count].all() or mask[count:].any():
            raise ValueError(f"Invalid first-episode mask: {path}, trial {trial}")
        done = d["done"][mask, trial].astype(bool)
        if done[:-1].any() or not done[-1]:
            raise ValueError(f"Incomplete or multiple terminal episodes: {path}, trial {trial}")
        t, body, feet = (d[key][mask, trial] for key in ["time", "body", "feet"])
        ct, requested, cmds = (d[key][mask, trial] for key in ["contacts", "desired", "commands"])
        if not np.allclose(np.diff(t), DT, atol=2e-6) or not np.isfinite(body).all():
            raise ValueError(f"Invalid trajectory samples: {path}")
        if not np.allclose(cmds, cmds[0]) or not np.allclose(cmds[:, 3], period):
            raise ValueError(f"Evaluation commands must remain fixed: {path}")
        if not np.allclose(cmds[:, 1], float(d["df"])) or not np.allclose(cmds[:, 2], float(d["step_width"])):
            raise ValueError(f"Command metadata mismatch: {path}")
        if not np.allclose(cmds[:, 4], ("trot", "walk").index(gait)):
            raise ValueError(f"Gait command metadata mismatch: {path}")
        settled = t >= period
        stance = settled[:, None] & ct
        motion_metrics = {}
        motion_keys = ("body_lateral_velocity", "body_yaw_rate")
        present_motion = [key in d for key in motion_keys]
        if any(present_motion) and not all(present_motion):
            raise ValueError(f"Body lateral velocity and yaw rate must be recorded together: {path}")
        if all(present_motion):
            for key in motion_keys:
                raw = np.asarray(d[key])
                if raw.shape != d["valid"].shape or not np.isfinite(raw[d["valid"]]).all():
                    raise ValueError(f"Malformed {key}: {path}")
                values = raw[mask, trial][settled]
                motion_metrics[f"{key}_mean"] = float(values.mean()) if values.size else np.nan
                motion_metrics[f"{key}_rmse"] = float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan
        if "world_lateral_velocity" in d:
            raw = np.asarray(d["world_lateral_velocity"])
            if raw.shape != d["valid"].shape or not np.isfinite(raw[d["valid"]]).all():
                raise ValueError(f"Malformed world_lateral_velocity: {path}")
            values = raw[mask, trial][settled]
            motion_metrics["world_lateral_velocity_mean"] = float(values.mean()) if values.size else np.nan
            motion_metrics["world_lateral_velocity_rmse"] = float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan
        width_metrics = {}
        for axes, y in [("world", feet[:, :, 1] - body[:, 1:2]),
                        ("body", body_feet[mask, trial, :, 1] if body_feet is not None else None)]:
            width, mae = np.nan, np.nan
            if y is not None:
                leg_y = [float(y[settled & ct[:, j], j].mean()) if (settled & ct[:, j]).any() else np.nan for j in range(4)]
                width = float(np.mean([leg_y[0], leg_y[2]]) - np.mean([leg_y[1], leg_y[3]]))
                errors = np.abs(y - cmds[:, 2:3] * np.array([1, -1, 1, -1]) / 2)
                mae = float(errors[stance].mean()) if stance.any() else np.nan
            width_metrics[f"{axes}_achieved_width"] = width
            width_metrics[f"{axes}_foot_lateral_mae"] = mae
        failure = bool(d["failure"][mask, trial][-1])
        success = bool(d["success"][mask, trial][-1]) and not failure
        row = {
            "file": path.name, "trial": trial, "seed": int(d["seeds"][trial]),
            "step_width": float(d["step_width"]), "command_df": float(d["df"]),
            "period": period, "gait": gait, "disturbed": bool(d["disturbed"]),
            "success": int(success), "failure": int(failure), "timeout": int(not success and not failure),
            "duration": float(t[-1]), "traversal_time": float(t[-1]) if success else np.nan,
            "course_distance": float(np.clip(body[:, 0].max(), 0, 3.35)),
            "forward_distance": float(max(0., body[:, 0].max() + .65)),
            "mean_speed": float(d["speed"][mask, trial].mean()),
            "forward_speed": float(d["forward_velocity"][mask, trial][settled].mean()) if settled.any() else np.nan,
            "settled_body_forward_speed": float(d["speed"][mask, trial][settled].mean()) if settled.any() else np.nan,
            "command_speed": float(cmds[0, 0]), "command_width": float(cmds[0, 2]),
            "step_width_frame": frame,
            "achieved_width": width_metrics[f"{frame}_achieved_width"],
            "foot_lateral_mae": width_metrics[f"{frame}_foot_lateral_mae"],
            **width_metrics,
            "heading_rmse_rad": float(np.sqrt(np.mean(yaw[mask, trial] ** 2))) if yaw is not None else np.nan,
            "max_abs_heading_rad": float(np.abs(yaw[mask, trial]).max()) if yaw is not None else np.nan,
            "settled_samples": int(settled.sum()),
            "lateral_rmse": float(np.sqrt(np.mean(body[:, 1] ** 2))),
            "max_lateral_deviation": float(np.max(np.abs(body[:, 1]))),
            "timing_accuracy": float((ct[settled] == requested[settled]).mean()) if settled.any() else np.nan,
            **motion_metrics,
            "applied_impulse": (float(np.linalg.norm(d["force"][mask, trial], axis=-1).sum() * DT)
                                if "force" in d else 0.0),
            "planned_force": (float(np.linalg.norm(d["planned_force"][trial]))
                              if "planned_force" in d else 0.0),
            "planned_push_duration": (float(d["push_plan"][trial, 5])
                                      if "push_plan" in d else 0.0),
            "planned_push_time": (float(d["push_plan"][trial, 2])
                                  if "push_plan" in d else np.nan),
        }
        for leg, name in enumerate(LEGS):
            for values_key, prefix in [("contacts", "df"), ("desired", "schedule_df")]:
                values = complete_cycle_df(t, d[values_key][mask, trial, leg], float(d["phase_offsets"][leg]), period=period)
                row[f"{prefix}_{name}"] = float(np.mean(values)) if values else np.nan
                if prefix == "df":
                    row[f"cycles_{name}"] = len(values)
            for state, label in [(True, "stance_recall"), (False, "swing_recall")]:
                selected = (requested[:, leg] == state) & settled
                row[f"{label}_{name}"] = float((ct[selected, leg] == state).mean()) if selected.any() else np.nan
        rows.append(row)
    return rows


def analyze(directory):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    files = validate_manifest(directory)
    table = pd.DataFrame([row for path in files for row in trial_rows(path)])
    for key in ["period", "gait", "command_speed", "step_width_frame"]:
        if table[key].nunique() != 1:
            raise ValueError(f"Analyze one fixed {key} per directory")
    table["achieved_df"] = table[[f"df_{x}" for x in LEGS]].mean(axis=1)
    table["df_max_abs_error"] = table[[f"df_{x}" for x in LEGS]].sub(table.command_df, axis=0).abs().max(axis=1)
    table["compliance_pass"] = (
        table[[f"cycles_{x}" for x in LEGS]].ge(1).all(axis=1)
        & table[[f"df_{x}" for x in LEGS]].sub(table.command_df, axis=0).abs().le(.05).all(axis=1)
        & table[[f"swing_recall_{x}" for x in LEGS]].ge(.9).all(axis=1)
        & table[[f"stance_recall_{x}" for x in LEGS]].ge(.9).all(axis=1)
        & table.settled_samples.ge(np.rint(table.period / DT))
        & table.foot_lateral_mae.le(.015)
        & table.lateral_rmse.le(.10)
        & table.max_lateral_deviation.le(.20)
        & (table.achieved_width - table.command_width).abs().le(.03)
        & (table.forward_speed - table.command_speed).abs().le(.04))
    table.to_csv(directory / "trials.csv", index=False)
    records = []
    for (width, df, disturbed), group in table.groupby(["step_width", "command_df", "disturbed"]):
        lo, hi = wilson(int(group.success.sum()), len(group))
        record = {"step_width": width, "command_df": df, "disturbed": disturbed, "n": len(group),
                  "successes": int(group.success.sum()), "failures": int(group.failure.sum()),
                  "timeouts": int(group.timeout.sum()), "success_rate": group.success.mean(),
                  "ci_low": lo, "ci_high": hi, "period": group.period.iloc[0], "gait": group.gait.iloc[0],
                  "step_width_frame": group.step_width_frame.iloc[0]}
        metrics = ["course_distance", "forward_distance", "mean_speed", "forward_speed", "duration",
                   "traversal_time", "achieved_df", "achieved_width", "command_width", "command_speed",
                   "timing_accuracy", "applied_impulse", "df_max_abs_error", "lateral_rmse",
                   "max_lateral_deviation", "foot_lateral_mae", "settled_samples",
                   "world_achieved_width", "world_foot_lateral_mae", "body_achieved_width",
                   "body_foot_lateral_mae", "heading_rmse_rad", "max_abs_heading_rad"]
        metrics += [key for key in ["world_lateral_velocity_mean", "world_lateral_velocity_rmse",
                                    "body_lateral_velocity_mean", "body_lateral_velocity_rmse",
                                    "body_yaw_rate_mean", "body_yaw_rate_rmse"] if key in group]
        for metric in metrics:
            record[metric] = group[metric].mean()
        for leg in LEGS:
            for prefix in ["df", "schedule_df", "stance_recall", "swing_recall"]:
                record[f"{prefix}_{leg}"] = group[f"{prefix}_{leg}"].mean()
            record[f"trials_with_cycles_{leg}"] = int((group[f"cycles_{leg}"] > 0).sum())
        record["compliant_trial_fraction"] = group.compliance_pass.mean()
        record["compliance_screen_pass"] = bool(record["compliant_trial_fraction"] >= .8)
        records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(directory / "summary.csv", index=False)
    rng, contrasts = np.random.default_rng(8102026), []
    for (width, disturbed), group in table.groupby(["step_width", "disturbed"]):
        pivot = group.pivot(index="seed", columns="command_df", values="success")
        if pivot.isna().any().any():
            raise ValueError("Duty-factor comparisons require identical trial seed sets")
        if len(pivot.columns) < 2:
            continue
        low, high = min(pivot.columns), max(pivot.columns)
        diffs = (pivot[high] - pivot[low]).to_numpy()
        bootstrap = diffs[rng.integers(len(diffs), size=(10000, len(diffs)))].mean(axis=1)
        high_ci = wilson(int(pivot[high].sum()), len(pivot), z=2.241402727604947)
        low_ci = wilson(int(pivot[low].sum()), len(pivot), z=2.241402727604947)
        contrasts.append({"step_width": float(width), "disturbed": bool(disturbed), "low_df": float(low), "high_df": float(high),
            "success_difference": float(diffs.mean()), "ci_low": float(np.quantile(bootstrap, .025)),
            "ci_high": float(np.quantile(bootstrap, .975)), "bootstrap_degenerate": bool(np.ptp(bootstrap) == 0),
            "conservative_ci_low": high_ci[0] - low_ci[1], "conservative_ci_high": high_ci[1] - low_ci[0]})
    (directory / "paired_contrasts.json").write_text(json.dumps(contrasts, indent=2))

    def save(fig, name):
        fig.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(directory / f"{name}.{ext}", dpi=180)
        plt.close(fig)

    conditions = sorted(summary.disturbed.unique())
    width_frame = table.step_width_frame.iloc[0]
    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 3.6), squeeze=False)
    for ax, disturbed in zip(axes[0], conditions):
        for width, group in summary[summary.disturbed == disturbed].groupby("step_width"):
            ax.errorbar(group.command_df, group.success_rate,
                        yerr=[group.success_rate - group.ci_low, group.ci_high - group.success_rate],
                        marker="o", capsize=3, label=f"{width:.2f} m")
        ax.set(title="Pushes" if disturbed else "Nominal", xlabel="Commanded DF",
               ylabel="Straight crossing success (95% Wilson CI)", ylim=(-.03, 1.03))
        ax.legend(title=f"{width_frame.title()}-axis step width")
    save(fig, "success_vs_df")
    fig, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 3.8), squeeze=False)
    for ax, disturbed in zip(axes[0], conditions):
        grid = summary[summary.disturbed == disturbed].pivot(index="step_width", columns="command_df", values="success_rate")
        im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for i in range(len(grid)):
            for j in range(len(grid.columns)):
                ax.text(j, i, f"{grid.iloc[i, j]:.0%}", ha="center", color="white" if grid.iloc[i, j] < .55 else "black")
        ax.set(xticks=range(len(grid.columns)), xticklabels=grid.columns, yticks=range(len(grid)),
               yticklabels=grid.index, xlabel="Commanded DF", ylabel=f"{width_frame.title()}-axis step width (m)",
               title="Pushes" if disturbed else "Nominal")
        fig.colorbar(im, ax=ax, label="Success")
    save(fig, "success_heatmap")
    limits = [table.command_df.min() - .02, table.command_df.max() + .02]
    fig, axes = plt.subplots(len(conditions), 4, figsize=(15, 3.7 * len(conditions)), squeeze=False)
    for row, disturbed in enumerate(conditions):
        for width, group in summary[summary.disturbed == disturbed].groupby("step_width"):
            for col, key in enumerate(["achieved_df", "achieved_width", "forward_speed", "lateral_rmse"]):
                axes[row, col].plot(group.command_df, group[key], "o-", label=f"{width:.2f} m")
            axes[row, 1].axhline(width, ls=":", alpha=.4)
        axes[row, 0].plot(limits, limits, "k--", label="Command")
        axes[row, 2].axhline(table.command_speed.iloc[0], color="k", ls="--", label="Command")
        axes[row, 3].axhline(0., color="k", ls="--", label="Straight path")
        for col, label in enumerate(["Achieved DF", f"{width_frame.title()}-axis stance width (m)", "Forward speed (m/s)", "Lateral RMSE (m)"]):
            axes[row, col].set(xlabel="Commanded DF", ylabel=label, title="Pushes" if disturbed else "Nominal")
            axes[row, col].legend(fontsize=7, title=f"{width_frame.title()} width")
    save(fig, "command_compliance")
    fig, axes = plt.subplots(len(conditions), 4, figsize=(13, 3 * len(conditions)), squeeze=False)
    for row, disturbed in enumerate(conditions):
        for col, leg in enumerate(LEGS):
            for width, group in summary[summary.disturbed == disturbed].groupby("step_width"):
                axes[row, col].plot(group.command_df, group[f"df_{leg}"], "o-", label=f"{width:.2f} m")
            axes[row, col].plot(limits, limits, "k--")
            axes[row, col].set(title=f"{leg}, {'push' if disturbed else 'nominal'}", xlabel="Commanded DF", ylabel="Achieved DF")
            axes[row, col].legend(fontsize=7, title=f"{width_frame.title()} width")
    save(fig, "per_leg_df")
    fig, axes = plt.subplots(1, len(conditions), figsize=(7 * len(conditions), 4), squeeze=False)
    for ax, disturbed in zip(axes[0], conditions):
        for path in files:
            with np.load(path) as data:
                if bool(data["disturbed"]) != disturbed:
                    continue
                for trial in range(data["valid"].shape[1]):
                    xy = data["body"][data["valid"][:, trial], trial, :2]
                    ax.plot(xy[:, 0], xy[:, 1], color="tab:blue", alpha=.12, lw=.6)
        ax.axhline(0., color="black", ls="--", label="Commanded straight path")
        ax.set(xlabel="Forward position (m)", ylabel="Lateral displacement (m)", title="Pushes" if disturbed else "Nominal")
        ax.legend()
    save(fig, "straight_path_tracking")
    print(summary.to_string(index=False))
    return summary
