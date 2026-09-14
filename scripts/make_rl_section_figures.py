"""Figures and source tables for the learned-locomotion (RL) paper section.

Outputs go to ``PAPER_GRAPHS/rl_section``. The push-disturbance tables are
computed from the local evaluation archives under ``results/`` (which are not
tracked); the CSVs written here are tracked so the figures can be regenerated
with ``--from-csv`` on a machine without the archives. No simulator or GPU is
used.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PAPER_GRAPHS" / "rl_section"
RESULTS = ROOT / "results"
PUSH_RUNS = {
    ("narrow", "nominal"): "validation_tight3_nominal_20260913",
    ("narrow", "perturbed"): "validation_tight3_perturbed_20260913",
    ("push_trained", "nominal"): "validation_tight3perturb_nominal_20260913",
    ("push_trained", "perturbed"): "validation_tight3perturb_perturbed_20260913",
}
POLICY_LABEL = {"narrow": "Narrow-stance policy", "push_trained": "Push-trained policy"}
CONTROL_DT = .02
SETTLE_S = 2.0

# Validated reference palette (dataviz skill): ordinal blue ramp for ordered
# duty-factor levels, categorical slots 1/2 for the two policies, orange for walk.
BLUE = {"250": "#86b6ef", "400": "#3987e5", "450": "#2a78d6", "550": "#1c5cab", "650": "#104281"}
ORANGE = "#eb6834"
GREY = "#8a8985"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
DF_COLOR = {0.5: BLUE["250"], 0.625: BLUE["450"], 0.75: BLUE["650"]}
DF_MARKER = {0.5: "o", 0.625: "s", 0.75: "^"}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": GREY,
    "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": "#e4e3df",
    "grid.linewidth": .6, "legend.frameon": False, "pdf.fonttype": 42,
})


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def trial_metrics(path):
    """Per-condition outcome counts plus realized DF and stance width."""
    with np.load(path) as d:
        valid = d["valid"].astype(bool)
        failure = d["failure"].astype(bool)
        success = d["success"].astype(bool)
        time = d["time"]
        contacts = d["contacts"]
        feet_y = d["feet_body"][:, :, :, 1]
        body_y = d["body"][:, :, 1]
        speed = d["forward_velocity"]
        envelope = d["perturbation_envelope"] if "perturbation_envelope" in d else None
    trials = valid.shape[1]
    last = np.array([np.flatnonzero(valid[:, i])[-1] for i in range(trials)])
    idx = (last, np.arange(trials))
    failed = failure[idx]
    succeeded = success[idx] & ~failed
    settled = valid & (time > SETTLE_S)
    df, width, lateral, forward = [], [], [], []
    for i in range(trials):
        m = settled[:, i]
        df.append(contacts[m, i].mean())
        y, c = feet_y[m, i], contacts[m, i]
        pairs = []
        for left, right in ((0, 1), (2, 3)):
            if c[:, left].any() and c[:, right].any():
                pairs.append(y[c[:, left], left].mean() - y[c[:, right], right].mean())
        width.append(np.mean(pairs))
        lateral.append(np.sqrt((body_y[m, i] ** 2).mean()))
        forward.append(speed[m, i].mean())
    row = {
        "trials": trials, "success": int(succeeded.sum()), "failure": int(failed.sum()),
        "timeout": int(trials - succeeded.sum() - failed.sum()),
        "realized_df": float(np.mean(df)), "realized_width_m": float(np.mean(width)),
        "lateral_rmse_m": float(np.mean(lateral)), "forward_speed_mps": float(np.mean(forward)),
        "mean_failure_time_s": float((last[failed] * CONTROL_DT).mean()) if failed.any() else "",
    }
    if envelope is not None and failed.any():
        row["mean_envelope_at_failure"] = float(envelope[idx][failed].mean())
    else:
        row["mean_envelope_at_failure"] = ""
    return row, last, failed


PUSH_FIELDS = ["policy", "condition", "command_width_m", "command_df", "trials", "success",
               "failure", "timeout", "success_rate", "ci_low", "ci_high", "realized_df",
               "realized_width_m", "lateral_rmse_m", "forward_speed_mps",
               "mean_failure_time_s", "mean_envelope_at_failure"]


def collect_push_tables():
    rows, survival = [], {}
    for (policy, condition), run in PUSH_RUNS.items():
        directory = RESULTS / run
        ends, fails = [], []
        for path in sorted(directory.glob("trial_*.npz")):
            parts = path.stem.split("_")
            width, df = float(parts[1][1:]), float(parts[2][1:])
            row, last, failed = trial_metrics(path)
            lo, hi = wilson(row["success"], row["trials"])
            rows.append({"policy": policy, "condition": condition, "command_width_m": width,
                         "command_df": df, "success_rate": row["success"] / row["trials"],
                         "ci_low": lo, "ci_high": hi, **row})
            ends.extend(last.tolist())
            fails.extend(failed.tolist())
        if condition == "perturbed":
            ends, fails = np.array(ends), np.array(fails)
            t = np.arange(0, ends.max() + 2) * CONTROL_DT
            alive = np.array([1 - ((ends < k) & fails).mean() for k in range(len(t))])
            survival[policy] = (t, alive)
    with (OUT / "push_study_conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PUSH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PUSH_FIELDS})
    with (OUT / "push_study_survival.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["policy", "time_s", "fraction_not_fallen"])
        for policy, (t, alive) in survival.items():
            for ti, ai in zip(t, alive):
                writer.writerow([policy, f"{ti:.2f}", f"{ai:.6f}"])
    return rows, survival


def read_push_tables():
    rows = []
    with (OUT / "push_study_conditions.csv").open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            for key in PUSH_FIELDS[2:]:
                if key in ("policy", "condition"):
                    continue
                row[key] = float(raw[key]) if raw[key] != "" else ""
            rows.append(row)
    survival = {}
    with (OUT / "push_study_survival.csv").open(newline="") as stream:
        for raw in csv.DictReader(stream):
            survival.setdefault(raw["policy"], ([], []))
            survival[raw["policy"]][0].append(float(raw["time_s"]))
            survival[raw["policy"]][1].append(float(raw["fraction_not_fallen"]))
    survival = {k: (np.array(v[0]), np.array(v[1])) for k, v in survival.items()}
    return rows, survival


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def figure_push_success(rows):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, policy in zip(axes, ("narrow", "push_trained")):
        for df in (0.5, 0.625, 0.75):
            cells = sorted((r for r in rows if r["policy"] == policy and r["condition"] == "perturbed"
                            and abs(r["command_df"] - df) < 1e-9), key=lambda r: r["realized_width_m"])
            x = [c["realized_width_m"] for c in cells]
            y = [100 * c["success_rate"] for c in cells]
            err = [[100 * (c["success_rate"] - c["ci_low"]) for c in cells],
                   [100 * (c["ci_high"] - c["success_rate"]) for c in cells]]
            realized = np.mean([c["realized_df"] for c in cells])
            ax.errorbar(x, y, yerr=err, color=DF_COLOR[df], marker=DF_MARKER[df], ms=4.5,
                        lw=1.6, capsize=2, elinewidth=.8,
                        label=f"Commanded DF {df:g} (realized {realized:.2f})")
        ax.set_title(POLICY_LABEL[policy], loc="left", color=TEXT)
        ax.set_xlabel("Realized full stance width (m)")
        ax.set_xlim(0.04, 0.19)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y")
        ax.legend(loc="upper left" if policy == "narrow" else "lower left")
    axes[0].set_ylabel("Trials completing 4 m under pushes (%)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "push_success_vs_stance_width")


def figure_survival(survival):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2.4]})
    tmax = max(t[-1] for t, _ in survival.values())
    t = np.arange(0, tmax + CONTROL_DT, CONTROL_DT)
    bound = 25 * (.1 + .9 * np.clip(t / 10., 0, 1))
    top.plot(t, bound, color=GREY, lw=1.4)
    top.fill_between(t, 0, bound, color=GREY, alpha=.15, lw=0)
    top.set_ylabel("Push force\nbound (N)")
    top.set_ylim(0, 27)
    top.set_yticks([0, 25])
    colors = {"narrow": BLUE["450"], "push_trained": ORANGE}
    for policy, (tt, alive) in survival.items():
        bottom.plot(tt, 100 * alive, color=colors[policy], lw=1.8, label=POLICY_LABEL[policy])
        bottom.text(tt[-1], 100 * alive[-1] + 2.5, f"{100 * alive[-1]:.0f}%", color=colors[policy],
                    ha="right", va="bottom", fontsize=7)
    bottom.set_xlabel("Episode time (s)")
    bottom.set_ylabel("Trials not fallen (%)")
    bottom.set_xlim(0, tmax)
    bottom.set_ylim(0, 105)
    bottom.grid(axis="y")
    bottom.legend(loc="lower left")
    fig.tight_layout(h_pad=.6)
    save(fig, "push_survival_vs_time")


def figure_cot_periodicity():
    cells = list(csv.DictReader((ROOT / "PAPER_GRAPHS" / "old_policy_walk_df" / "surface_cells.csv").open()))
    groups = list(csv.DictReader((ROOT / "PAPER_GRAPHS" / "old_policy_walk_df" / "gait_df_summary.csv").open()))
    speeds = (0.25, 0.30, 0.35, 0.40)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5))
    series = [("trot", 0.5, BLUE["250"], "o", "Trot, DF 0.50"),
              ("trot", 0.625, BLUE["450"], "s", "Trot, DF 0.625"),
              ("trot", 0.75, BLUE["650"], "^", "Trot, DF 0.75"),
              ("walk", 0.75, ORANGE, "D", "Walk, DF 0.75")]
    for gait, df, color, marker, label in series:
        x, y, complete = [], [], []
        for v in speeds:
            valid = [float(c["positive_mechanical_cot_median"]) for c in cells
                     if c["gait"] == gait and abs(float(c["command_df"]) - df) < 1e-9
                     and abs(float(c["speed"]) - v) < 1e-9 and c["energy_condition_valid"] == "1"]
            if valid:
                x.append(v)
                y.append(float(np.median(valid)))
                complete.append(len(valid) == 5)
        full = [i for i, ok in enumerate(complete) if ok]
        part = [i for i, ok in enumerate(complete) if not ok]
        if full:
            a.plot([x[i] for i in full], [y[i] for i in full], color=color, marker=marker,
                   ms=4.5, lw=1.6, label=label)
        if part:
            a.plot([x[i] for i in part], [y[i] for i in part], color=color, marker=marker,
                   ms=4.5, lw=0, mfc="white", mew=1.2,
                   label=f"{label} (partial widths)" if not full else None)
    a.set_xlabel("Commanded speed (m/s)")
    a.set_ylabel("Positive mechanical CoT")
    a.set_xticks(speeds)
    a.set_ylim(0.10, 0.45)
    a.grid(axis="y")
    a.legend(loc="lower center", ncol=2)
    for gait, color in (("trot", BLUE["450"]), ("walk", ORANGE)):
        rows = sorted((g for g in groups if g["gait"] == gait), key=lambda g: float(g["command_df"]))
        x = [float(g["command_df"]) for g in rows]
        y = [100 * int(g["periodic_trials"]) / int(g["trials"]) for g in rows]
        realized = [float(g["mean_achieved_df"]) for g in rows]
        compliant = [int(g["compliant_trials"]) > 0 for g in rows]
        b.plot(x, y, color=color, lw=1.6, label=gait.capitalize())
        for xi, yi, ok, r in zip(x, y, compliant, realized):
            b.plot(xi, yi, marker="o", ms=5, color=color, mfc=color if ok else "white", mew=1.2)
            if not ok:
                b.annotate(f"realized {r:.2f}", (xi, yi), textcoords="offset points",
                           xytext=(6, -10), fontsize=6.5, color=TEXT2)
    b.set_xlabel("Commanded duty factor")
    b.set_ylabel("Trials reaching a\nperiod-one gait (%)")
    b.set_xticks([0.5, 0.625, 0.75])
    b.set_ylim(0, 105)
    b.grid(axis="y")
    b.legend(loc="lower right")
    fig.tight_layout(w_pad=2)
    save(fig, "cot_and_periodicity_seed2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-csv", action="store_true",
                        help="Plot from the tracked CSVs instead of the result archives")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, survival = read_push_tables() if args.from_csv else collect_push_tables()
    figure_push_success(rows)
    figure_survival(survival)
    figure_cot_periodicity()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
