"""Create the concise final figure for the validated high-duty walk policy."""

import argparse
import csv
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def plot(validation_rows, candidate_rows, output):
    widths = sorted({float(row["step_width"]) for row in validation_rows})
    speeds = sorted({float(row["speed"]) for row in validation_rows})
    selected, achieved, achieved_low, achieved_high = [], [], [], []
    for width in widths:
        group = [row for row in validation_rows
                 if np.isclose(float(row["step_width"]), width)]
        selected_values = np.asarray([float(row["selected_df"]) for row in group])
        achieved_values = np.asarray([
            float(row["achieved_df_median"]) for row in group])
        selected.append(float(np.median(selected_values)))
        achieved.append(float(np.median(achieved_values)))
        achieved_low.append(min(float(row["achieved_df_q025"]) for row in group))
        achieved_high.append(max(float(row["achieved_df_q975"]) for row in group))

    duties = sorted({float(row["command_df"]) for row in candidate_rows})
    pooled_compliance, context_low, context_high, eligible = [], [], [], []
    for duty in duties:
        group = [row for row in candidate_rows
                 if np.isclose(float(row["command_df"]), duty)]
        pooled_compliance.append(
            sum(int(row["compliant_trials"]) for row in group)
            / sum(int(row["trials"]) for row in group))
        rates = [float(row["compliance_rate"]) for row in group]
        context_low.append(min(rates))
        context_high.append(max(rates))
        eligible.append(sum(rate >= .90 for rate in rates))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axis = axes[0]
    selected = np.asarray(selected)
    achieved = np.asarray(achieved)
    axis.plot(
        widths, selected, "o--", color="#0072B2", markerfacecolor="white",
        markersize=7, linewidth=2, label="Selected command")
    axis.errorbar(
        widths, achieved,
        yerr=np.vstack((
            achieved - np.asarray(achieved_low),
            np.asarray(achieved_high) - achieved)),
        fmt="D-", color="#E69F00", capsize=3, markersize=6, linewidth=2,
        label="Achieved median")
    axis.set_xlabel("Commanded full step width (m)")
    axis.set_ylabel("Duty factor")
    axis.set_xticks(widths)
    axis.set_yticks((.75, .80, .8229, .85, .90))
    axis.set_ylim(.74, .91)
    axis.grid(alpha=.25)
    axis.legend(frameon=False, loc="upper left")
    axis.set_title("Validated selector output")
    axis.text(
        .5, .04,
        f"All {len(speeds)} speeds ({min(speeds):.2f}–{max(speeds):.2f} m/s) overlap",
        transform=axis.transAxes, ha="center", va="bottom", fontsize=10,
        bbox={"boxstyle": "round,pad=.3", "facecolor": "white",
              "edgecolor": ".75", "alpha": .9})

    axis = axes[1]
    x = np.arange(len(duties))
    colors = ["#E69F00" if np.isclose(duty, .80) else "#B8B8B8"
              for duty in duties]
    bars = axis.bar(x, pooled_compliance, color=colors, width=.62)
    axis.errorbar(
        x, pooled_compliance,
        yerr=np.vstack((
            np.asarray(pooled_compliance) - np.asarray(context_low),
            np.asarray(context_high) - np.asarray(pooled_compliance))),
        fmt="none", ecolor="#333333", capsize=5, linewidth=1.5)
    axis.axhline(.90, color="#333333", linestyle="--", linewidth=1.2,
                 label="90% eligibility gate")
    for bar, count in zip(bars, eligible):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(1.04, bar.get_height() + .035),
            f"{count}/20 eligible", ha="center", va="bottom", fontsize=9)
    axis.set_xticks(x, [f"{duty:.2f}" for duty in duties])
    axis.set_xlabel("Candidate duty factor")
    axis.set_ylabel("Pooled compliant-trial fraction")
    axis.set_ylim(0, 1.12)
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, loc="upper right")
    axis.set_title("Candidate command compliance")

    fig.suptitle(
        "High-duty walking policy, period 0.48 s\n"
        "DF 0.80 was the lowest-energy eligible choice in all 20 contexts",
        fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "selected_vs_achieved_duty_factor.png", dpi=200)
    fig.savefig(output / "selected_vs_achieved_duty_factor.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    png = args.output / "selected_vs_achieved_duty_factor.png"
    pdf = args.output / "selected_vs_achieved_duty_factor.pdf"
    if png.is_file() and not (args.output / "selected_vs_achieved_duty_factor_by_speed.png").exists():
        shutil.copy2(png, args.output / "selected_vs_achieved_duty_factor_by_speed.png")
    if pdf.is_file() and not (args.output / "selected_vs_achieved_duty_factor_by_speed.pdf").exists():
        shutil.copy2(pdf, args.output / "selected_vs_achieved_duty_factor_by_speed.pdf")
    plot(read_csv(args.validation), read_csv(args.candidates), args.output)
    print("HIGH_DUTY_WALK_FINAL_FIGURE", args.output, flush=True)


if __name__ == "__main__":
    main()
