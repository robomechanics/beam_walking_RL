"""Combine the original and high-duty fresh selector validations at 0.48 s."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

SPEEDS = (.25, .30, .35, .40)
COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
WIDTH_OFFSET = .006


def read_rows(path, controller):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    converted = []
    for row in rows:
        if not np.isclose(float(row["period"]), .48):
            continue
        converted.append({
            "controller": controller,
            "gait": row["gait"],
            "speed": float(row["speed"]),
            "period": float(row["period"]),
            "step_width": float(row["step_width"]),
            "selected_df": float(row["selected_df"]),
            "achieved_df_median": float(row["achieved_df_median"]),
            "achieved_df_q025": float(row["achieved_df_q025"]),
            "achieved_df_q975": float(row["achieved_df_q975"]),
            "compliance_rate": float(row["compliance_rate"]),
            "trials": int(row["trials"]),
        })
    if not converted:
        raise ValueError(f"No period-0.48 validation rows in {path}")
    return converted


def plot_series(axis, rows, gait, controller, speed, color, marker, offset):
    group = sorted(
        (row for row in rows
         if row["gait"] == gait and row["controller"] == controller
         and np.isclose(row["speed"], speed)
         and row["step_width"] <= .400001),
        key=lambda row: row["step_width"])
    if not group:
        return
    width = np.asarray([row["step_width"] for row in group]) + offset
    achieved = np.asarray([row["achieved_df_median"] for row in group])
    low = np.asarray([row["achieved_df_q025"] for row in group])
    high = np.asarray([row["achieved_df_q975"] for row in group])
    is_high_duty = controller == "high_duty_walk_seed4"
    axis.errorbar(
        width, achieved,
        yerr=np.vstack((achieved - low, high - achieved)),
        color=color, marker=marker, linestyle="--" if is_high_duty else "-",
        markerfacecolor="white" if is_high_duty else color,
        markeredgewidth=1.7 if is_high_duty else 1,
        linewidth=2, markersize=6.5, capsize=2.5, zorder=3)


def write_combined_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", type=Path, required=True)
    parser.add_argument("--high-duty", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = (read_rows(args.adaptive, "adaptive_seed3")
            + read_rows(args.high_duty, "high_duty_walk_seed4"))
    write_combined_csv(
        args.output / "combined_selector_validation_summary.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), sharey=True)
    centre = (len(SPEEDS) - 1) / 2
    for gait, axis in zip(("trot", "walk"), axes):
        for index, (speed, color, marker) in enumerate(
                zip(SPEEDS, COLORS, MARKERS)):
            offset = (index - centre) * WIDTH_OFFSET
            plot_series(
                axis, rows, gait, "adaptive_seed3", speed,
                color, marker, offset)
            if gait == "walk":
                plot_series(
                    axis, rows, gait, "high_duty_walk_seed4", speed,
                    color, marker, offset)
        axis.set_title(gait.title())
        axis.set_xlabel("Commanded full step width (m)")
        axis.set_xticks((.10, .20, .30, .40))
        axis.set_xlim(.075, .425)
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Achieved duty factor")
    axes[0].set_ylim(.54, .86)
    axes[0].set_yticks(np.arange(.55, .851, .05))
    handles = [
        Line2D([], [], color=color, marker=marker, linewidth=2,
               markersize=6.5, label=f"{speed:.2f} m/s")
        for speed, color, marker in zip(SPEEDS, COLORS, MARKERS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=10, frameon=False, bbox_to_anchor=(.5, -.01))

    fig.suptitle(
        "Fresh selector validation at period 0.48 s\n"
        "Combined measurements from two separately trained low-level controllers",
        fontsize=15)
    fig.tight_layout(rect=(0, .06, 1, 1))
    stem = args.output / "combined_fresh_selector_validation_p048"
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("COMBINED_DUTY_VALIDATION", stem, flush=True)


if __name__ == "__main__":
    main()
