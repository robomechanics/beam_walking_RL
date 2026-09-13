"""Plot every measured duty-factor candidate for trot and walk at period 0.48 s."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPEEDS = (.25, .30, .35, .40)
GAITS = ("trot", "walk")
COMPLIANCE_GATE = .90


def load_grid(path: Path, controller: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "gait", "speed", "period", "step_width", "command_df",
        "achieved_df", "compliant",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[np.isclose(frame["period"], .48)].copy()
    if frame.empty:
        raise ValueError(f"No period-0.48 rows in {path}")
    frame["controller"] = controller
    return frame


def summarize(grids: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "controller", "gait", "speed", "period", "step_width", "command_df"
    ]
    return (
        grids.groupby(keys, as_index=False)
        .agg(
            achieved_df_median=("achieved_df", "median"),
            achieved_df_q025=("achieved_df", lambda values: values.quantile(.025)),
            achieved_df_q975=("achieved_df", lambda values: values.quantile(.975)),
            compliance_rate=("compliant", "mean"),
            trials=("compliant", "size"),
        )
        .sort_values(keys)
    )


def load_selected(path: Path, controller: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[np.isclose(frame["period"], .48)].copy()
    frame["controller"] = controller
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive-grid", type=Path, required=True)
    parser.add_argument("--high-duty-grid", type=Path, required=True)
    parser.add_argument("--adaptive-selected", type=Path, required=True)
    parser.add_argument("--high-duty-selected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    grids = pd.concat([
        load_grid(args.adaptive_grid, "adaptive_seed3"),
        load_grid(args.high_duty_grid, "high_duty_walk_seed4"),
    ], ignore_index=True)
    candidate = summarize(grids)
    candidate.to_csv(
        args.output / "all_candidate_duty_response_summary.csv", index=False)

    selected = pd.concat([
        load_selected(args.adaptive_selected, "adaptive_seed3"),
        load_selected(args.high_duty_selected, "high_duty_walk_seed4"),
    ], ignore_index=True)

    commands = sorted(candidate["command_df"].unique())
    palette = plt.colormaps["turbo"](
        np.linspace(.05, .90, len(commands)))
    colors = dict(zip(commands, palette))

    fig, axes = plt.subplots(
        len(GAITS), len(SPEEDS), figsize=(17.0, 8.6),
        sharex=True, sharey=True)
    controller_style = {
        "adaptive_seed3": {"linestyle": "-", "marker": "o"},
        "high_duty_walk_seed4": {"linestyle": "--", "marker": "s"},
    }

    for gait_index, gait in enumerate(GAITS):
        for speed_index, speed in enumerate(SPEEDS):
            axis = axes[gait_index, speed_index]
            panel = candidate[
                candidate["gait"].eq(gait)
                & np.isclose(candidate["speed"], speed)
                & candidate["step_width"].le(.400001)
            ]
            for controller, style in controller_style.items():
                controller_rows = panel[panel["controller"].eq(controller)]
                for command_df, group in controller_rows.groupby("command_df"):
                    group = group.sort_values("step_width")
                    x = group["step_width"].to_numpy()
                    y = group["achieved_df_median"].to_numpy()
                    low = group["achieved_df_q025"].to_numpy()
                    high = group["achieved_df_q975"].to_numpy()
                    color = colors[command_df]
                    axis.fill_between(x, low, high, color=color, alpha=.06)
                    axis.plot(
                        x, y, color=color, linewidth=1.45,
                        linestyle=style["linestyle"], marker=style["marker"],
                        markersize=4.2, alpha=.86)
                    failed = group["compliance_rate"].to_numpy() < COMPLIANCE_GATE
                    if failed.any():
                        axis.scatter(
                            x[failed], y[failed], marker="x", color=color,
                            s=37, linewidths=1.5, zorder=5)

            for controller, linestyle, marker in (
                    ("adaptive_seed3", "-", "*"),
                    ("high_duty_walk_seed4", "--", "D")):
                chosen = selected[
                    selected["controller"].eq(controller)
                    & selected["gait"].eq(gait)
                    & np.isclose(selected["speed"], speed)
                    & selected["step_width"].le(.400001)
                ].sort_values("step_width")
                if chosen.empty:
                    continue
                axis.plot(
                    chosen["step_width"], chosen["achieved_df_median"],
                    color="black", linestyle=linestyle, marker=marker,
                    linewidth=2.4, markersize=6.2, zorder=7)

            axis.set_xticks((.10, .20, .30, .40))
            axis.set_xlim(.075, .425)
            axis.set_ylim(.50, .92)
            axis.set_yticks(np.arange(.50, .901, .05))
            axis.grid(alpha=.22)
            if gait_index == 0:
                axis.set_title(f"{speed:.2f} m/s")
            if speed_index == 0:
                axis.set_ylabel(f"{gait.title()}\nAchieved duty factor")
            if gait_index == len(GAITS) - 1:
                axis.set_xlabel("Full step width (m)")

    # Direct labels are clearer than a large legend for the nine candidates.
    for gait_index, gait in enumerate(GAITS):
        axis = axes[gait_index, -1]
        right_edge = candidate[
            candidate["gait"].eq(gait)
            & np.isclose(candidate["speed"], SPEEDS[-1])
            & np.isclose(candidate["step_width"], .40)
        ]
        for command_df, group in right_edge.groupby("command_df"):
            axis.text(
                .405, group["achieved_df_median"].mean(), f"{command_df:.2f}",
                color=colors[command_df], fontsize=7.5, va="center")
    fig.suptitle(
        "All measured duty-factor candidates at period 0.48 s\n"
        "Curve labels are commanded DF; black curves are fresh selector outputs; "
        "crosses failed compliance",
        fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))

    stem = args.output / "all_candidate_duty_response_p048"
    fig.savefig(stem.with_suffix(".png"), dpi=210, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("ALL_DUTY_CANDIDATES", stem, flush=True)


if __name__ == "__main__":
    main()
