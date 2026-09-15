"""Success under pushes against duty factor for the narrow-stance specialists.

Reads the pushed evaluation archives (`perturbation_summary.json`) of each gait
and period, and writes one figure per gait: one panel per stance width, success
rate against commanded duty factor, one line per period, with 95% Wilson
intervals. The per-cell table also carries the realized duty factor from the
nominal fidelity run of the same checkpoint.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {"trot": (.36, .48, .54), "walk": (.40, .48, .54)}
PERIOD_COLORS = ("#86b6ef", "#2a78d6", "#104281")   # ordinal blue, short to long
TEXT, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def wilson(successes, trials, z=1.96):
    p = successes / trials
    d = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / d
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return max(0., centre - half), min(1., centre + half)


def nominal_duty(results, gait, period, tag):
    path = results / f"validation_narrow_{gait}_v030_p{period:.2f}_{tag}/summary.csv"
    if not path.exists():
        return {}
    return {(round(float(r["step_width"]), 2), round(float(r["command_df"]), 3)): float(r["achieved_df"])
            for r in csv.DictReader(path.open()) if r["achieved_df"]}


def level_suffix(force):
    """The training push level (25 N) has no suffix; other levels are named by peak force."""
    return "" if force == 25 else f"_f{force:g}"


def collect(results, gait, tag, force=25):
    rows = []
    for period in PERIODS[gait]:
        path = results / (f"push_robustness_narrow_{gait}_v030_p{period:.2f}{level_suffix(force)}_{tag}"
                          "/perturbation_summary.json")
        if not path.exists():
            raise FileNotFoundError(path)
        realized = nominal_duty(results, gait, period, tag)
        for c in json.loads(path.read_text())["conditions"].values():
            width, duty = round(float(c["step_width"]), 2), round(float(c["df"]), 3)
            low, high = wilson(c["success"], c["trials"])
            rows.append({"gait": gait, "push_force_n": force, "period": period, "step_width": width, "command_df": duty,
                         "nominal_achieved_df": realized.get((width, duty), float("nan")),
                         "trials": c["trials"], "success": c["success"], "failure": c["failure"],
                         "timeout": c["timeout"], "success_rate": c["success"] / c["trials"],
                         "ci_low": low, "ci_high": high,
                         "applied_force_n_mean": c.get("applied_force_n_mean")})
    return rows


def figure(rows, gait, output, force=25):
    widths = sorted({r["step_width"] for r in rows})
    duties = sorted({r["command_df"] for r in rows})
    dodge = .06 * (duties[-1] - duties[0])   # separates periods that overlap at 100%
    fig, axes = plt.subplots(1, len(widths), figsize=(2.2 * len(widths), 2.9), sharey=True)
    for ax, width in zip(axes, widths):
        for i, (period, color) in enumerate(zip(PERIODS[gait], PERIOD_COLORS)):
            cells = sorted((r for r in rows if r["step_width"] == width and r["period"] == period),
                           key=lambda r: r["command_df"])
            if not cells:
                continue
            x = [r["command_df"] + (i - 1) * dodge for r in cells]
            y = [r["success_rate"] for r in cells]
            err = [[r["success_rate"] - r["ci_low"] for r in cells],
                   [r["ci_high"] - r["success_rate"] for r in cells]]
            ax.errorbar(x, y, yerr=err, color=color, lw=2, marker="o", ms=5, capsize=0,
                        elinewidth=1, label=f"{period:.2f} s")
        ax.set_title(f"{width:.2f} m", color=TEXT, fontsize=10)
        ax.set_xticks(duties, [f"{d:g}" for d in duties])
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(axis="y", color=GRID, lw=.6)
        ax.set_ylim(-.03, 1.03)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
    axes[0].set_ylabel("Success under pushes", color=TEXT, fontsize=9)
    fig.supxlabel("Commanded duty factor", color=TEXT, fontsize=9)
    fig.suptitle(f"{gait.capitalize()}, pushes up to {force:g} N", color=TEXT, fontsize=11, x=.02, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False,
               fontsize=8, labelcolor=TEXT, title="Period", title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{gait}_push_success_vs_duty{level_suffix(force)}.{ext}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--forces", type=float, nargs="+", default=[25.],
                        help="Peak push force levels to plot, in N")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = []
    for force in args.forces:
        for gait in PERIODS:
            try:
                rows = collect(args.results, gait, args.tag, force)
            except FileNotFoundError:
                if force == 25:
                    raise
                continue   # stronger levels run only for gaits at full success at 25 N
            figure(rows, gait, args.output, force)
            table += rows
    with (args.output / "push_robustness_success.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
