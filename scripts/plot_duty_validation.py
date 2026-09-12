"""Summarize and plot achieved duty factor from fresh selector rollouts."""

import argparse
import csv
from pathlib import Path

import numpy as np

# Fixed speed -> hue order, validated for colour-vision deficiency separation.
SPEED_COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
SPEED_MARKERS = ("o", "s", "^", "D")
SERIES_OFFSET = .008  # metres of horizontal nudge between overlapping series
MAX_PLOT_STEP_WIDTH = .45  # widest commanded step width drawn in the figure


def summarize_validation_rows(rows):
    """Aggregate per-trial rollout rows into one record per command context."""
    contexts = {}
    for row in rows:
        key = (row["gait"], row["speed"], row["period"],
               row["step_width"], row["command_df"])
        contexts.setdefault(key, []).append(row)
    summaries = []
    for key, trials in sorted(contexts.items()):
        achieved = np.asarray([trial["achieved_df"] for trial in trials])
        compliant = np.asarray([trial["compliant"] for trial in trials], dtype=bool)
        summaries.append({
            "gait": key[0], "speed": key[1], "period": key[2],
            "step_width": key[3], "selected_df": key[4],
            "achieved_df_median": float(np.median(achieved)),
            "achieved_df_q025": float(np.quantile(achieved, .025)),
            "achieved_df_q975": float(np.quantile(achieved, .975)),
            "compliance_rate": float(compliant.mean()),
            "finite_energy_compliant_trials": int(sum(
                trial["compliant"] and np.isfinite(
                    trial["positive_mechanical_cot"]) for trial in trials)),
            "trials": len(trials),
        })
    return summaries


def plot_validation_summaries(summaries, output, gaits,
                              max_step_width=MAX_PLOT_STEP_WIDTH):
    """Write the achieved-DF figure, one page per commanded period."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    summaries = [row for row in summaries
                 if row["step_width"] <= max_step_width + 1e-9]
    periods = sorted({row["period"] for row in summaries})
    speeds = sorted({row["speed"] for row in summaries})
    # Speeds often land on the same duty-factor level, so nudge each series
    # sideways to keep coincident markers readable.
    centre = (len(speeds) - 1) / 2
    styles = {speed: (SPEED_COLORS[index % len(SPEED_COLORS)],
                      SPEED_MARKERS[index % len(SPEED_MARKERS)],
                      (index - centre) * SERIES_OFFSET)
              for index, speed in enumerate(speeds)}
    with PdfPages(output / "selected_vs_achieved_duty_factor.pdf") as pdf:
        for period in periods:
            fig, axes = plt.subplots(1, len(gaits), figsize=(11, 4.2), sharey=True)
            for axis, gait in zip(np.atleast_1d(axes), gaits):
                gait_rows = [row for row in summaries
                             if row["gait"] == gait and row["period"] == period]
                for speed in sorted({row["speed"] for row in gait_rows}):
                    group = sorted(
                        (row for row in gait_rows if row["speed"] == speed),
                        key=lambda row: row["step_width"])
                    width = np.asarray([row["step_width"] for row in group])
                    achieved = np.asarray([
                        row["achieved_df_median"] for row in group])
                    low = np.asarray([row["achieved_df_q025"] for row in group])
                    high = np.asarray([row["achieved_df_q975"] for row in group])
                    color, marker, offset = styles[speed]
                    axis.errorbar(
                        width + offset, achieved,
                        yerr=np.vstack((achieved - low, high - achieved)),
                        fmt=f"{marker}-", color=color, capsize=3, markersize=6,
                        linewidth=2, label=f"{speed:.2f} m/s")
                axis.set_title(gait.title())
                axis.set_xlabel("Commanded full step width (m)")
                axis.grid(alpha=.25)
                axis.set_ylim(.49, .76)
            np.atleast_1d(axes)[0].set_ylabel("Achieved duty factor")
            handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                       frameon=False, title="Commanded speed",
                       bbox_to_anchor=(.5, .005))
            fig.suptitle(
                f"Achieved duty factor, fresh selector validation, "
                f"period {period:.2f} s")
            fig.tight_layout(rect=(0, .1, 1, 1))
            pdf.savefig(fig)
            fig.savefig(
                output / f"selected_vs_achieved_duty_factor_p{period:.2f}.png",
                dpi=180)
            if len(periods) == 1:
                fig.savefig(output / "selected_vs_achieved_duty_factor.png", dpi=180)
            plt.close(fig)


def read_summary_csv(path):
    with path.open() as handle:
        return [{key: value if key == "gait" else float(value)
                 for key, value in row.items()}
                for row in csv.DictReader(handle)]


def main():
    parser = argparse.ArgumentParser(
        description="Redraw the selector validation figure from a summary CSV")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gaits", nargs="+", default=["trot", "walk"])
    parser.add_argument("--max-step-width", type=float,
                        default=MAX_PLOT_STEP_WIDTH)
    args = parser.parse_args()
    output = args.output or args.summary.parent
    summaries = read_summary_csv(args.summary)
    gaits = [gait for gait in args.gaits
             if any(row["gait"] == gait for row in summaries)]
    plot_validation_summaries(summaries, output, gaits,
                              args.max_step_width)
    print("DUTY_VALIDATION_FIGURE", output, "contexts", len(summaries))


if __name__ == "__main__":
    main()
