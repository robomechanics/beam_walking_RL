"""Trend figures for the gait specialists: cost of transport against stance
width and against duty factor.

The 3-D surface figures drop any cell that fails the frozen compliance gates,
which punches holes in the trend at the extreme widths even though the energy
measurement there is sound. These figures therefore plot two things at once:

  * a solid line through every condition whose trials completed with a finite
    positive mechanical cost of transport, which is the energy trend; and
  * open markers on the conditions that additionally passed the compliance,
    contact-topology and periodicity gates.

A condition never enters either series if any trial in it suffered a physical
failure, and the gate outcome is written to the accompanying CSV, so nothing is
hidden by the choice of series.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

GAIT_COLORS = {"trot": "#0072BD", "walk": "#D95319"}
DUTY_STYLE = ["-o", "-s", "-^", "-D"]


def load(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "surface_manifest.json").read_text())
    trials = pd.read_csv(directory / "surface_trials.csv")
    return manifest, trials


def cells(trials):
    """One row per condition: energy trend value, gate outcome, denominators."""
    keys = ["gait", "speed", "period", "step_width", "command_df"]
    rows = []
    for key, group in trials.groupby(keys):
        failed = group.any_failure.astype(bool)
        finite = np.isfinite(group.positive_mechanical_cot) & ~failed
        gated = group.energy_trial_valid.astype(bool)
        rows.append(dict(
            zip(keys, key), trials=len(group),
            physical_failures=int(failed.sum()),
            finite_trials=int(finite.sum()),
            cot_all=float(group.loc[finite, "positive_mechanical_cot"].median())
            if finite.any() else np.nan,
            gate_pass_rate=float(gated.mean()),
            gate_valid=int(gated.mean() >= .90),
            cot_gated=float(group.loc[gated, "positive_mechanical_cot"].median())
            if gated.mean() >= .90 else np.nan,
            compliance_rate=float(group.compliant.mean()),
            periodic_rate=float(group.periodic_orbit_gate_pass.mean()),
            achieved_df=float(group.achieved_df.mean())))
    return pd.DataFrame(rows)


def trend(cell_table, x, fixed, out, stem, xlabel, title):
    """One panel per gait; one line per level of the non-x factor."""
    other = "command_df" if x == "step_width" else "step_width"
    gaits = [g for g in ("trot", "walk") if (cell_table.gait == g).any()]
    fig, axes = plt.subplots(1, len(gaits), figsize=(5.6 * len(gaits), 4.6),
                             squeeze=False)
    for axis, gait in zip(axes[0], gaits):
        data = cell_table[(cell_table.gait == gait)
                          & np.isclose(cell_table.period, fixed["period"])
                          & np.isclose(cell_table.speed, fixed["speed"])]
        for index, level in enumerate(sorted(data[other].unique())):
            group = data[np.isclose(data[other], level)].sort_values(x)
            if group.empty:
                continue
            style = DUTY_STYLE[index % len(DUTY_STYLE)]
            shade = .45 + .55 * index / max(1, len(data[other].unique()) - 1)
            label = (f"DF {level:g}" if other == "command_df" else f"{level:.2f} m")
            axis.plot(group[x], group.cot_all, style, color=GAIT_COLORS[gait],
                      alpha=shade, linewidth=1.9, markersize=6,
                      markerfacecolor=GAIT_COLORS[gait], label=label)
            passed = group[group.gate_valid == 1]
            axis.plot(passed[x], passed.cot_all, style[1:], color=GAIT_COLORS[gait],
                      alpha=shade, markersize=11, markerfacecolor="none",
                      markeredgewidth=1.6, linestyle="none")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Positive mechanical CoT")
        axis.set_title(gait.title())
        axis.grid(alpha=.25)
        axis.legend(frameon=False, title=("Duty factor" if other == "command_df"
                                          else "Stance width"))
    fig.suptitle(f"{title}   (period {fixed['period']:.2f} s, "
                 f"speed {fixed['speed']:.2f} m/s)", fontsize=14)
    fig.text(.5, .005, "Filled markers: every completed trial with finite energy. "
             "Open rings: condition also passed the compliance gates.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .96))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Surface grid directories, one per gait")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "PAPER_GRAPHS/specialist_trends")
    parser.add_argument("--period", type=float, default=.48)
    parser.add_argument("--speed", type=float, default=.30)
    args = parser.parse_args()
    manifests, frames = zip(*(load(d) for d in args.input))
    table = cells(pd.concat(frames, ignore_index=True))
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "trend_cells.csv", index=False)
    fixed = {"period": args.period, "speed": args.speed}
    trend(table, "step_width", fixed, args.output, "cot_vs_stance_width",
          "Commanded full stance width (m)", "Energy against stance width")
    trend(table, "command_df", fixed, args.output, "cot_vs_duty_factor",
          "Commanded duty factor", "Energy against duty factor")
    selected = table[np.isclose(table.period, args.period)
                     & np.isclose(table.speed, args.speed)]
    (args.output / "provenance.json").write_text(json.dumps(dict(
        input=[str(d.resolve()) for d in args.input],
        checkpoint_sha256=[m["checkpoint_sha256"] for m in manifests],
        period=args.period, speed=args.speed,
        conditions=len(selected),
        conditions_with_physical_failures=int((selected.physical_failures > 0).sum()),
        gate_valid_conditions=int(selected.gate_valid.sum()),
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        indent=2))
    print(json.dumps(dict(output=str(args.output), conditions=len(selected),
                          gate_valid=int(selected.gate_valid.sum()))))


if __name__ == "__main__":
    main()
