"""Paper figures for the narrow-stance specialists.

  1  periodicity (cycle RMS, median and interquartile band) against realized
     duty factor, trot and walk on one axis, one panel per stance width
  2  positive mechanical CoT against realized duty factor, trot and walk on one
     axis, one panel per stance width
  3  robustness: success rate under random base disturbances against stance
     width, one line per commanded duty factor, trot and walk panels, one row per
     disturbance level
  4  realized duty factor by stance width and commanded duty factor, trot and
     walk panels
  5  the duty-factor selector network's choice against stance width: mean and
     interquartile range over its speed and period contexts, trot and walk on one
     axis

Every figure pools the periods at which every duty factor of the gait was run, so
each duty factor averages the same periods (trot 0.75 needs 0.40 s or longer).
CoT uses every trial that completed without a failure and with a finite
positive value. Each figure is written with the table it plots.
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GAITS = ("trot", "walk")
GAIT_COLORS = {"trot": "#2a78d6", "walk": "#eb6834"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
TEXT, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
# Ordinal blue, light to dark as duty factor rises; marker shape is the second cue.
DUTY_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
MARKERS = ("o", "s", "^", "D", "v", "P")


def style(ax, grid=True):
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    if grid:
        ax.grid(axis="y", color=GRID, lw=.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def save(fig, out, stem, rows):
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        # no PDF creation date, so regenerating unchanged figures leaves the files unchanged
        fig.savefig(out / f"{stem}.{ext}", dpi=200, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    with (out / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(s, n, z=1.96):
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., c - h), min(1., c + h)


def balanced_periods(pairs):
    """Periods at which every condition (width, duty factor, ...) of the gait was run, from
    (period, condition) pairs, so pooling never mixes periods with different grids."""
    by_period = defaultdict(set)
    for period, condition in pairs:
        by_period[period].add(condition)
    conditions = set().union(*by_period.values())
    return sorted(p for p, c in by_period.items() if c == conditions)


def runs(results, prefix, suffix):
    """{period: directory} for directories named exactly <prefix>p<period><suffix>."""
    pattern = re.compile("^" + re.escape(prefix) + r"p(\d\.\d\d)" + re.escape(suffix) + "$")
    found = {}
    for directory in sorted(results.iterdir()):
        match = pattern.match(directory.name)
        if match and directory.is_dir():
            found[float(match.group(1))] = directory
    return found


def quartiles(values):
    values = sorted(values)
    if len(values) < 2:
        return values[0], values[0]
    q = st.quantiles(values, n=4)
    return q[0], q[2]


def load_sweep(results, tag, gait):
    path = results / f"narrow_surfaces_sweep_{gait}_{tag}/grid/surface_trials.csv"
    rows = list(csv.DictReader(path.open()))
    keep = balanced_periods({(round(float(r["period"]), 2), (round(float(r["speed"]), 2), round(float(r["step_width"]), 2),
                                                             round(float(r["command_df"]), 3))) for r in rows})
    return [r for r in rows if round(float(r["period"]), 2) in keep], keep


def heatmap(ax, values, rows, columns, fmt, vmin, vmax):
    """values[(row, column)] -> number; rows drawn bottom to top."""
    grid = [[values.get((r, c), float("nan")) for c in columns] for r in rows]
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap=SEQUENTIAL, vmin=vmin, vmax=vmax)
    for i, r in enumerate(rows):
        for j, c in enumerate(columns):
            v = grid[i][j]
            if not math.isnan(v):
                light = (v - vmin) / (vmax - vmin) < .55
                ax.text(j, i, fmt(v), ha="center", va="center", fontsize=7,
                        color=TEXT if light else "#ffffff")
    ax.set_xticks(range(len(columns)), [f"{c:g}" for c in columns])
    ax.set_yticks(range(len(rows)), [f"{r:.2f}" for r in rows])
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    return image


def figure_periodicity(results, tag, out):
    table = []
    for gait in GAITS:
        rows, periods = load_sweep(results, tag, gait)
        cells = defaultdict(list)
        for r in rows:
            if r["cycle_rms"] not in ("", "nan"):
                cells[(round(float(r["step_width"]), 2), round(float(r["command_df"]), 3))].append(r)
        for (w, d), rs in sorted(cells.items()):
            q = quartiles(float(r["cycle_rms"]) for r in rs)
            table.append({"gait": gait, "step_width": w, "command_df": d,
                          "achieved_df_median": st.median(float(r["achieved_df"]) for r in rs),
                          "cycle_rms_median": st.median(float(r["cycle_rms"]) for r in rs),
                          "cycle_rms_q25": q[0], "cycle_rms_q75": q[1],
                          "trials": len(rs), "periods": " ".join(f"{p:g}" for p in periods)})
    widths = sorted({t["step_width"] for t in table})
    fig, axes = plt.subplots(1, len(widths), figsize=(2.1 * len(widths), 2.9), sharey=True, sharex=True,
                             squeeze=False)
    axes = axes[0]
    for ax, w in zip(axes, widths):
        for gait in GAITS:
            pts = sorted((t for t in table if t["gait"] == gait and t["step_width"] == w),
                         key=lambda t: t["achieved_df_median"])
            x = [t["achieved_df_median"] for t in pts]
            ax.fill_between(x, [t["cycle_rms_q25"] for t in pts], [t["cycle_rms_q75"] for t in pts],
                            color=GAIT_COLORS[gait], alpha=.15, lw=0)
            ax.plot(x, [t["cycle_rms_median"] for t in pts], color=GAIT_COLORS[gait], lw=2, marker="o",
                    ms=4, label=gait.capitalize())
        ax.set_yscale("log")
        ax.set_title(f"Stance width {w:.2f} m", color=TEXT, fontsize=10)
        ax.set_xlabel("Realized duty factor", color=TEXT, fontsize=9)
        style(ax)
    axes[0].set_ylabel("Periodicity error (lower is more repeatable)", color=TEXT, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper right")
    save(fig, out, "fig1_periodicity_vs_duty_factor", table)


def figure_cot(results, tag, out):
    table = []
    for gait in GAITS:
        rows, periods = load_sweep(results, tag, gait)
        cells = defaultdict(list)
        for r in rows:
            cot = r["positive_mechanical_cot"]
            if r["any_failure"] not in ("1", "True") and cot not in ("", "nan") and float(cot) > 0:
                cells[(round(float(r["step_width"]), 2), round(float(r["command_df"]), 3))].append(r)
        for (w, d), rs in sorted(cells.items()):
            table.append({"gait": gait, "step_width": w, "command_df": d,
                          "achieved_df_median": st.median(float(r["achieved_df"]) for r in rs),
                          "positive_mechanical_cot_median": st.median(float(r["positive_mechanical_cot"]) for r in rs),
                          "trials": len(rs), "periods": " ".join(f"{p:g}" for p in periods)})
    widths = sorted({t["step_width"] for t in table})
    fig, axes = plt.subplots(1, len(widths), figsize=(2.1 * len(widths), 2.9), sharey=True, sharex=True,
                             squeeze=False)
    axes = axes[0]
    for ax, w in zip(axes, widths):
        for gait in GAITS:
            pts = sorted((t for t in table if t["gait"] == gait and t["step_width"] == w),
                         key=lambda t: t["achieved_df_median"])
            ax.plot([t["achieved_df_median"] for t in pts], [t["positive_mechanical_cot_median"] for t in pts],
                    color=GAIT_COLORS[gait], lw=2, marker="o", ms=4, label=gait.capitalize())
        ax.set_title(f"Stance width {w:.2f} m", color=TEXT, fontsize=10)
        ax.set_xlabel("Realized duty factor", color=TEXT, fontsize=9)
        style(ax)
    axes[0].set_ylabel("Cost of transport", color=TEXT, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left")
    save(fig, out, "fig2_cot_vs_duty_factor", table)


def figure_push(results, tag, out):
    table = []
    for gait in GAITS:
        levels = {}
        for force, level in ((25, ""), (50, "_f50")):
            levels[force] = {p: list(json.loads((d / "perturbation_summary.json").read_text())["conditions"].values())
                             for p, d in runs(results, f"push_robustness_narrow_{gait}_v030_", f"{level}_{tag}").items()
                             if (d / "perturbation_summary.json").exists()}
        if not levels[25]:
            continue
        # Both push levels pool the periods balanced at the training push level.
        keep = balanced_periods({(p, (round(c["step_width"], 2), round(c["df"], 3)))
                                 for p, cs in levels[25].items() for c in cs})
        for force, summaries in levels.items():
            if not keep or any(p not in summaries for p in keep):
                continue
            counts = defaultdict(lambda: [0, 0])
            for p in keep:
                for c in summaries[p]:
                    k = (round(c["step_width"], 2), round(c["df"], 3))
                    counts[k][0] += c["success"]
                    counts[k][1] += c["trials"]
            for (w, d), (s_, n) in sorted(counts.items()):
                if n == 0:
                    continue
                low, high = wilson(s_, n)
                table.append({"gait": gait, "push_force_n": force, "step_width": w, "command_df": d,
                              "success": s_, "trials": n, "success_rate": s_ / n, "ci_low": low, "ci_high": high,
                              "periods": " ".join(f"{p:g}" for p in keep)})
    forces = sorted({t["push_force_n"] for t in table})
    fig, axes = plt.subplots(len(forces), 2, figsize=(9.6, 3.6 * len(forces)), sharex=True, sharey=True,
                             squeeze=False)
    for i, force in enumerate(forces):
        for j, gait in enumerate(GAITS):
            ax = axes[i, j]
            cells = [t for t in table if t["gait"] == gait and t["push_force_n"] == force]
            if not cells:
                ax.axis("off")
                continue
            duties = sorted({t["command_df"] for t in cells})
            colors = DUTY_RAMP if len(duties) > 4 else ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")[-len(duties):]
            for k, (d, color) in enumerate(zip(duties, colors)):
                pts = sorted((t for t in cells if t["command_df"] == d), key=lambda t: t["step_width"])
                y = [100 * t["success_rate"] for t in pts]
                err = [[max(0., 100 * (t["success_rate"] - t["ci_low"])) for t in pts],
                       [max(0., 100 * (t["ci_high"] - t["success_rate"])) for t in pts]]
                ax.errorbar([t["step_width"] for t in pts], y, yerr=err, color=color, lw=2,
                            marker=MARKERS[k % len(MARKERS)], ms=5, elinewidth=.8, capsize=0, label=f"{d:g}")
            ax.set_ylim(-3, 103)
            ax.set_title(gait.capitalize(), color=TEXT, fontsize=10)
            if i == len(forces) - 1:
                ax.set_xlabel("Stance width (m)", color=TEXT, fontsize=9)
            label = "Success rate (%)" if len(forces) == 1 else f"Success rate (%), disturbance up to {force} N"
            ax.set_ylabel(label, color=TEXT, fontsize=9)
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, title="Duty factor", title_fontsize=8,
                      loc="lower right")
            style(ax)
    save(fig, out, "fig3_robustness_vs_stance_width", table)


def figure_realized_duty(results, tag, out):
    table = []
    for gait in GAITS:
        summaries = {p: [r for r in csv.DictReader((d / "summary.csv").open()) if r["achieved_df"]]
                     for p, d in runs(results, f"validation_narrow_{gait}_v030_", f"_{tag}").items()
                     if (d / "summary.csv").exists()}
        if not summaries:
            continue
        keep = balanced_periods({(p, (round(float(r["step_width"]), 2), round(float(r["command_df"]), 3)))
                                 for p, rs in summaries.items() for r in rs})
        cells = defaultdict(list)
        for p in keep:
            for r in summaries[p]:
                cells[(round(float(r["step_width"]), 2), round(float(r["command_df"]), 3))].append(float(r["achieved_df"]))
        for (w, d), v in sorted(cells.items()):
            table.append({"gait": gait, "step_width": w, "command_df": d, "achieved_df_mean": st.mean(v),
                          "periods": " ".join(f"{p:g}" for p in keep)})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    for ax, gait in zip(axes, GAITS):
        cells = [t for t in table if t["gait"] == gait]
        if not cells:
            ax.axis("off")
            continue
        widths = sorted({t["step_width"] for t in cells})
        duties = sorted({t["command_df"] for t in cells})
        image = heatmap(ax, {(t["step_width"], t["command_df"]): t["achieved_df_mean"] for t in cells},
                        widths, duties, lambda v: f"{v:.2f}", .45, .95)
        ax.set_title(gait.capitalize(), color=TEXT, fontsize=10)
        ax.set_xlabel("Commanded duty factor", color=TEXT, fontsize=9)
        ax.set_ylabel("Stance width (m)", color=TEXT, fontsize=9)
    bar = fig.colorbar(image, ax=list(axes), fraction=.025, pad=.02)
    bar.set_label("Realized duty factor", color=TEXT, fontsize=9)
    bar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    bar.outline.set_visible(False)
    save(fig, out, "fig4_realized_duty_factor_by_width", table)


def figure_selector(results, tag, out):
    """The selector network's duty factor for each of its speed and period contexts against stance
    width, per gait, over the periods at which every duty factor ran: mean and interquartile range
    of the contexts at each width. Widths where no context has any success have no selection."""
    fit = results / f"narrow_robust_selector_{tag}"
    predictions = list(csv.DictReader((fit / "selector_predictions.csv").open()))
    candidates = list(csv.DictReader((fit / "candidate_success.csv").open()))
    rejected = (list(csv.DictReader((fit / "rejected_contexts.csv").open()))
                if (fit / "rejected_contexts.csv").exists() else [])
    report = results / f"narrow_robust_selector_promoted_{tag}/validation_report.json"
    validated = json.loads(report.read_text())["all_contexts_passed"] if report.exists() else ""
    table, curves = [], {}
    for gait in GAITS:
        grid = [r for r in candidates if r["gait"] == gait]
        if not grid:
            continue
        keep = balanced_periods({(round(float(r["period"]), 2), (round(float(r["speed"]), 2),
                                  round(float(r["step_width"]), 2), round(float(r["command_df"]), 3)))
                                 for r in grid})
        speeds = sorted({round(float(r["speed"]), 2) for r in grid})
        at = lambda rows, w: [r for r in rows if r["gait"] == gait and round(float(r["period"]), 2) in keep
                              and round(float(r["step_width"]), 2) == w]
        curves[gait] = {}
        for w in sorted({round(float(r["step_width"]), 2) for r in grid}):
            chosen = sorted(float(r["selected_df"]) for r in at(predictions, w))
            if chosen:
                q25, q75 = (float(q) for q in np.quantile(chosen, [.25, .75], method="inverted_cdf"))
                curves[gait][w] = (float(np.mean(chosen)), q25, q75)
            else:
                curves[gait][w] = (None, None, None)
            mean, q25, q75 = curves[gait][w]
            table.append({"gait": gait, "step_width": w, "contexts": len(chosen),
                          "contexts_without_success": len(at(rejected, w)),
                          "selected_df_mean": "" if mean is None else mean,
                          "selected_df_q25": "" if q25 is None else q25,
                          "selected_df_q75": "" if q75 is None else q75,
                          "selections": " ".join(f"{d:g}x{chosen.count(d)}" for d in sorted(set(chosen))),
                          "speeds": " ".join(f"{v:g}" for v in speeds),
                          "periods": " ".join(f"{p:g}" for p in keep),
                          "selector_validated": validated})
    if not table:
        raise FileNotFoundError("no selector fit")
    tested = sorted({round(float(r["command_df"]), 3) for r in candidates})
    ceiling = tested[-1] + .07      # widths where no duty factor succeeds sit above the highest duty factor
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for j, (gait, points) in enumerate(curves.items()):
        shift = (j - (len(curves) - 1) / 2) * .008
        found = [(w + shift, c, lo, hi) for w, (c, lo, hi) in sorted(points.items()) if c is not None]
        ax.vlines([x for x, *_ in found], [lo for _, _, lo, _ in found], [hi for *_, hi in found],
                  color=GAIT_COLORS[gait], lw=.8)
        ax.plot([x for x, *_ in found], [c for _, c, _, _ in found], color=GAIT_COLORS[gait], lw=2, marker="o",
                ms=5, label=gait.capitalize())
        none = [w + shift for w, (c, _, _) in sorted(points.items()) if c is None]
        ax.plot(none, [ceiling] * len(none), ls="none", marker="x", ms=6, mew=1.6, color=GAIT_COLORS[gait])
    ax.plot([], [], ls="none", marker="x", ms=6, mew=1.6, color=MUTED, label="no duty factor succeeds")
    widths = sorted({t["step_width"] for t in table})
    ax.set_xticks(widths, [f"{w:.2f}" for w in widths])
    ax.set_yticks(tested, [f"{d:g}" for d in tested])
    # "none" is a category, not a duty factor: no gridline, and the axis line stops below it
    ax.set_yticks([ceiling], ["none"], minor=True)
    ax.tick_params(axis="y", which="minor", colors=MUTED, labelsize=8, length=0)
    ax.set_ylim(tested[0] - .03, ceiling + .03)
    ax.spines["left"].set_bounds(tested[0] - .03, tested[-1] + .02)
    ax.set_xlabel("Stance width (m)", color=TEXT, fontsize=9)
    ax.set_ylabel("Duty factor chosen by the selector", color=TEXT, fontsize=9)
    legend = ax.legend(frameon=True, facecolor="#ffffff", edgecolor="none", framealpha=1, fontsize=8,
                       labelcolor=TEXT, loc="upper right")
    legend.set_zorder(5)
    style(ax)
    save(fig, out, "fig5_selector_duty_factor_vs_stance_width", table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in args.output.glob("fig[0-9]_*"):
        stale.unlink()
    missing = []
    for figure in (figure_periodicity, figure_cot, figure_push, figure_realized_duty, figure_selector):
        try:
            figure(args.results, args.tag, args.output)
        except Exception as error:  # one figure's failure must not stop the others
            missing.append(f"{figure.__name__}: {type(error).__name__}: {error}")
    for line in missing:
        print("FIGURE_SKIPPED", line)
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()
