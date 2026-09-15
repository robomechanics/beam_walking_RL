"""Fit and graph the unified trot/walk duty-factor selector from the surface grid.

Input is the unified surface grid trial table
(results/<run>/grid/surface_trials.csv from collect_policy_surfaces.py
--task unified). Labels, bootstrap intervals, and the classifier are the same
construction as the high-duty walk selector, extended to both gaits with
gait-specific candidate sets.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.protocol import GAITS  # noqa: E402
from beam_walking.experiment.unified_duty_selector import (  # noqa: E402
    CONTEXT_FIELDS,
    GAIT_LEVELS,
    PRIMARY_TRIALS_PER_CANDIDATE,
    aggregate_candidates,
    bootstrap_target_intervals,
    fit_selector,
    predict_rows,
    select_targets,
    selector_payload,
    validate_completed_surface_grid,
)

CANDIDATE_FIELDS = (
    *CONTEXT_FIELDS, "command_df", "trials", "compliant_trials",
    "finite_energy_compliant_trials", "compliance_rate",
    "finite_energy_compliant_rate", "median_positive_mechanical_cot",
)
TARGET_FIELDS = (
    *CONTEXT_FIELDS, "target_df", "target_positive_mechanical_cot",
    "target_compliance_rate", "eligible_candidates",
    "bootstrap_valid_fraction", "target_df_ci_low", "target_df_ci_high",
)
REJECTED_FIELDS = (*CONTEXT_FIELDS, "reason")
PREDICTION_FIELDS = (*TARGET_FIELDS, "selected_df")
GAIT_COLORS = {"trot": "#0072BD", "walk": "#D95319"}
SPEED_MARKERS = {.25: "o", .30: "s", .35: "^", .40: "D"}
PERIOD_MARKERS = {.36: "o", .40: "s", .48: "^", .54: "D"}


def write_rows(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _periods(rows):
    return sorted({round(row["period"], 6) for row in rows})


def plot_selector(predictions, rejected, output):
    """Selected duty factor versus step width; rows: gait, columns: period."""
    periods = _periods(predictions)
    fig, grid = plt.subplots(len(GAITS), len(periods),
                             figsize=(4.2 * len(periods) + 1, 8.4), squeeze=False)
    panels = [(grid[r, c], gait, period) for r, gait in enumerate(GAITS)
              for c, period in enumerate(periods)]
    for axis, gait, period in panels:
        rows = [row for row in predictions if row["gait"] == gait
                and abs(row["period"] - period) < 1e-6]
        if not rows:
            axis.set_axis_off()
            continue
        for speed in sorted({row["speed"] for row in rows}):
            group = sorted((row for row in rows if row["speed"] == speed),
                           key=lambda row: row["step_width"])
            width = np.asarray([row["step_width"] for row in group])
            selected = np.asarray([row["selected_df"] for row in group])
            low = np.asarray([row["target_df_ci_low"] for row in group])
            high = np.asarray([row["target_df_ci_high"] for row in group])
            axis.errorbar(
                width, selected,
                yerr=np.vstack((np.clip(selected - low, 0, None),
                                np.clip(high - selected, 0, None))),
                fmt=f"{SPEED_MARKERS.get(speed, 'o')}-", color=GAIT_COLORS[gait],
                alpha=min(1., .45 + .55 * (speed - .25) / .15), capsize=3,
                linewidth=1.8, markersize=6, label=f"{speed:.2f} m/s")
        for row in rejected:
            if row["gait"] == gait and abs(row["period"] - period) < 1e-6:
                axis.plot(row["step_width"], min(GAIT_LEVELS[gait]) - .02,
                          marker="x", color="#777777", markersize=8)
        levels = GAIT_LEVELS[gait]
        axis.set_yticks(levels)
        axis.set_ylim(min(levels) - .04, max(levels) + .04)
        axis.set_xlabel("Commanded full step width (m)")
        axis.set_ylabel("Selected duty factor")
        axis.set_title(f"{gait.title()}, period {period:.2f} s")
        axis.grid(alpha=.25)
        axis.legend(title="Speed", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "duty_factor_vs_step_width.png", dpi=200)
    fig.savefig(output / "duty_factor_vs_step_width.pdf")
    plt.close(fig)


def plot_candidates(candidates, output, min_compliance_rate):
    """Median CoT and compliance for every candidate, per gait, pooled over widths.

    One line per period at the 0.30 m/s core speed is drawn per gait; the
    per-speed detail lives in candidate_summary.csv.
    """
    fig, axes = plt.subplots(2, len(GAITS), figsize=(11, 7.2), sharex="col")
    for column, gait in enumerate(GAITS):
        rows = [row for row in candidates if row["gait"] == gait]
        for period in _periods(rows):
            group = [row for row in rows if abs(row["period"] - period) < 1e-6]
            duties = sorted({row["command_df"] for row in group})
            cot = [np.nanmedian([r["median_positive_mechanical_cot"] for r in group
                                 if r["command_df"] == d]) for d in duties]
            comp = [np.mean([r["compliance_rate"] for r in group
                             if r["command_df"] == d]) for d in duties]
            style = dict(marker=PERIOD_MARKERS.get(round(period, 2), "o"),
                         color=GAIT_COLORS[gait],
                         alpha=min(1., .4 + .6 * (period - .36) / .18), linewidth=1.8,
                         markersize=6, label=f"{period:.2f} s")
            axes[0, column].plot(duties, cot, **style)
            axes[1, column].plot(duties, comp, **style)
        axes[1, column].axhline(min_compliance_rate, color="#777777", linestyle="--")
        axes[0, column].set_title(gait.title())
        axes[1, column].set_xlabel("Commanded duty factor")
        axes[1, column].set_xticks(GAIT_LEVELS[gait])
        for axis in axes[:, column]:
            axis.grid(alpha=.25)
        axes[0, column].legend(title="Period", frameon=False)
    axes[0, 0].set_ylabel("Median positive mechanical CoT")
    axes[1, 0].set_ylabel("Compliance rate")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 1].set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output / "candidate_energy_and_compliance.png", dpi=200)
    fig.savefig(output / "candidate_energy_and_compliance.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path, nargs="+",
                        help="surface_trials.csv of one grid, or one per gait "
                             "when each gait has its own checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-compliance-rate", type=float, default=.90)
    parser.add_argument("--min-trials", type=int,
                        default=PRIMARY_TRIALS_PER_CANDIDATE)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploratory", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new or empty")
    if not args.exploratory and (
            args.min_trials != 32 or args.min_compliance_rate != .90
            or args.bootstrap_samples != 1000 or args.epochs != 3000
            or args.seed != 0):
        parser.error("Primary fitting parameters are fixed; use --exploratory")
    args.output.mkdir(parents=True, exist_ok=True)

    # Narrow-stance grids record their width domain; set it before any row is
    # validated so contexts below 0.10 m are accepted.
    import json as _json
    from beam_walking.experiment import unified_duty_selector as _selector
    _manifest = _json.loads((args.trials[0].parent / "surface_manifest.json").read_text())
    if _manifest.get("width_range_m"):
        _selector.set_width_range(_manifest["width_range_m"])
    if _manifest.get("selector_candidate_duties"):
        _selector.set_gait_levels(_manifest["selector_candidate_duties"])
    rows, grids = [], []
    for trials_path in args.trials:
        grid_rows, manifest, hashes = validate_completed_surface_grid(trials_path)
        gaits = sorted({row["gait"] for row in grid_rows})
        if any(g in grid["gaits"] for grid in grids for g in gaits):
            parser.error("Each gait must come from exactly one grid")
        grids.append({"gaits": gaits, "manifest": manifest, "hashes": hashes,
                      "trials_path": str(trials_path.resolve())})
        rows.extend(grid_rows)
    manifest, hashes = grids[0]["manifest"], grids[0]["hashes"]
    candidates = aggregate_candidates(rows)
    targets, rejected = select_targets(
        rows, args.min_compliance_rate, args.min_trials)
    if len(targets) < 2:
        raise ValueError("Too few compliant contexts to fit the selector")
    intervals = bootstrap_target_intervals(
        rows, args.min_compliance_rate, args.min_trials,
        samples=args.bootstrap_samples, seed=args.seed)
    interval_by_context = {
        tuple(row[field] for field in CONTEXT_FIELDS): row for row in intervals}
    targets = [dict(row, **interval_by_context[
        tuple(row[field] for field in CONTEXT_FIELDS)]) for row in targets]
    model, losses = fit_selector(targets, seed=args.seed, epochs=args.epochs)
    predictions = predict_rows(model, targets)
    if any(abs(row["selected_df"] - row["target_df"]) > 1e-6 for row in predictions):
        raise RuntimeError(
            "Classifier did not reproduce every registered grid label exactly")

    write_rows(args.output / "candidate_summary.csv", candidates, CANDIDATE_FIELDS)
    write_rows(args.output / "selector_targets.csv", targets, TARGET_FIELDS)
    write_rows(args.output / "rejected_contexts.csv", rejected, REJECTED_FIELDS)
    write_rows(args.output / "selector_predictions.csv", predictions, PREDICTION_FIELDS)
    payload = selector_payload(
        model, targets, rejected, manifest, hashes, root=ROOT,
        min_compliance_rate=args.min_compliance_rate, min_trials=args.min_trials,
        exploratory=args.exploratory, grids=grids if len(grids) > 1 else None)
    payload["grid_trials_path"] = [grid["trials_path"] for grid in grids]
    payload["grid_gaits"] = [grid["gaits"] for grid in grids]
    torch.save(payload, args.output / "unified_duty_selector.pt")
    per_gait = {
        gait: sorted({row["selected_df"] for row in predictions if row["gait"] == gait})
        for gait in GAITS}
    report = {
        "schema": "unified_duty_selector_fit_v1",
        "contexts": len(targets), "rejected_contexts": len(rejected),
        "contexts_per_gait": {g: sum(r["gait"] == g for r in targets) for g in GAITS},
        "periods": _periods(targets),
        "selected_levels_per_gait": per_gait,
        "final_training_cross_entropy": losses[-1],
        "fresh_selector_rollout_validation_required": True,
        "exploratory": args.exploratory,
        "grid_checkpoint_sha256": payload["grid_checkpoint_sha256"],
        "selector_checkpoint_sha256": hashlib.sha256(
            (args.output / "unified_duty_selector.pt").read_bytes()).hexdigest(),
    }
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2))
    plot_selector(predictions, rejected, args.output)
    plot_candidates(candidates, args.output, args.min_compliance_rate)
    print("UNIFIED_DUTY_SELECTOR_FIT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
