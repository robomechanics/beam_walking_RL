"""Fit and graph the high-duty walk selector from its completed grid."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.high_duty_walk import (  # noqa: E402
    HIGH_DUTY_WALK_LEVELS,
    HIGH_DUTY_WALK_PERIOD,
)
from beam_walking.experiment.high_duty_walk_selector import (  # noqa: E402
    CONTEXT_FIELDS,
    PRIMARY_TRIALS_PER_CANDIDATE,
    aggregate_candidates,
    bootstrap_target_intervals,
    fit_selector,
    predict_rows,
    select_targets,
    selector_runtime_sha256,
    selector_state_sha256,
    validate_completed_grid,
)
from beam_walking.experiment.protocol import SPEED_RANGE, STEP_WIDTH_RANGE  # noqa: E402

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
COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")
MARKERS = ("o", "s", "^", "D")


def write_rows(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_selector(predictions, rejected, output):
    """Plot one labeled curve per speed; targets use matching markers."""
    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for index, speed in enumerate(sorted({
            row["speed"] for row in predictions + rejected})):
        group = sorted(
            (row for row in predictions if row["speed"] == speed),
            key=lambda row: row["step_width"])
        color, marker = COLORS[index], MARKERS[index]
        if group:
            widths = np.asarray([row["step_width"] for row in group])
            selected = np.asarray([row["selected_df"] for row in group])
            axis.scatter(
                widths, selected, color=color, marker=marker,
                label=f"{speed:.2f} m/s", zorder=3)
            rejected_for_speed = {
                row["step_width"] for row in rejected if row["speed"] == speed}
            for left, right in zip(group[:-1], group[1:]):
                if (right["step_width"] - left["step_width"] <= .100001
                        and not any(left["step_width"] < width < right["step_width"]
                                    for width in rejected_for_speed)):
                    axis.plot(
                        [left["step_width"], right["step_width"]],
                        [left["selected_df"], right["selected_df"]],
                        color=color, linewidth=2)
            x = np.asarray([row["step_width"] for row in group])
            target = np.asarray([row["target_df"] for row in group])
            lo = np.asarray([row["target_df_ci_low"] for row in group])
            hi = np.asarray([row["target_df_ci_high"] for row in group])
            axis.errorbar(
                x, target, yerr=np.vstack((
                    np.where(np.isfinite(lo), np.maximum(0, target - lo), 0),
                    np.where(np.isfinite(hi), np.maximum(0, hi - target), 0))),
                fmt=marker, color=color, capsize=3, markersize=4,
                linestyle="none")
        rejected_widths = [row["step_width"] for row in rejected
                           if row["speed"] == speed]
        if rejected_widths:
            axis.scatter(rejected_widths, [.745] * len(rejected_widths),
                         marker="x", color=color,
                         label=f"{speed:.2f} m/s" if not group else None)
    axis.set_xlabel("Commanded full step width (m)")
    axis.set_ylabel("Selected duty factor")
    axis.set_ylim(.74, .91)
    axis.set_xlim(STEP_WIDTH_RANGE)
    axis.set_yticks(HIGH_DUTY_WALK_LEVELS)
    axis.grid(alpha=.25)
    axis.legend(title="Commanded speed", frameon=False, ncol=2)
    axis.set_title(
        "High-duty walk selector fit, period 0.48 s\n"
        "Exact grid contexts; fresh rollout validation pending")
    fig.tight_layout()
    fig.savefig(output / "duty_factor_vs_step_width.png", dpi=180)
    with PdfPages(output / "duty_factor_vs_step_width.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def plot_candidates(candidates, output, threshold):
    with PdfPages(output / "candidate_energy_and_compliance.pdf") as pdf:
        for speed in sorted({row["speed"] for row in candidates}):
            rows = [row for row in candidates if row["speed"] == speed]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for width in sorted({row["step_width"] for row in rows}):
                group = sorted(
                    (row for row in rows if row["step_width"] == width),
                    key=lambda row: row["command_df"])
                axes[0].plot(
                    [row["command_df"] for row in group],
                    [row["median_positive_mechanical_cot"] for row in group],
                    marker="o", label=f"width {width:.2f} m")
                axes[1].plot(
                    [row["command_df"] for row in group],
                    [row["compliance_rate"] for row in group], marker="o")
            axes[0].set_ylabel("Median positive mechanical CoT")
            axes[0].legend(fontsize=8)
            axes[1].set_ylabel("Compliant-trial fraction")
            axes[1].axhline(threshold, color="black", linestyle="--")
            for axis in axes:
                axis.set_xlabel("Commanded duty factor")
                axis.set_xticks(HIGH_DUTY_WALK_LEVELS)
                axis.grid(alpha=.25)
            fig.suptitle(f"Walk, {speed:.2f} m/s, period 0.48 s")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trials", type=Path)
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

    rows, manifest, hashes = validate_completed_grid(
        args.trials, require_primary=not args.exploratory)
    candidates = aggregate_candidates(rows)
    targets, rejected = select_targets(
        rows, args.min_compliance_rate, args.min_trials)
    if len(targets) < 2:
        raise ValueError("Too few compliant contexts to fit the selector")
    intervals = bootstrap_target_intervals(
        rows, args.min_compliance_rate, args.min_trials,
        samples=args.bootstrap_samples, seed=args.seed)
    interval_by_context = {
        tuple(row[field] for field in CONTEXT_FIELDS): row
        for row in intervals
    }
    targets = [dict(row, **interval_by_context[
        tuple(row[field] for field in CONTEXT_FIELDS)]) for row in targets]
    model, losses = fit_selector(targets, seed=args.seed, epochs=args.epochs)
    predictions = predict_rows(model, targets)
    if any(row["selected_df"] != row["target_df"] for row in predictions):
        raise RuntimeError(
            "Classifier did not reproduce every registered grid label exactly")

    write_rows(args.output / "candidate_summary.csv", candidates, CANDIDATE_FIELDS)
    write_rows(args.output / "selector_targets.csv", targets, TARGET_FIELDS)
    write_rows(args.output / "rejected_contexts.csv", rejected, REJECTED_FIELDS)
    write_rows(args.output / "selector_predictions.csv", predictions,
               PREDICTION_FIELDS)
    source_path = (ROOT / "source/beam_walking/beam_walking/experiment/"
                   "high_duty_walk_selector.py")
    payload = {
        "schema": "high_duty_walk_selector_v1",
        "state_dict": model.state_dict(), "hidden_dims": [32, 32],
        "input_fields": ["step_width", "speed"],
        "input_ranges": {
            "step_width": list(STEP_WIDTH_RANGE), "speed": list(SPEED_RANGE)},
        "fixed_gait": "walk", "fixed_period_s": HIGH_DUTY_WALK_PERIOD,
        "output_range": list((HIGH_DUTY_WALK_LEVELS[0],
                              HIGH_DUTY_WALK_LEVELS[-1])),
        "label_rule": (
            "minimum median positive mechanical CoT among candidates passing "
            "the registered compliance and finite-energy gates"),
        "minimum_compliance_rate": args.min_compliance_rate,
        "minimum_trials_per_candidate": args.min_trials,
        "supported_contexts": [
            {"step_width": row["step_width"], "speed": row["speed"]}
            for row in targets],
        "rejected_contexts": rejected,
        "deployment_ready": False,
        "requires_fresh_rollout_validation": True,
        "exploratory": args.exploratory,
        "selector_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "selector_runtime_sha256": selector_runtime_sha256(ROOT),
        "selector_state_sha256": selector_state_sha256(model.state_dict()),
        **hashes,
        "grid_task_sha256": manifest["task_sha256"],
        "grid_checkpoint_sha256": manifest["checkpoint_sha256"],
        "grid_collector_sha256": manifest["collector_sha256"],
    }
    torch.save(payload, args.output / "high_duty_walk_selector.pt")
    errors = np.asarray([row["selected_df"] - row["target_df"]
                         for row in predictions])
    report = {
        "schema": "high_duty_walk_selector_fit_v1",
        "contexts": len(targets), "rejected_contexts": len(rejected),
        "final_training_cross_entropy": losses[-1],
        "resubstitution_label_fit_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "fresh_selector_rollout_validation_required": True,
        "exploratory": args.exploratory,
    }
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2))
    plot_selector(predictions, rejected, args.output)
    plot_candidates(candidates, args.output, args.min_compliance_rate)
    print("HIGH_DUTY_WALK_SELECTOR_FIT", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
