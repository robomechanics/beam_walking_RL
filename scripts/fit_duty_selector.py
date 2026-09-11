"""Fit and graph the flat-ground duty-factor selector from a completed grid."""

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

from beam_walking.experiment.duty_selector import (  # noqa: E402
    CONTEXT_FIELDS,
    PRIMARY_TRIALS_PER_CANDIDATE,
    aggregate_candidates,
    bootstrap_target_intervals,
    fit_selector,
    predict_rows,
    select_targets,
    selector_state_sha256,
    validate_completed_grid,
)
from beam_walking.experiment.protocol import (  # noqa: E402
    CONTROL_DT,
    GAITS,
    SPEED_RANGE,
    STEP_WIDTH_RANGE,
)

CANDIDATE_FIELDS = (
    *CONTEXT_FIELDS, "command_df", "trials", "compliant_trials",
    "finite_energy_compliant_trials", "compliance_rate",
    "finite_energy_compliant_rate",
    "median_positive_mechanical_cot",
)
TARGET_FIELDS = (
    *CONTEXT_FIELDS, "target_df", "target_positive_mechanical_cot",
    "target_compliance_rate", "eligible_candidates",
    "bootstrap_valid_fraction", "target_df_ci_low", "target_df_ci_high",
)
REJECTED_FIELDS = (*CONTEXT_FIELDS, "reason")
PREDICTION_FIELDS = (*TARGET_FIELDS, "selected_df")


def write_rows(path, rows, fieldnames):
    """Write a stable CSV schema, including for an empty result."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dense_segments(model, valid_rows, rejected_rows, gait, speed, period):
    """Evaluate the actual network without bridging an unsupported context gap."""
    valid_widths = sorted({
        row["step_width"] for row in valid_rows
        if (row["gait"], row["speed"], row["period"])
        == (gait, speed, period)
    })
    rejected_widths = {
        row["step_width"] for row in rejected_rows
        if (row["gait"], row["speed"], row["period"])
        == (gait, speed, period)
    }
    segments = []
    for lower, upper in zip(valid_widths[:-1], valid_widths[1:]):
        if any(lower < value < upper for value in rejected_widths):
            continue
        if upper - lower > .100001:
            continue
        widths = np.linspace(lower, upper, 51)
        rows = [{
            "step_width": float(width), "speed": speed,
            "period": period, "gait": gait,
        } for width in widths]
        prediction = predict_rows(model, rows)
        segments.append((widths, np.asarray([
            row["selected_df"] for row in prediction])))
    if len(valid_widths) == 1:
        row = {
            "step_width": valid_widths[0], "speed": speed,
            "period": period, "gait": gait,
        }
        segments.append((np.asarray(valid_widths), np.asarray(
            [predict_rows(model, [row])[0]["selected_df"]])))
    return segments


def plot_selector(model, predictions, rejected, output):
    """Plot fitted policy and labeled targets for every evaluated period."""
    periods = sorted({row["period"] for row in predictions + rejected})
    with PdfPages(output / "duty_factor_vs_step_width.pdf") as pdf:
        for period in periods:
            fig, axes = plt.subplots(1, len(GAITS), figsize=(11, 4), sharey=True)
            for axis, gait in zip(np.atleast_1d(axes), GAITS):
                gait_rows = [row for row in predictions
                             if row["gait"] == gait and row["period"] == period]
                gait_rejected = [row for row in rejected
                                 if row["gait"] == gait and row["period"] == period]
                speeds = sorted({row["speed"] for row in gait_rows + gait_rejected})
                for speed in speeds:
                    group = sorted(
                        (row for row in gait_rows if row["speed"] == speed),
                        key=lambda row: row["step_width"])
                    first_segment = True
                    for x, y in dense_segments(
                            model, predictions, rejected, gait, speed, period):
                        axis.plot(
                            x, y, linestyle="--",
                            label=(f"{speed:.2f} m/s fitted policy (unvalidated)"
                                   if first_segment else None))
                        first_segment = False
                    if group:
                        x = np.asarray([row["step_width"] for row in group])
                        target = np.asarray([row["target_df"] for row in group])
                        low = np.asarray([row["target_df_ci_low"] for row in group])
                        high = np.asarray([row["target_df_ci_high"] for row in group])
                        validity = np.asarray([
                            row["bootstrap_valid_fraction"] for row in group])
                        axis.errorbar(
                            x, target, yerr=np.vstack((
                                np.maximum(0, target - low),
                                np.maximum(0, high - target))),
                            fmt="o", capsize=3,
                            label=(f"{speed:.2f} m/s measured target "
                                   f"(bootstrap valid {validity.min():.2f}–"
                                   f"{validity.max():.2f})"),
                        )
                    rejected_widths = [
                        row["step_width"] for row in gait_rejected
                        if row["speed"] == speed]
                    if rejected_widths:
                        axis.scatter(
                            rejected_widths, [.505] * len(rejected_widths),
                            marker="x", color="red",
                            label=f"{speed:.2f} m/s abstained context")
                axis.set_title(gait.title())
                axis.set_xlabel("Commanded full step width (m)")
                axis.grid(alpha=.25)
                axis.set_ylim(.49, .76)
                axis.legend(fontsize=6)
            axes = np.atleast_1d(axes)
            axes[0].set_ylabel("Duty factor")
            fig.suptitle(
                f"Grid-derived duty-factor selector, period {period:.2f} s\n"
                "Dashed network output requires fresh rollout validation")
            fig.tight_layout()
            pdf.savefig(fig)
            filename = f"duty_factor_vs_step_width_p{period:.2f}.png"
            fig.savefig(output / filename, dpi=180)
            if len(periods) == 1:
                fig.savefig(output / "duty_factor_vs_step_width.png", dpi=180)
            plt.close(fig)


def plot_candidate_diagnostics(candidates, output, compliance_threshold):
    with PdfPages(output / "candidate_energy_and_compliance.pdf") as pdf:
        contexts = sorted({
            (row["gait"], row["speed"], row["period"])
            for row in candidates
        })
        for gait, speed, period in contexts:
            rows = [row for row in candidates if (
                row["gait"], row["speed"], row["period"]
            ) == (gait, speed, period)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for width in sorted({row["step_width"] for row in rows}):
                group = sorted(
                    (row for row in rows if row["step_width"] == width),
                    key=lambda row: row["command_df"])
                x = [row["command_df"] for row in group]
                axes[0].plot(
                    x, [row["median_positive_mechanical_cot"] for row in group],
                    marker="o", label=f"width {width:.2f} m")
                axes[1].plot(
                    x, [row["compliance_rate"] for row in group], marker="o")
            axes[0].set_ylabel("Median positive mechanical CoT")
            axes[0].legend(fontsize=8)
            axes[1].set_ylabel("Compliant-trial fraction")
            axes[1].axhline(
                compliance_threshold, color="black", linestyle="--", linewidth=1)
            for axis in axes:
                axis.set_xlabel("Commanded duty factor")
                axis.grid(alpha=.25)
            fig.suptitle(
                f"{gait.title()}, speed {speed:.2f} m/s, period {period:.2f} s")
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
    if not args.exploratory and args.min_trials != PRIMARY_TRIALS_PER_CANDIDATE:
        parser.error("Primary fitting requires 32 trials per candidate")
    if not args.exploratory and not np.isclose(
            args.min_compliance_rate, .90, rtol=0, atol=1e-12):
        parser.error("Primary fitting requires the registered 0.90 compliance gate")
    if not args.exploratory and (
            args.bootstrap_samples != 1000
            or args.epochs != 3000
            or args.seed != 0):
        parser.error(
            "Primary fitting requires 1000 bootstrap samples, 3000 epochs, "
            "and seed 0; use --exploratory for development settings")
    args.output.mkdir(parents=True, exist_ok=True)

    rows, grid_manifest, source_hashes = validate_completed_grid(
        args.trials, require_primary=not args.exploratory)
    candidates = aggregate_candidates(rows)
    targets, rejected = select_targets(
        rows, args.min_compliance_rate, args.min_trials)
    if len(targets) < 2:
        raise ValueError("Too few valid contexts to train the selector")
    intervals = bootstrap_target_intervals(
        rows, args.min_compliance_rate, args.min_trials,
        samples=args.bootstrap_samples, seed=args.seed)
    interval_by_context = {
        (row["step_width"], row["speed"], row["period"], row["gait"]): row
        for row in intervals
    }
    targets = [dict(row, **interval_by_context[
        (row["step_width"], row["speed"], row["period"], row["gait"])])
        for row in targets]
    model, losses = fit_selector(targets, seed=args.seed, epochs=args.epochs)
    predictions = predict_rows(model, targets)

    write_rows(args.output / "candidate_summary.csv", candidates, CANDIDATE_FIELDS)
    write_rows(args.output / "selector_targets.csv", targets, TARGET_FIELDS)
    write_rows(args.output / "rejected_contexts.csv", rejected, REJECTED_FIELDS)
    write_rows(
        args.output / "selector_predictions.csv", predictions, PREDICTION_FIELDS)
    selector_source_sha256 = hashlib.sha256(
        (ROOT / "source/beam_walking/beam_walking/experiment/duty_selector.py").read_bytes()
    ).hexdigest()
    checkpoint_payload = {
        "schema": "flat_duty_selector_v1",
        "state_dict": model.state_dict(), "hidden_dims": [32, 32],
        "input_fields": ["step_width", "speed", "period", "gait_id"],
        "input_ranges": {
            "step_width": list(STEP_WIDTH_RANGE), "speed": list(SPEED_RANGE),
            "period": [.36, .54]},
        "gait_id_mapping": {name: index for index, name in enumerate(GAITS)},
        "control_dt_s": CONTROL_DT,
        "output": "feasibility_bounded_duty_factor",
        "label_rule": (
            "minimum median positive mechanical CoT among candidates with "
            "sufficient finite-energy compliant trials"),
        "minimum_compliance_rate": args.min_compliance_rate,
        "minimum_trials_per_candidate": args.min_trials,
        "supported_contexts": [
            {field: row[field] for field in CONTEXT_FIELDS} for row in targets],
        "rejected_contexts": rejected,
        "requires_fresh_rollout_validation": True,
        "deployment_ready": False,
        "exploratory": args.exploratory,
        "selector_source_sha256": selector_source_sha256,
        "selector_state_sha256": selector_state_sha256(model.state_dict()),
        **source_hashes,
        "grid_task_sha256": grid_manifest["task_sha256"],
        "grid_checkpoint_sha256": grid_manifest["checkpoint_sha256"],
        "grid_collector_sha256": grid_manifest["collector_sha256"],
    }
    torch.save(checkpoint_payload, args.output / "duty_selector.pt")
    errors = np.asarray([
        row["selected_df"] - row["target_df"] for row in predictions])
    report = {
        "schema": "flat_duty_selector_fit_v1",
        "trial_source": str(args.trials.resolve()), **source_hashes,
        "contexts": len(targets), "rejected_contexts": len(rejected),
        "minimum_compliance_rate": args.min_compliance_rate,
        "minimum_trials_per_candidate": args.min_trials,
        "bootstrap_samples": args.bootstrap_samples,
        "training_epochs": args.epochs, "seed": args.seed,
        "final_training_mse": losses[-1],
        "resubstitution_label_fit_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "generalization_claimed": False,
        "fresh_selector_rollout_validation_required": True,
        "scientific_scope": "flat_ground_commanded_step_width",
        "terrain_width_input": False, "stability_chi_used": False,
        "exploratory": args.exploratory,
    }
    (args.output / "fit_report.json").write_text(json.dumps(report, indent=2))
    plot_selector(model, predictions, rejected, args.output)
    plot_candidate_diagnostics(
        candidates, args.output, args.min_compliance_rate)
    print("DUTY_SELECTOR_FIT", json.dumps(report))


if __name__ == "__main__":
    main()
