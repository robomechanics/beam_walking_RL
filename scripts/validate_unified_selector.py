"""Promote a unified duty-factor selector from fresh rollout evidence.

Reads the selector validation collection written by
``collect_policy_surfaces.py --task unified --selector-checkpoint`` and checks
every supported context: at least 90% compliant trials, at least 90% finite
energy compliant trials, and median achieved duty factor within 0.05 of the
selected value. When all contexts pass, the selector is re-saved next to the
evidence as ``validated_unified_duty_selector.pt`` with deployment_ready=True.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
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
    GAIT_LEVELS,
    VALIDATION_COMPLETE_SCHEMA,
    VALIDATION_SCHEMA,
    load_selector,
    predict_supported_rows,
    read_trial_rows,
)

GAIT_COLORS = {"trot": "#0072BD", "walk": "#D95319"}
SPEED_MARKERS = {.25: "o", .30: "s", .35: "^", .40: "D"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot(summaries, output):
    periods = sorted({round(row["period"], 6) for row in summaries})
    fig, grid = plt.subplots(len(GAITS), len(periods),
                             figsize=(4.2 * len(periods) + 1, 8.4), squeeze=False)
    panels = [(grid[r, c], gait, period) for r, gait in enumerate(GAITS)
              for c, period in enumerate(periods)]
    for axis, gait, period in panels:
        rows = [row for row in summaries if row["gait"] == gait
                and abs(row["period"] - period) < 1e-6]
        if not rows:
            axis.set_axis_off()
            continue
        for speed in sorted({row["speed"] for row in rows}):
            group = sorted((row for row in rows if row["speed"] == speed),
                           key=lambda row: row["step_width"])
            width = np.asarray([row["step_width"] for row in group])
            selected = np.asarray([row["selected_df"] for row in group])
            achieved = np.asarray([row["achieved_df_median"] for row in group])
            low = np.asarray([row["achieved_df_q025"] for row in group])
            high = np.asarray([row["achieved_df_q975"] for row in group])
            marker = SPEED_MARKERS.get(speed, "o")
            alpha = min(1., .45 + .55 * (speed - .25) / .15)
            axis.plot(width, selected, linestyle="--", marker=marker, markerfacecolor="white",
                      color=GAIT_COLORS[gait], alpha=alpha, linewidth=1.4, markersize=7)
            axis.errorbar(width, achieved, yerr=np.vstack((achieved - low, high - achieved)),
                          fmt=f"{marker}-", color=GAIT_COLORS[gait], alpha=alpha, capsize=3,
                          linewidth=2, markersize=6, label=f"{speed:.2f} m/s")
        levels = GAIT_LEVELS[gait]
        axis.set_yticks(levels)
        axis.set_ylim(min(levels) - .04, max(levels) + .06)
        axis.set_xlabel("Commanded full step width (m)")
        axis.set_ylabel("Duty factor (dashed: selected, solid: achieved)")
        axis.set_title(f"{gait.title()}, period {period:.2f} s")
        axis.grid(alpha=.25)
        axis.legend(title="Speed", frameon=False)
    fig.tight_layout()
    fig.savefig(output / "selected_vs_achieved_duty_factor.png", dpi=200)
    fig.savefig(output / "selected_vs_achieved_duty_factor.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True, nargs="+",
                        help="Output directory (one per low-level checkpoint) of the "
                             "selector validation collection")
    parser.add_argument("--output", type=Path,
                        help="Where to write the promoted selector and copied evidence "
                             "(default: the single validation directory)")
    parser.add_argument("--min-compliance-rate", type=float, default=.90)
    parser.add_argument("--max-df-error", type=float, default=.05)
    args = parser.parse_args()
    model, payload = load_selector(args.selector)
    if payload.get("exploratory") is not False or payload.get("deployment_ready") is not False:
        parser.error("Only a primary, not-yet-validated selector can be promoted")
    if args.output is None:
        if len(args.validation) != 1:
            parser.error("--output is required with more than one validation directory")
        args.output = args.validation[0]
    selector_sha = sha256(args.selector)
    grid_checkpoints = payload["grid_checkpoint_sha256"]
    rows, evidence, manifests = [], [], []
    for directory in args.validation:
        manifest_path = directory / "surface_manifest.json"
        trials_path = directory / "surface_trials.csv"
        complete_path = directory / "SURFACE_COMPLETE"
        manifest = json.loads(manifest_path.read_text())
        completion = json.loads(complete_path.read_text())
        gaits = sorted(manifest.get("gaits") or {e["gait"] for e in manifest["conditions"]})
        expected_ckpt = (grid_checkpoints if not isinstance(grid_checkpoints, dict)
                         else grid_checkpoints.get(gaits[0]))
        if (manifest.get("schema") != VALIDATION_SCHEMA
                or manifest.get("smoke") is not False
                or manifest.get("selector_checkpoint_sha256") != selector_sha
                or manifest.get("checkpoint_sha256") != expected_ckpt
                or (isinstance(grid_checkpoints, dict)
                    and any(grid_checkpoints.get(g) != expected_ckpt for g in gaits))
                or completion.get("schema") != VALIDATION_COMPLETE_SCHEMA
                or completion.get("manifest_sha256") != sha256(manifest_path)
                or completion.get("trials_sha256") != sha256(trials_path)):
            raise ValueError(f"Validation evidence in {directory} does not match this selector and policy")
        for name, digest in completion["archive_sha256"].items():
            if sha256(directory / name) != digest:
                raise ValueError(f"Validation archive hash mismatch: {name}")
        rows.extend(read_trial_rows(trials_path))
        manifests.append(manifest)
        evidence.append({
            "source_directory": str(directory.resolve()), "gaits": gaits,
            "directory": f"validation_{'_'.join(gaits)}",
            "manifest_sha256": sha256(manifest_path),
            "trials_sha256": sha256(trials_path),
            "completion_sha256": sha256(complete_path),
            "low_level_checkpoint_sha256": manifest["checkpoint_sha256"],
            "seed_start": manifest["seed_start"],
            "trials_per_context": manifest["trials_per_condition"],
        })
    manifest = manifests[0]
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    contexts = payload["supported_contexts"]
    predictions = predict_supported_rows(
        model, payload, [dict(row) for row in contexts],
        require_deployment_ready=False)
    selected_by_context = {
        (row["step_width"], row["speed"], round(row["period"], 6), row["gait"]):
            row["selected_df"] for row in predictions}
    grouped = {}
    for row in rows:
        key = (row["step_width"], row["speed"], round(row["period"], 6), row["gait"])
        if key not in selected_by_context:
            raise ValueError(f"Validation contains an unsupported context: {key}")
        if abs(row["command_df"] - selected_by_context[key]) > 1e-6:
            raise ValueError(f"Validation command DF differs from the selector at {key}")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(selected_by_context):
        raise ValueError("Validation does not cover every supported context exactly once")
    summaries, all_passed = [], True
    for key, trials in sorted(grouped.items()):
        achieved = np.asarray([float(row["achieved_df"]) for row in trials])
        compliant = np.asarray([row["compliant"] for row in trials])
        finite = np.asarray([np.isfinite(row["positive_mechanical_cot"]) for row in trials])
        selected = selected_by_context[key]
        summary = {
            "gait": key[3], "speed": key[1], "period": key[2], "step_width": key[0],
            "selected_df": selected,
            "achieved_df_median": float(np.median(achieved)),
            "achieved_df_q025": float(np.quantile(achieved, .025)),
            "achieved_df_q975": float(np.quantile(achieved, .975)),
            "compliance_rate": float(compliant.mean()),
            "finite_energy_compliant_rate": float((compliant & finite).mean()),
            "trials": len(trials),
        }
        summary["passed"] = int(
            summary["compliance_rate"] >= args.min_compliance_rate
            and summary["finite_energy_compliant_rate"] >= args.min_compliance_rate
            and abs(summary["achieved_df_median"] - selected) <= args.max_df_error)
        all_passed &= bool(summary["passed"])
        summaries.append(summary)
    with (directory / "selector_validation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot(summaries, directory)
    report = {
        "schema": "unified_selector_validation_report_v1",
        "contexts": len(summaries), "passed_contexts": int(sum(s["passed"] for s in summaries)),
        "all_contexts_passed": bool(all_passed),
        "max_selected_vs_achieved_error": float(max(
            abs(s["achieved_df_median"] - s["selected_df"]) for s in summaries)),
        "selected_levels_per_gait": {
            g: sorted({s["selected_df"] for s in summaries if s["gait"] == g}) for g in GAITS},
    }
    if all_passed:
        validated = dict(payload)
        for item in evidence:
            # Copy the evidence next to the promoted selector so the binding
            # in load_selector is self-contained.
            source = Path(item["source_directory"])
            target_dir = directory / item["directory"]
            if source.resolve() != target_dir.resolve():
                target_dir.mkdir(parents=True, exist_ok=True)
                for name in ("surface_manifest.json", "surface_trials.csv", "SURFACE_COMPLETE"):
                    shutil.copy2(source / name, target_dir / name)
        validated.update({
            "deployment_ready": True,
            "validation_evidence": evidence,
            "validation_selector_checkpoint_sha256": selector_sha,
            "validation_all_contexts_passed": True,
        })
        target = directory / "validated_unified_duty_selector.pt"
        torch.save(validated, target)
        load_selector(target)  # Round-trip check of the evidence binding.
        report["validated_selector"] = str(target.resolve())
    (directory / "validation_report.json").write_text(json.dumps(report, indent=2))
    print("UNIFIED_SELECTOR_VALIDATION", json.dumps(report), flush=True)
    if not all_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
