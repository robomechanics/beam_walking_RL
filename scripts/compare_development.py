"""CPU-only comparison of measured development validation summary CSVs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
STAGES = (("r2_50", "R2 checkpoint 50"), ("r2_99", "R2 checkpoint 99"),
          ("r3_199", "R3 checkpoint 199"))
LEGS = ("FL", "FR", "RL", "RR")


def compare(results, output):
    frames, sources = [], []
    reference = None
    for stage, label in STAGES:
        for gait in ("trot", "walk"):
            directory = results / f"validation_{stage}_{gait}"
            path = directory / "summary.csv"
            manifest_path = directory / "evaluation_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            data = pd.read_csv(path)
            if not (data.gait.eq(gait).all() and data.period.eq(.48).all()
                    and data.command_df.eq(.625).all() and data.disturbed.eq(False).all()
                    and np.allclose(data.command_speed, .3)):
                raise ValueError(f"Unexpected diagnostic condition: {path}")
            matching = (tuple(data.step_width), tuple(data.n), tuple(manifest["seeds"]))
            if reference is None:
                reference = matching
            elif matching != reference:
                raise ValueError(f"Width/trial/seed mismatch: {path}")
            if not data.compliant_trial_fraction.eq(0).all():
                raise ValueError("Update the development-status caption: compliance is no longer zero")
            data.insert(0, "development_checkpoint", label)
            data.insert(1, "stage", stage)
            data["source_summary"] = str(path.relative_to(results))
            data["checkpoint_sha256"] = manifest["checkpoint_sha256"]
            frames.append(data)
            sources.append({"summary": str(path.resolve()),
                            "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                            "checkpoint_sha256": manifest["checkpoint_sha256"],
                            "source_sha256": manifest["source_sha256"]})
    table = pd.concat(frames, ignore_index=True)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "condition_comparison.csv", index=False)
    overview = []
    for (label, gait), group in table.groupby(["development_checkpoint", "gait"], sort=False):
        ordered = group.sort_values("step_width")
        overview.append({"development_checkpoint": label, "gait": gait,
            "trials": int(group.n.sum()), "completions": int(group.successes.sum()),
            "physical_failures": int(group.failures.sum()), "timeouts": int(group.timeouts.sum()),
            "compliant_trials": int(np.rint((group.n * group.compliant_trial_fraction).sum())),
            "width_response_ratio": float((ordered.achieved_width.iloc[-1] - ordered.achieved_width.iloc[0])
                                          / (ordered.step_width.iloc[-1] - ordered.step_width.iloc[0])),
            "mean_lateral_rmse_m": float(np.average(group.lateral_rmse, weights=group.n)),
            "mean_forward_speed_m_s": float(np.average(group.forward_speed, weights=group.n))})
    overview = pd.DataFrame(overview)
    overview.to_csv(output / "checkpoint_overview.csv", index=False)
    colors = ("#2b6cb0", "#d97706", "#15803d")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for row, gait in enumerate(("trot", "walk")):
        for (stage, label), color in zip(STAGES, colors):
            group = table[(table.stage == stage) & (table.gait == gait)].sort_values("step_width")
            axes[row, 0].plot(group.step_width, group.achieved_width, "o-", color=color, label=label)
            axes[row, 1].plot(group.step_width, group.lateral_rmse, "o-", color=color, label=label)
            # Equal trial counts: mean across the three tested width conditions.
            means = [np.average(group[f"df_{leg}"], weights=group.n) for leg in LEGS]
            axes[row, 2].plot(range(4), means, "o-", color=color, label=label)
        axes[row, 0].plot([.1, .5], [.1, .5], "k--", lw=1.2, label="Command identity")
        axes[row, 0].set(xlim=(.08, .52), ylim=(.08, .52), xlabel="Commanded step width (m)",
                         ylabel="Achieved stance width (m)", title=f"{gait.title()}: width execution")
        axes[row, 1].axhline(0., color="black", ls="--", lw=1.2, label="Straight-path reference")
        axes[row, 1].set(xlabel="Commanded step width (m)", ylabel="Lateral RMSE (m)",
                         title=f"{gait.title()}: straightness")
        axes[row, 2].axhline(.625, color="black", ls="--", lw=1.2, label="Commanded DF = 0.625")
        axes[row, 2].set(xticks=range(4), xticklabels=LEGS, ylim=(.5, .95), ylabel="Achieved duty factor",
                         title=f"{gait.title()}: per-leg DF, widths averaged")
        for ax in axes[row]:
            ax.legend(fontsize=7, loc="best")
    fig.suptitle("Development checkpoints — command execution remains incomplete", fontsize=14)
    fig.text(.5, .015,
        "Nominal flat ground; period 0.48 s; speed command 0.30 m/s; DF command 0.625; "
        "8 shared validation seeds per width/gait.\n"
        "Zero fully compliant trials in every run. Condition means only; no convergence or final-test claim.",
        ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .95))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"development_comparison.{ext}", dpi=200)
    plt.close(fig)
    provenance = {"purpose": "Development diagnostic comparison, not final results",
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "sources": sources, "output_rows": len(table)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "README.md").write_text(
        "# Development checkpoint comparison\n\n"
        "Measured nominal validation only: R2 checkpoints 50/99 and R3 checkpoint 199, "
        "trot/walk, widths 0.10/0.30/0.50 m, DF 0.625, period 0.48 s, forward speed 0.30 m/s. "
        "Each condition uses the same eight validation seed IDs. No final test seeds are included.\n\n"
        "All six runs have zero fully compliant trials. Completion alone does not establish "
        "straightness or execution of the commanded width and DF. These checkpoints are "
        "development diagnostics; no convergence or final-controller claim is supported.\n\n"
        "The figure's width identity line represents perfect command execution. The straightness "
        "reference is zero lateral RMSE. Per-leg DF values average the three width-condition "
        "means with their trial counts; the dashed line is commanded DF. No error bars are "
        "inferred from condition means. condition_comparison.csv retains each width separately; "
        "checkpoint_overview.csv describes aggregate counts and the observed width-response "
        "ratio (achieved-width span / commanded-width span).\n\n"
        "Source summary CSV and evaluation-manifest hashes are recorded in provenance.json.\n\n"
        "Reproduce from the repository root:\n\n"
        "```bash\nMPLCONFIGDIR=/tmp/development_comparison_mpl "
        "/home/rml2/anaconda3/envs/isaaclab/bin/python scripts/compare_development.py\n```\n")
    print(overview.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results/development_comparison")
    args = parser.parse_args()
    compare(args.results, args.output)
