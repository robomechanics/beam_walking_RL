"""Copy the paper-relevant subset of ``results/`` into the tracked ``paper_data/``.

``results/`` is gitignored and holds about 8 GB of development runs, smoke
tests, videos, and raw per-condition ``.npz`` archives. This script collects
only what the manuscript's learned-locomotion section rests on: final
checkpoints, training provenance, evaluation manifests,
per-trial and per-condition tables, selector fits, and the push-study
results. Raw ``.npz`` archives (about 1.3 GB) are left in ``results/`` and
listed in the manifest; per-trial CSV tables are exported from them where no
table existed. Run from the repository root; no simulator or GPU is used.
"""
import csv
import fnmatch
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper_data"
CONTROL_DT = .02
SETTLE_S = 2.0

TRAINING_FILES = ["model_1799.pt", "provenance.json", "parent_provenance.json", "agent.yaml",
                  "env.yaml", "throughput.json", "capacity.json",
                  "settled_stance.json", "watcher_final.json"]
EVAL_FILES = ["*.csv", "*.json", "*.md", "*.pdf", "*.png", "*.pt", "*.py", "*.sh",
              "GRID_COMPLETE", "SURFACE_COMPLETE", "SELECTOR_VALIDATION_COMPLETE"]
EVAL_SKIP = ["*.partial.csv", "return_map_index.json"]

# destination -> list of (source directory or file, patterns, export per-trial CSV from npz)
PLAN = {
    "policy_seed2": [
        ("paper_ppo_forcefix_nominalgains_seed2_3072_20260911", TRAINING_FILES, False)],
    "command_fidelity_seed2": [
        ("validation_v4_seed2_v030_p048_trot_20260911", EVAL_FILES, False),
        ("validation_v4_seed2_v030_p048_walk_20260911", EVAL_FILES, False)],
    "extended_grid_seed2": [
        ("old_policy_walk_df_seed2_20260911", ["*.json"], False),
        ("old_policy_walk_df_seed2_20260911/grid", EVAL_FILES, False)],
    "energy_and_return_map_seed2": [
        ("stability_v4_seed2_descriptive_20260911", EVAL_FILES, False),
        ("stability_v5_estimator_pilot_seed1100000_20260911", EVAL_FILES, False)],
    "adaptive_selector_seed3": [
        ("adaptive_duty_seed3_3072_20260911", TRAINING_FILES, False),
        ("adaptive_duty_grid_seed3_20260911", EVAL_FILES, False),
        ("adaptive_duty_selector_seed3_20260911", EVAL_FILES, False),
        ("adaptive_duty_selector_validation_seed3_20260911", EVAL_FILES, False)],
    "high_duty_walk_seed4": [
        ("high_duty_walk_seed4_3072_20260912", TRAINING_FILES, False),
        ("high_duty_walk_grid_seed4_20260912", EVAL_FILES, False),
        ("high_duty_walk_selector_seed4_20260912", EVAL_FILES, False),
        ("high_duty_walk_selector_validation_seed4_20260912", EVAL_FILES, False),
        ("high_duty_walk_pipeline_20260912.json", None, False)],
    "narrow_stance_lineage": [
        ("narrow_beam_center_from_seed2_3072_20260912", TRAINING_FILES, False),
        ("narrow_beam_tight2_from_center_3072_20260912", TRAINING_FILES, False),
        ("narrow_beam_tight3_center10_from_tight2_3072_20260912", TRAINING_FILES, False),
        ("narrow_beam_tight3_perturb_from_tight3_3072_20260913", TRAINING_FILES, False),
        ("validation_narrow_center_v030_p048_trot_20260912", EVAL_FILES, True),
        ("validation_narrow_tight2_v030_p048_trot_20260912", EVAL_FILES, True)],
    "push_study": [
        ("validation_tight3_nominal_20260913", EVAL_FILES, True),
        ("validation_tight3_perturbed_20260913", EVAL_FILES, True),
        ("validation_tight3perturb_nominal_20260913", EVAL_FILES, True),
        ("validation_tight3perturb_perturbed_20260913", EVAL_FILES, True),
        ("perturbation_results_20260913", EVAL_FILES, False),
        ("perturbation_smoke_20260913", EVAL_FILES, False),
        ("perturbation_pipeline_20260913.sh", None, False)],
}
DOCS = ["experiment_summary.md", "paper_claim_coverage.md", "paper_aligned_evaluation.md",
        "perturbation.md", "adaptive_duty_selector_plan.md", "high_duty_walk_plan.md",
        "old_policy_surface_plan.md", "deployment_dr_plan.md", "policy_quad_sdk_port.md"]
NOT_COLLECTED = [
    ("results/**/*.log, results/**/events.out.tfevents.*",
     "training and pipeline console logs and TensorBoard event files (about 50 MB); "
     "the numbers the section uses are in the CSV/JSON tables"),
    ("results/*/**.npz", "raw per-condition trial archives (about 1.3 GB for the paper runs); "
                        "per-trial CSVs are exported instead"),
    ("results/deployment_gain_sweep*", "sim-to-sim motor-gain sweep (4.4 GB); not used in the section"),
    ("results/deployment_dr_*", "deployment domain-randomization smoke and solver runs"),
    ("results/flat_train_42*, results/train_42, results/validation_r[2-9]_*",
     "development lineage R1-R9 superseded by the seed-2 policy"),
    ("results/validation_seed0_*, results/paper_ppo_seed0_*, results/paper_ppo_phasefix_*, "
     "results/validation_phasefix_*, results/validation_forcefix_*",
     "earlier seed-0/seed-1 pilots and the pre-audit seed-2 evaluation"),
    ("results/narrow_beam_from_seed2*, results/narrow_beam_hdg5*, results/narrow_beam_tight_from_seed2*",
     "abandoned branches of the narrow-stance lineage"),
    ("results/*video*, results/*.mp4, results/port_*, results/sim2sim_*",
     "videos and ONNX exports for the hardware port"),
    ("results/benchmark_*, results/*smoke*, results/preflight, results/diagnostic_*",
     "throughput benchmarks and smoke tests"),
]


def per_trial_rows(path):
    with np.load(path) as d:
        valid = d["valid"].astype(bool)
        failure = d["failure"].astype(bool)
        success = d["success"].astype(bool)
        time = d["time"]
        contacts = d["contacts"]
        feet_y = d["feet_body"][:, :, :, 1]
        body = d["body"]
        speed = d["forward_velocity"]
        seeds = d["seeds"]
        width, df = float(d["step_width"]), float(d["df"])
        envelope = d["perturbation_envelope"] if "perturbation_envelope" in d else None
    rows = []
    for i in range(valid.shape[1]):
        last = int(np.flatnonzero(valid[:, i])[-1])
        failed = bool(failure[last, i])
        succeeded = bool(success[last, i]) and not failed
        m = valid[:, i] & (time[:, i] > SETTLE_S)
        y, c = feet_y[m, i], contacts[m, i]
        pairs = [y[c[:, a], a].mean() - y[c[:, b], b].mean()
                 for a, b in ((0, 1), (2, 3)) if c[:, a].any() and c[:, b].any()]
        rows.append({
            "archive": path.name, "seed": int(seeds[i]), "command_width_m": width, "command_df": df,
            "success": int(succeeded), "failure": int(failed),
            "timeout": int(not succeeded and not failed), "end_time_s": round(last * CONTROL_DT, 2),
            "forward_travel_m": round(float(body[last, i, 0] + .65), 4),
            "realized_df": round(float(contacts[m, i].mean()), 4),
            "realized_width_m": round(float(np.mean(pairs)), 4) if pairs else "",
            "lateral_rmse_m": round(float(np.sqrt((body[m, i, 1] ** 2).mean())), 4),
            "forward_speed_mps": round(float(speed[m, i].mean()), 4),
            "envelope_at_end": round(float(envelope[last, i]), 4) if envelope is not None else "",
        })
    return rows


def copy_matching(source, patterns, destination):
    copied = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name, skip) for skip in EVAL_SKIP):
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination / path.name)
            copied.append(destination / path.name)
    return copied


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    manifest = ["# Curated paper data", "",
                f"Collected on {date.today().isoformat()} by `scripts/collect_paper_data.py` "
                "from the local, untracked `results/` tree. Sizes are of the copied files.", ""]
    total = 0
    for group, entries in PLAN.items():
        manifest += [f"## `{group}/`", "", "| Source in `results/` | Files | Size (MB) |", "|---|---:|---:|"]
        for source, patterns, export in entries:
            src = RESULTS / source
            dest = OUT / group / Path(source).name
            if src.is_file():
                (OUT / group).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, OUT / group / src.name)
                copied = [OUT / group / src.name]
            else:
                copied = copy_matching(src, patterns, dest)
                if export:
                    rows = []
                    for archive in sorted(src.glob("trial_*.npz")):
                        rows.extend(per_trial_rows(archive))
                    table = dest / "per_trial.csv"
                    with table.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    copied.append(table)
                    npz = sorted(src.glob("*.npz"))
                    (dest / "ARCHIVES_NOT_COPIED.txt").write_text(
                        "Raw archives left in results/ (regenerate per_trial.csv with "
                        "scripts/collect_paper_data.py):\n" + "\n".join(
                            f"{p.name}\t{p.stat().st_size} bytes" for p in npz) + "\n")
                    copied.append(dest / "ARCHIVES_NOT_COPIED.txt")
            size = sum(p.stat().st_size for p in copied)
            total += size
            manifest.append(f"| `{source}` | {len(copied)} | {size / 1e6:.1f} |")
        manifest.append("")
    docs_dest = OUT / "docs"
    docs_dest.mkdir()
    size = 0
    for name in DOCS:
        shutil.copy2(ROOT / "docs" / name, docs_dest / name)
        size += (docs_dest / name).stat().st_size
    total += size
    manifest += ["## `docs/`", "", f"Protocol and plan documents ({len(DOCS)} files, {size / 1e6:.1f} MB): "
                 + ", ".join(f"`{n}`" for n in DOCS), ""]
    manifest += [f"**Total copied: {total / 1e6:.1f} MB.**", "", "## Left in `results/` (not tracked)", "",
                 "| Pattern | Why |", "|---|---|"]
    manifest += [f"| `{pattern}` | {why} |" for pattern, why in NOT_COLLECTED]
    manifest += ["", "Regenerate the section figures from the tracked tables with "
                 "`python scripts/make_rl_section_figures.py --from-csv`.", ""]
    (OUT / "MANIFEST.md").write_text("\n".join(manifest))
    print(f"collected {total / 1e6:.1f} MB into {OUT}")


if __name__ == "__main__":
    main()
