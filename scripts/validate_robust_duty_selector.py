"""Validate and promote the robust duty-factor selector on fresh seeds.

For every supported context, the selector's duty factor is run again under the
same disturbances on the held-out test split
(results/narrow_robust_selector_validation_<gait>_v<speed>_p<period>_<tag>, one
run per gait, speed and period over the context widths and the selected duty
factors). A context fails when its fresh success rate is significantly below 90%
of the best success rate in its labelling runs: one-sided exact binomial test,
family-wise 5% with a Bonferroni correction over contexts. When no context
fails, the selector is saved next to the evidence as
validated_robust_duty_selector.pt with deployment_ready=True.

--print-jobs lists the validation runs for the pipeline instead.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment import unified_duty_selector as selector  # noqa: E402
from beam_walking.experiment.duty_selector import selector_state_sha256  # noqa: E402

SCHEMA = "robust_duty_selector_v1"
FAMILY_ALPHA = .05
VALIDATION_SEED = 1000000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("Not a robust duty-factor selector")
    selector.set_width_range(payload["input_ranges"]["step_width"])
    selector.set_gait_levels(payload["gait_levels"])
    if payload["selector_runtime_sha256"] != selector.selector_runtime_sha256(ROOT):
        raise ValueError("Selector runtime source differs from the fit")
    model = selector.UnifiedDutySelector(tuple(payload["hidden_dims"]))
    model.load_state_dict(payload["state_dict"])
    if selector_state_sha256(model.state_dict()) != payload["selector_state_sha256"]:
        raise ValueError("Selector parameter hash mismatch")
    model.eval()
    contexts = payload["supported_contexts"]
    predictions = selector.predict_rows(model, [dict(row) for row in contexts])
    if any(abs(row["selected_df"] - row["target_df"]) > 1e-6 for row in predictions):
        raise ValueError("Selector no longer reproduces its labels")
    return payload, predictions


def groups(predictions):
    """{(gait, speed, period): [prediction rows]}"""
    out = defaultdict(list)
    for row in predictions:
        out[(row["gait"], row["speed"], row["period"])].append(row)
    return dict(sorted(out.items()))


def run_directory(results, tag, gait, speed, period):
    return results / f"narrow_robust_selector_validation_{gait}_v{round(speed * 100):03d}_p{period:.2f}_{tag}"


def binomial_cdf(x, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-jobs", action="store_true")
    args = parser.parse_args()
    payload, predictions = load(args.selector)
    by_run = groups(predictions)
    if args.print_jobs:
        for (gait, speed, period), rows in by_run.items():
            widths = ",".join(f"{w:g}" for w in sorted({row["step_width"] for row in rows}))
            duties = ",".join(f"{d:g}" for d in sorted({row["selected_df"] for row in rows}))
            print(f"selector_validation_job {gait} {speed:g} {period:g} {widths} {duties}")
        return
    if args.output is None:
        parser.error("--output is required")
    if payload.get("exploratory") is not False or payload.get("deployment_ready") is not False:
        parser.error("Only a primary, not-yet-validated selector can be promoted")
    alpha = FAMILY_ALPHA / len(predictions)
    summaries, evidence = [], []
    for (gait, speed, period), rows in by_run.items():
        directory = run_directory(args.results, args.tag, gait, speed, period)
        paths = {name: directory / name for name in
                 ("evaluation_complete.json", "evaluation_manifest.json", "perturbation_summary.json")}
        if not all(path.is_file() for path in paths.values()):
            raise ValueError(f"Validation run is missing: {directory}")
        completion = json.loads(paths["evaluation_complete.json"].read_text())
        manifest = json.loads(paths["evaluation_manifest.json"].read_text())
        summary = json.loads(paths["perturbation_summary.json"].read_text())
        counts = {(round(c["step_width"], 2), round(c["df"], 3)): (int(c["success"]), int(c["trials"]))
                  for c in summary["conditions"].values()}
        widths = {row["step_width"] for row in rows}
        duties = {row["selected_df"] for row in rows}
        if (completion.get("complete") is not True or completion.get("external_pushes") is not True
                or manifest.get("split") != "test" or manifest.get("seeds", [None])[0] != VALIDATION_SEED
                or manifest.get("gait") != gait or abs(manifest.get("speed") - speed) > 1e-6
                or abs(manifest.get("period") - period) > 1e-6
                or float(summary["perturbation"]["max_force_n"]) != payload["disturbance_max_force_n"]
                or completion.get("checkpoint_sha256") != payload["checkpoint_sha256"][gait]
                or set(counts) != {(round(w, 2), round(d, 3)) for w in widths for d in duties}):
            raise ValueError(f"Validation run does not match this selector: {directory}")
        evidence.append({"gait": gait, "speed": speed, "period": period, "directory": directory.name,
                         "sha256": {name: sha256(path) for name, path in paths.items()}})
        for row in rows:
            s, n = counts[(round(row["step_width"], 2), round(row["selected_df"], 3))]
            bar = payload["retained_success"][0] / payload["retained_success"][1] * (
                row["best_success"] / row["best_trials"])
            p_value = binomial_cdf(s, n, bar)
            summaries.append({
                "gait": gait, "speed": speed, "period": period, "step_width": row["step_width"],
                "selected_df": row["selected_df"], "fresh_success": s, "fresh_trials": n, "fresh_success_rate": s / n,
                "labelled_success_rate": row["target_success"] / row["target_trials"],
                "best_labelled_success_rate": row["best_success"] / row["best_trials"],
                "required_success_rate": bar, "p_value": p_value, "passed": int(p_value >= alpha)})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "selector_validation_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    failed = [row for row in summaries if not row["passed"]]
    report = {
        "schema": "robust_duty_selector_validation_v1",
        "contexts": len(summaries), "passed_contexts": len(summaries) - len(failed),
        "all_contexts_passed": not failed, "per_context_alpha": alpha, "family_alpha": FAMILY_ALPHA,
        "failed_contexts": [{key: row[key] for key in ("gait", "speed", "period", "step_width", "selected_df",
                                                      "fresh_success", "fresh_trials", "required_success_rate")}
                            for row in failed],
        "selector_checkpoint_sha256": sha256(args.selector),
    }
    if not failed:
        validated = dict(payload, deployment_ready=True, validation_evidence=evidence,
                         validation_selector_checkpoint_sha256=sha256(args.selector),
                         validation_split="test", validation_seed_start=VALIDATION_SEED)
        torch.save(validated, args.output / "validated_robust_duty_selector.pt")
        report["validated_selector"] = "validated_robust_duty_selector.pt"
    (args.output / "validation_report.json").write_text(json.dumps(report, indent=2))
    print("ROBUST_DUTY_SELECTOR_VALIDATION", json.dumps(report), flush=True)
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
