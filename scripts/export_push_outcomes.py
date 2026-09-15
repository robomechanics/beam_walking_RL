"""Export one tidy table of push-study outcomes from the evaluation archives.

The nominal arms get per-trial tables from `analyze_beam.py`, but that analyzer
refuses perturbed runs by design, so the pushed arms live only in their hashed
`.npz` archives. This script reads every archive directly and writes two CSVs:
per-trial outcomes and per-condition rates with Wilson intervals, covering both
the nominal and the pushed arms on the same footing.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"push_(?P<gait>trot|walk)_(?P<policy>base|tuned)_"
                  r"(?P<condition>nominal|perturbed)_p(?P<period>[\d.]+)_v(?P<speed>[\d.]+)_")


def wilson(successes, trials, z=1.96):
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    d = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / d
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / d
    return max(0., centre - half), min(1., centre + half)


def trial_rows(directory):
    directory = Path(directory)
    match = NAME.search(directory.name + "_")
    if match is None:
        return []
    meta = match.groupdict()
    manifest = json.loads((directory / "evaluation_manifest.json").read_text())
    out = []
    for entry in manifest["expected_conditions"]:
        archive = directory / entry["filename"]
        if not archive.is_file():
            continue
        with np.load(archive, allow_pickle=False) as payload:
            valid = payload["valid"].astype(bool)
            contacts = payload["contacts"]
            feet_body = payload["feet_body"]
            forward = payload["forward_velocity"]
            lateral = payload["body"][:, :, 1]
            force_w = (payload["perturbation_force_w"]
                       if "perturbation_force_w" in payload else None)
            control_dt = float(payload["control_dt"])
            failure = payload["failure"].astype(bool)
            success = payload["success"].astype(bool)
            body = payload["body"]
            seeds = payload["seeds"]
            envelope = (payload["perturbation_envelope"]
                        if "perturbation_envelope" in payload else None)
            achieved_df = (payload["achieved_df"] if "achieved_df" in payload else None)
            trials = valid.shape[1]
            last = np.array([np.flatnonzero(valid[:, i])[-1] for i in range(trials)])
            index = (last, np.arange(trials))
            failed = failure[index]
            finished = success[index] & ~failed
            travel = body[index][:, 0] + .65
            # Descriptive per-trial metrics over each trial's own valid window.
            # These are computed directly from the archive, not by the audited
            # analyzer, so treat them as descriptive rather than gate outcomes.
            for trial in range(trials):
                window = valid[:, trial]
                stance = contacts[window, trial]
                separation = np.abs(feet_body[window, trial, 0, 1]
                                    - feet_body[window, trial, 1, 1])
                rear = np.abs(feet_body[window, trial, 2, 1]
                              - feet_body[window, trial, 3, 1])
                row = dict(meta, step_width=entry["step_width"], command_df=entry["df"],
                           seed=int(seeds[trial]),
                           success=int(finished[trial]), failure=int(failed[trial]),
                           timeout=int(not finished[trial] and not failed[trial]),
                           travel_m=float(travel[trial]),
                           end_time_s=float(last[trial] * .02))
                if envelope is not None:
                    row["push_envelope_at_end"] = float(envelope[index][trial])
                if achieved_df is not None:
                    row["achieved_df"] = float(np.asarray(achieved_df)[trial])
                row["stance_fraction"] = float(stance.mean())
                row["achieved_width_m"] = float(np.mean(
                    np.concatenate((separation, rear))))
                row["mean_forward_speed"] = float(forward[window, trial].mean())
                row["lateral_rmse_m"] = float(np.sqrt(
                    np.mean(lateral[window, trial] ** 2)))
                if force_w is not None:
                    magnitude = np.linalg.norm(force_w[window, trial], axis=-1)
                    row["peak_push_force_n"] = float(magnitude.max())
                    row["push_impulse_ns"] = float(magnitude.sum() * control_dt)
                out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", default="20260914")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    sources = []
    for directory in sorted(args.results.glob(f"push_*_{args.tag}")):
        if not (directory / "evaluation_complete.json").is_file():
            continue
        found = trial_rows(directory)
        if found:
            rows.extend(found)
            sources.append(directory.name)
    if not rows:
        raise SystemExit("No completed push arms found")
    args.output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (args.output / "push_trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    keys = ("gait", "policy", "condition", "period", "speed", "step_width", "command_df")
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        n = len(group)
        s = sum(r["success"] for r in group)
        low, high = wilson(s, n)
        summary.append(dict(zip(keys, key), trials=n, success=s,
                            failure=sum(r["failure"] for r in group),
                            timeout=sum(r["timeout"] for r in group),
                            success_rate=s / n, ci_low=low, ci_high=high,
                            mean_travel_m=float(np.mean([r["travel_m"] for r in group])),
                            mean_end_time_s=float(np.mean([r["end_time_s"] for r in group]))))
    with (args.output / "push_conditions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        sources=sources, trials=len(rows), conditions=len(summary),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print(json.dumps(dict(output=str(args.output), arms=len(sources),
                          trials=len(rows), conditions=len(summary))))


if __name__ == "__main__":
    main()
