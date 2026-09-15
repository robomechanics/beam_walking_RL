"""Duty-factor fidelity gate for freshly trained specialists.

Reads the command-fidelity archives directly and measures, per commanded duty
factor, the realized stance fraction (foot contact over valid samples after a
2 s settle) and where in the commanded phase the extra or missing contact sits.
Passes when every duty anchor is realized within the tolerance, which is the
condition the duty factor by stance width comparison depends on.
"""
import argparse
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np


def measure(directory):
    by_duty, profile = {}, {}
    for path in sorted(glob.glob(f"{directory}/trial_*.npz")):
        match = re.search(r"_s([\d.]+)_d([\d.]+)_", Path(path).name)
        duty = round(float(match.group(2)), 3)
        with np.load(path, allow_pickle=False) as z:
            contacts = z["contacts"].astype(bool)
            keep = z["valid"].astype(bool).copy()
            keep[:100] = False
            mask = np.repeat(keep[..., None], 4, axis=-1)
            phase = z["phase"][mask]
            touched = contacts[mask]
        by_duty.setdefault(duty, []).append(float(touched.mean()))
        bins = np.minimum((phase * 10).astype(int), 9)
        profile.setdefault(duty, []).append(
            [float(touched[bins == b].mean()) if (bins == b).any() else float("nan")
             for b in range(10)])
    return ({d: float(np.mean(v)) for d, v in by_duty.items()},
            {d: list(np.nanmean(np.asarray(v), axis=0)) for d, v in profile.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="gait=directory")
    parser.add_argument("--tolerance", type=float, default=.05)
    parser.add_argument("--criterion", choices=("tracking", "contrast"), default="tracking",
                        help="tracking: every anchor within the tolerance. contrast: realized "
                             "duty rises with command and keeps at least --min-span of the "
                             "commanded span, for policies expected to shift duty somewhat")
    parser.add_argument("--min-span", type=float, default=.60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, passed = {}, True
    for item in args.runs:
        gait, directory = item.split("=", 1)
        realized, profile = measure(directory)
        errors = {d: realized[d] - d for d in realized}
        ordered = [realized[d] for d in sorted(realized)]
        span = (ordered[-1] - ordered[0]) / (max(realized) - min(realized)) if len(ordered) > 1 else 0.
        if args.criterion == "tracking":
            ok = bool(errors) and all(abs(e) <= args.tolerance for e in errors.values())
        else:
            ok = (len(ordered) > 1 and all(b > a for a, b in zip(ordered, ordered[1:]))
                  and span >= args.min_span)
        passed &= ok
        report[gait] = dict(directory=directory, realized_stance_fraction=realized,
                            error=errors, realized_over_commanded_span=span, passed=ok,
                            contact_by_phase_decile=profile)
        print(f"{gait}: " + "  ".join(f"{d:g}->{realized[d]:.3f}" for d in sorted(realized))
              + f"   span {span:.2f}   {'PASS' if ok else 'FAIL'}", flush=True)
    report["tolerance"] = args.tolerance
    report["criterion"] = args.criterion
    report["min_span"] = args.min_span
    report["passed"] = bool(passed)
    args.output.write_text(json.dumps(report, indent=2))
    sys.exit(0 if passed else 3)


if __name__ == "__main__":
    main()
