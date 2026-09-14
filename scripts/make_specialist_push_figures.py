"""Figures for the specialist push study.

Reads the push-study evaluation archives and compares the base specialist
against its push fine-tune, nominal and perturbed, over the factors the
specialists were trained on: step width, duty factor, period, and speed.
Cells are identified from the archive metadata, not the file name, and every
archive is checked against the checkpoint it is supposed to have come from.
See docs/unified_gait_plan.md and TASK_MEMORY.md.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("base_nominal", "base_perturbed", "tuned_nominal", "tuned_perturbed")
LABELS = {"base_nominal": "specialist, nominal",
          "base_perturbed": "specialist, pushed",
          "tuned_nominal": "push-trained, nominal",
          "tuned_perturbed": "push-trained, pushed"}
STYLES = {"base_nominal": ("#2f6db3", "--"), "base_perturbed": ("#2f6db3", "-"),
          "tuned_nominal": ("#d8792a", "--"), "tuned_perturbed": ("#d8792a", "-")}
DT = .02
# Ramped push envelope used in training and evaluation: start_fraction .10
# growing to 1 over ramp_time 10 s, bound 25 N (docs/perturbation.md).
PUSH_BOUND_N = 25.
RAMP_S = 10.
START_FRACTION = .10


def trial_outcome(path):
    """Per-trial success/failure at each trial's own last valid step."""
    with np.load(path) as d:
        valid = d["valid"].astype(bool)
        failure = d["failure"].astype(bool)
        success = d["success"].astype(bool)
        meta = {"step_width": float(d["step_width"]), "df": float(d["df"]),
                "speed": float(d["command_speed"]), "period": float(d["period"]),
                "gait": str(d["gait"]), "disturbed": bool(d["disturbed"]),
                "checkpoint_sha256": str(d["checkpoint_sha256"])}
        envelope = d["perturbation_envelope"] if "perturbation_envelope" in d else None
        trials = valid.shape[1]
        alive = [np.flatnonzero(valid[:, i]) for i in range(trials)]
        if any(len(a) == 0 for a in alive):
            raise ValueError(f"{path} has a trial with no valid step")
        last = np.array([a[-1] for a in alive])
        idx = (last, np.arange(trials))
        failed = failure[idx]
        meta.update({
            "trials": trials, "last_step": last, "failed": failed,
            "succeeded": success[idx] & ~failed,
            "envelope_at_end": None if envelope is None else envelope[idx],
        })
    return meta


def load_arm(directory):
    """Every trial archive in one evaluation directory, keyed by cell."""
    directory = Path(directory)
    if not (directory / "evaluation_complete.json").exists():
        return {}
    cells = {}
    for path in sorted(directory.glob("trial_*.npz")):
        cell = trial_outcome(path)
        key = (cell["period"], cell["speed"], cell["step_width"], cell["df"])
        cells[key] = cell
    return cells


def collect(results, gait, tag, expected_sha):
    """Load all four arms across every period/speed run of one gait."""
    data = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for directory in sorted(results.glob(f"push_{gait}_{arm}_p*_v*_{tag}")):
            cells = load_arm(directory)
            for key, cell in cells.items():
                want = expected_sha["tuned" if arm.startswith("tuned") else "base"]
                if want and cell["checkpoint_sha256"] != want:
                    raise ValueError(
                        f"{directory} was produced by {cell['checkpoint_sha256'][:16]}, "
                        f"expected {want[:16]}")
                if cell["disturbed"] != arm.endswith("perturbed"):
                    raise ValueError(f"{directory} disturbed flag disagrees with arm {arm}")
                data[arm][key] = cell
    return data


def rate(cells):
    """Pooled success percentage over a set of cells."""
    total = sum(c["trials"] for c in cells)
    if not total:
        return np.nan
    return 100 * sum(int(c["succeeded"].sum()) for c in cells) / total


def line_over(ax, data, axis_index, title_keep, xlabel):
    """One line per arm: success rate against the factor at axis_index."""
    xs = sorted({k[axis_index] for arm in ARMS for k in data[arm]
                 if all(k[i] == v for i, v in title_keep.items())})
    drawn = False
    for arm in ARMS:
        ys = []
        for x in xs:
            cells = [c for k, c in data[arm].items()
                     if k[axis_index] == x
                     and all(k[i] == v for i, v in title_keep.items())]
            ys.append(rate(cells))
        if not all(np.isnan(y) for y in ys):
            color, ls = STYLES[arm]
            ax.plot(xs, ys, ls, color=color, marker="o", label=LABELS[arm])
            drawn = True
    ax.set_xlabel(xlabel)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=.3)
    return drawn


def survival_figure(data, gait, out):
    """Pooled survival against episode time under push, with the push bound."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax2 = ax.twinx()
    tmax = 0.
    for arm in ("base_perturbed", "tuned_perturbed"):
        cells = list(data[arm].values())
        if not cells:
            continue
        ends = np.concatenate([c["last_step"] for c in cells])
        fails = np.concatenate([c["failed"] for c in cells])
        steps = np.arange(0, int(ends.max()) + 2)
        t = steps * DT
        tmax = max(tmax, float(t[-1]))
        alive = np.array([1 - ((ends < k) & fails).mean() for k in steps])
        color, _ = STYLES[arm]
        ax.plot(t, 100 * alive, color=color, lw=2, label=LABELS[arm])
        ax.text(t[-1], 100 * alive[-1] + 1.5, f"{100 * alive[-1]:.0f}%",
                color=color, ha="right", fontsize=9)
    if tmax:
        t = np.arange(int(tmax / DT) + 1) * DT
        env = START_FRACTION + (1 - START_FRACTION) * np.clip(t / RAMP_S, 0, 1)
        ax2.fill_between(t, 0, PUSH_BOUND_N * env, color="#999999", alpha=.15, lw=0)
        ax2.plot(t, PUSH_BOUND_N * env, color="#777777", lw=1, ls="--", label="push bound")
        ax.set_xlim(0, tmax)
    ax.set_xlabel("episode time (s)")
    ax.set_ylabel("trials not fallen (%)")
    ax.set_ylim(0, 105)
    ax2.set_ylabel("push force bound (N)")
    ax2.set_ylim(0, PUSH_BOUND_N * 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True)
    ax.grid(alpha=.3)
    fig.suptitle(f"{gait}: survival under ramped push")
    save(fig, out, f"{gait}_survival_vs_time")


def width_figure(data, gait, out, period, speed):
    """Success against step width, one panel per duty factor, at one cell."""
    keep = {0: period, 1: speed}
    dfs = sorted({k[3] for arm in ARMS for k in data[arm]
                  if k[0] == period and k[1] == speed})
    if not dfs:
        return
    fig, axes = plt.subplots(1, len(dfs), figsize=(3.2 * len(dfs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    drawn = False
    for ax, df in zip(axes, dfs):
        drawn |= line_over(ax, data, 2, {**keep, 3: df}, "step width (m)")
        ax.set_title(f"DF {df:.3f}")
    axes[0].set_ylabel("success rate (%)")
    if drawn:
        axes[-1].legend(frameon=True, fontsize=8)
    fig.suptitle(f"{gait}: success by step width, period {period:.2f} s, {speed:.2f} m/s")
    save(fig, out, f"{gait}_success_by_width_per_df")


def factor_figure(data, gait, out, axis_index, fixed, xlabel, name):
    """Success against period or speed, pooled over widths and duty factors."""
    fig, ax = plt.subplots(figsize=(6.4, 4))
    drawn = line_over(ax, data, axis_index, fixed, xlabel)
    ax.set_ylabel("success rate (%)")
    if drawn:
        ax.legend(frameon=True)
    fixed_text = ", ".join(
        f"{'period' if i == 0 else 'speed'} {v:.2f}" for i, v in fixed.items())
    fig.suptitle(f"{gait}: success vs {xlabel} ({fixed_text})")
    save(fig, out, f"{gait}_{name}")


def save(fig, out, stem):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=160)
    plt.close(fig)


def tables(data, gait, out):
    """Per-cell outcomes and the arm totals the figures are drawn from."""
    rows = {}
    for arm in ARMS:
        for (period, speed, width, df), cell in sorted(data[arm].items()):
            failed = cell["failed"]
            key = f"{arm} | p{period:.2f} | v{speed:.2f} | w{width:.2f} | d{df:.3f}"
            row = {"trials": cell["trials"],
                   "success": int(cell["succeeded"].sum()),
                   "failure": int(failed.sum()),
                   "timeout": int(cell["trials"] - failed.sum() - cell["succeeded"].sum()),
                   "mean_time_at_failure_s":
                       float((cell["last_step"][failed] * DT).mean()) if failed.any() else None}
            if cell["envelope_at_end"] is not None and failed.any():
                row["mean_envelope_at_failure"] = float(cell["envelope_at_end"][failed].mean())
            rows[key] = row
    (out / f"{gait}_push_outcomes.json").write_text(json.dumps(rows, indent=1) + "\n")

    totals = defaultdict(lambda: {"success": 0, "trials": 0})
    for key, row in rows.items():
        arm = key.split(" | ")[0]
        totals[arm]["success"] += row["success"]
        totals[arm]["trials"] += row["trials"]
    lines = [f"# {gait} push study", "",
             "| arm | success rate |", "|---|---|"]
    for arm in ARMS:
        total = totals[arm]
        if total["trials"]:
            pct = 100 * total["success"] / total["trials"]
            lines.append(f"| {LABELS[arm]} | {total['success']}/{total['trials']} = {pct:.0f}% |")
    lines += ["", "| arm | period | speed | width | DF | success | failure | timeout | fail time s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, row in rows.items():
        arm, period, speed, width, df = key.split(" | ")
        fail_time = "" if row["mean_time_at_failure_s"] is None \
            else f"{row['mean_time_at_failure_s']:.1f}"
        lines.append(f"| {LABELS[arm]} | {period[1:]} | {speed[1:]} | {width[1:]} | {df[1:]} | "
                     f"{row['success']} | {row['failure']} | {row['timeout']} | {fail_time} |")
    (out / f"{gait}_PUSH_RESULTS.md").write_text("\n".join(lines) + "\n")
    return {arm: dict(total) for arm, total in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--tag", default="20260914",
                        help="Suffix of the push-study evaluation directories")
    parser.add_argument("--gaits", nargs="+", default=["trot", "walk"])
    parser.add_argument("--core-period", type=float, default=.48)
    parser.add_argument("--core-speed", type=float, default=.30)
    parser.add_argument("--output", type=Path, default=ROOT / "PAPER_GRAPHS/specialist_push")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for gait in args.gaits:
        base = args.results / f"unified_specialist_{gait}_seed5_3072_20260914/model_1799.pt"
        tuned = args.results / f"unified_specialist_push_{gait}_from_seed5_3072_20260914/model_1799.pt"
        expected = {}
        for name, path in (("base", base), ("tuned", tuned)):
            from hashlib import sha256
            expected[name] = sha256(path.read_bytes()).hexdigest() if path.exists() else ""
        data = collect(args.results, gait, args.tag, expected)
        if not any(data[arm] for arm in ARMS):
            print(f"{gait}: no push-study archives found, skipped")
            continue
        survival_figure(data, gait, args.output)
        width_figure(data, gait, args.output, args.core_period, args.core_speed)
        factor_figure(data, gait, args.output, 0, {1: args.core_speed},
                      "period (s)", "success_vs_period")
        factor_figure(data, gait, args.output, 1, {0: args.core_period},
                      "commanded speed (m/s)", "success_vs_speed")
        summary[gait] = {"arms": tables(data, gait, args.output),
                         "base_checkpoint_sha256": expected["base"],
                         "tuned_checkpoint_sha256": expected["tuned"]}
        print(f"{gait}: figures and tables written to {args.output}")
    (args.output / "push_summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
