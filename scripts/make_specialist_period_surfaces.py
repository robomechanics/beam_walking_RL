"""Per-period speed x duty-factor surfaces from the specialist period sweeps.

The standalone surface grids fix the period at 0.48 s, matching the original
paper figure. The period sweeps cover the whole trained range, so this script
renders one surface panel per period straight from the sweep archives. It is a
plotting step only: no simulation, and it re-derives every plotted endpoint
from the hashed .npz archives rather than trusting the trial table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from unified_surface_data import (  # noqa: E402
    DUTIES_BY_GAIT, SPEEDS, SWEEP_SEED, TRIALS, WIDTHS, feasible_duties,
    sha256, summarize, validate_measurement,
)

GAIT_COLORS = {"trot": "#0072BD", "walk": "#D95319"}
KEYS = ("gait", "speed", "period", "step_width", "command_df")


def load_sweep(directory):
    """Validate a completed period-sweep grid and recompute its endpoints."""
    directory = Path(directory)
    manifest_path = directory / "surface_manifest.json"
    complete_path = directory / "SURFACE_COMPLETE"
    manifest = json.loads(manifest_path.read_text())
    complete = json.loads(complete_path.read_text())
    if (manifest.get("period_sweep") is not True or manifest.get("smoke") is not False
            or complete.get("manifest_sha256") != sha256(manifest_path)
            or complete.get("trials_sha256") != sha256(directory / "surface_trials.csv")):
        raise ValueError(f"{directory} is not a complete period sweep")
    checkpoint = manifest["checkpoint_sha256"]
    rows = []
    for entry in manifest["conditions"]:
        archive = directory / entry["filename"]
        if sha256(archive) != complete["archive_sha256"][entry["filename"]]:
            raise ValueError(f"Archive hash mismatch: {entry['filename']}")
        condition = tuple(entry[key] for key in KEYS)
        with np.load(archive, allow_pickle=False) as payload:
            if str(payload["checkpoint_sha256"]) != checkpoint:
                raise ValueError(f"Archive checkpoint mismatch: {entry['filename']}")
            validate_measurement(payload, condition)
            rows.extend(summarize(payload, condition, SWEEP_SEED))
    table = pd.read_csv(directory / "surface_trials.csv")
    verified = pd.DataFrame(rows)
    order = list(KEYS) + ["seed"]
    pd.testing.assert_frame_equal(
        table.sort_values(order).reset_index(drop=True),
        verified.sort_values(order).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    return manifest, verified


def summarize_cells(trials):
    rows = []
    for key, group in trials.groupby(list(KEYS)):
        valid = group.energy_trial_valid.astype(bool)
        finite = np.isfinite(group.positive_mechanical_cot) & ~group.any_failure.astype(bool)
        rate = float(valid.mean())
        rows.append(dict(zip(KEYS, key), trials=len(group),
                         energy_valid_rate=rate, energy_condition_valid=int(rate >= .90),
                         positive_mechanical_cot_median=(
                             float(group.loc[valid, "positive_mechanical_cot"].median())
                             if rate >= .90 else np.nan),
                         positive_mechanical_cot_median_all=(
                             float(group.loc[finite, "positive_mechanical_cot"].median())
                             if finite.any() else np.nan),
                         periodic_rate=float(group.periodic_orbit_gate_pass.mean()),
                         compliance_rate=float(group.compliant.mean())))
    return pd.DataFrame(rows)


def summarize_factors(cells):
    """Require every width, then weight all five widths equally."""
    rows = []
    for (gait, period, speed, duty), group in cells.groupby(
            ["gait", "period", "speed", "command_df"]):
        complete = len(group) == len(WIDTHS) and set(group.step_width) == set(WIDTHS)
        valid = complete and bool(group.energy_condition_valid.all())
        rows.append(dict(
            gait=gait, period=period, speed=speed, command_df=duty,
            valid_widths=int(group.energy_condition_valid.sum()), widths=len(WIDTHS),
            positive_mechanical_cot_median=(
                float(group.positive_mechanical_cot_median.median()) if valid else np.nan),
            positive_mechanical_cot_median_all=(
                float(group.positive_mechanical_cot_median_all.median())
                if complete and group.positive_mechanical_cot_median_all.notna().all()
                else np.nan),
            periodic_rate=float(group.periodic_rate.mean()) if complete else np.nan,
            compliance_rate=float(group.compliance_rate.mean()) if complete else np.nan))
    return pd.DataFrame(rows)


def mesh(axis, data, metric, color, duties, alpha=.72, markers=True):
    grid = data.pivot(index="speed", columns="command_df", values=metric).reindex(
        index=SPEEDS, columns=list(duties))
    z = grid.to_numpy()
    x, y = np.meshgrid(list(duties), SPEEDS)
    faces = []
    for i in range(len(SPEEDS) - 1):
        for j in range(len(duties) - 1):
            corners = [(x[a, b], y[a, b], z[a, b])
                       for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
            if np.isfinite(corners).all():
                faces.append(corners)
    if faces:
        axis.add_collection3d(Poly3DCollection(
            faces, facecolors=color, edgecolors=color, linewidths=.7, alpha=alpha))
    finite = np.isfinite(z)
    if not markers:
        return int((~finite).sum())
    axis.scatter(x[finite], y[finite], z[finite], c=color, s=20, depthshade=False)
    for i in range(len(SPEEDS)):
        for j in range(len(duties) - 1):
            if finite[i, j:j + 2].all():
                axis.plot(x[i, j:j + 2], y[i, j:j + 2], z[i, j:j + 2], color=color)
    for j in range(len(duties)):
        for i in range(len(SPEEDS) - 1):
            if finite[i:i + 2, j].all():
                axis.plot(x[i:i + 2, j], y[i:i + 2, j], z[i:i + 2, j], color=color)
    return int((~finite).sum())


def plot(factors, output, stem, metric, label, zlim):
    periods = sorted(factors.period.unique())
    all_duties = sorted({d for g in DUTIES_BY_GAIT for d in DUTIES_BY_GAIT[g]})
    fig = plt.figure(figsize=(4.6 * len(periods) + 1, 5.4))
    for index, period in enumerate(periods):
        axis = fig.add_subplot(1, len(periods), index + 1, projection="3d")
        selected = factors[np.isclose(factors.period, period)]
        for gait, color in GAIT_COLORS.items():
            group = selected[selected.gait == gait]
            if group.empty:
                continue
            duties = feasible_duties(gait, period)
            if metric == "positive_mechanical_cot_median":
                mesh(axis, group, "positive_mechanical_cot_median_all", color,
                     duties, alpha=.28, markers=False)
            mesh(axis, group, metric, color, duties)
        axis.set(xlabel="Commanded duty factor", ylabel="Speed (m/s)", zlabel=label,
                 xlim=(min(all_duties) - .01, max(all_duties) + .01),
                 ylim=(.24, .41), zlim=zlim)
        axis.set_xticks(all_duties)
        axis.set_yticks(SPEEDS)
        axis.view_init(elev=20, azim=-72)
        axis.set_title(f"Period {period:.2f} s", fontsize=12)
    fig.suptitle(f"Gait specialists: {label.lower()} across the trained period range",
                 fontsize=15, y=.99)
    fig.legend(handles=[Patch(facecolor=GAIT_COLORS[g], edgecolor=GAIT_COLORS[g],
                              label=g.title(), alpha=.72) for g in GAIT_COLORS],
               loc="upper center", bbox_to_anchor=(.5, .93), ncol=2, frameon=True,
               fancybox=False, framealpha=1., edgecolor="#777777", fontsize=12)
    fig.subplots_adjust(left=.02, right=.97, bottom=.05, top=.84, wspace=.10)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Period-sweep grid directories, one per gait")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "PAPER_GRAPHS/specialist_period_surfaces")
    args = parser.parse_args()
    manifests, frames = [], []
    for directory in args.input:
        manifest, trials = load_sweep(directory)
        manifests.append(manifest)
        frames.append(trials)
    gaits = [sorted({r["gait"] for r in m["conditions"]}) for m in manifests]
    if any(set(a) & set(b) for i, a in enumerate(gaits) for b in gaits[i + 1:]):
        raise ValueError("Each gait must come from exactly one sweep")
    trials = pd.concat(frames, ignore_index=True)
    cells = summarize_cells(trials)
    factors = summarize_factors(cells)
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "period_surface_cells.csv", index=False)
    factors.to_csv(args.output / "period_surface_factors.csv", index=False)
    top = factors.positive_mechanical_cot_median_all.dropna()
    plot(factors, args.output, "period_surfaces_cot",
         "positive_mechanical_cot_median", "Positive mechanical CoT",
         (0, max(.1, float(top.max()) * 1.12) if len(top) else 1.))
    plot(factors, args.output, "period_surfaces_periodicity",
         "periodic_rate", "Periodicity of realized motion (not chi)", (0, 1.05))
    (args.output / "provenance.json").write_text(json.dumps(dict(
        input=[str(d.resolve()) for d in args.input],
        checkpoint_sha256=[m["checkpoint_sha256"] for m in manifests],
        completion_sha256=[sha256(d / "SURFACE_COMPLETE") for d in args.input],
        trials_per_condition=TRIALS,
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        paper_claims_allowed=False), indent=2))
    (args.output / "README.md").write_text(
        "# Per-period speed x duty-factor surfaces (gait specialists)\n\n"
        "Rendered from the period-sweep grids, one panel per commanded period. "
        "Solid markers are cells passing the frozen compliance and finite-energy "
        "gates; the faint plane behind them is the median over every non-failed "
        "trial with finite cost of transport, gates ignored. Duty anchors shown "
        "per gait are those feasible at that period: trot needs five swing "
        "control ticks, walk two.\n")
    print(json.dumps(dict(output=str(args.output), cells=len(cells),
                          factors=len(factors),
                          periods=sorted(factors.period.unique()))))


if __name__ == "__main__":
    main()
