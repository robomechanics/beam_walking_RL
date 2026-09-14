"""Plot audited measurements of the original policy's walking DF generalization."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pandas as pd

# --task selects the frozen seed-2 definitions (default) or the unified seed-5
# definitions (unified_surface_data.py).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--task", choices=("paper", "unified", "specialist"), default="paper")
TASK_VARIANT = _pre.parse_known_args()[0].task
import importlib  # noqa: E402
_data = importlib.import_module({
    "paper": "policy_surface_data", "unified": "unified_surface_data",
    "specialist": "specialist_surface_data"}[TASK_VARIANT])
CHECKPOINT_SHA256, DUTIES, GAITS = _data.CHECKPOINT_SHA256, _data.DUTIES, _data.GAITS
GRID_SEED, PERIOD, ROOT = _data.GRID_SEED, _data.PERIOD, _data.ROOT
SOURCE_FILES, SPEEDS, TASK_FILES = _data.SOURCE_FILES, _data.SPEEDS, _data.TASK_FILES
TRIALS, WIDTHS, conditions = _data.TRIALS, _data.WIDTHS, _data.conditions
sha256, summarize, validate_measurement = _data.sha256, _data.summarize, _data.validate_measurement
TRAINING_SEED = getattr(_data, "TRAINING_SEED", 2)
SCHEMA = getattr(_data, "SCHEMA", "old_policy_surface_exploratory_v1")
COMPLETE_SCHEMA = getattr(_data, "COMPLETE_SCHEMA", "old_policy_surface_complete_v1")
DUTIES_BY_GAIT = getattr(_data, "DUTIES_BY_GAIT", {"trot": DUTIES, "walk": (.75,)})
TEXT = {
    "paper": dict(
        output="PAPER_GRAPHS/old_policy_walk_df",
        title="Original RL policy: exploratory speed × duty-factor surfaces",
        title_width="Original RL policy: measured surfaces at each stance width",
        footer=True,
        readme_head="# Original-policy walking duty-factor extension\n\n"
        "These are new exploratory measurements of the exact seed-2 checkpoint used by the old figures. "
        "Walk DF 0.50 and 0.625 were not in its training distribution. No adaptive policy is used.\n\n"),
    "specialist": dict(
        output="PAPER_GRAPHS/specialist_surfaces",
        title="Gait specialists: speed × duty-factor surfaces",
        title_width="Gait specialists: measured surfaces at each stance width",
        footer=False,
        readme_head="# Gait-specialist speed × duty-factor surfaces\n\n"
        "Measurements of the trot and walk specialists trained under the unified "
        "protocol (docs/unified_gait_plan.md), each on its own checkpoint: trot at "
        "DF 0.50/0.625/0.75 and walk at DF 0.75/0.80/0.85/0.90, period 0.48 s.\n\n"),
    "unified": dict(
        output="PAPER_GRAPHS/unified_surfaces",
        title="Unified RL policy: speed × duty-factor surfaces",
        title_width="Unified RL policy: measured surfaces at each stance width",
        footer=False,
        readme_head="# Unified-policy speed × duty-factor surfaces\n\n"
        "Measurements of the fresh unified seed-5 checkpoint (docs/unified_gait_plan.md): "
        "trot at DF 0.50/0.625/0.75 and walk at DF 0.75/0.80/0.85/0.90, all within its "
        "training distribution, at period 0.48 s.\n\n"),
}[TASK_VARIANT]

GAIT_COLORS = {"trot": "#0072BD", "walk": "#D95319"}


def load_results(directory):
    manifest_path = directory / "surface_manifest.json"
    trials_path = directory / "surface_trials.csv"
    complete = json.loads((directory / "SURFACE_COMPLETE").read_text())
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != SCHEMA
            or manifest.get("smoke") is not False
            or manifest.get("paper_claims_allowed") is not False
            or manifest.get("chi_computed") is not False
            or complete.get("schema") != COMPLETE_SCHEMA
            or complete.get("manifest_sha256") != sha256(manifest_path)
            or complete.get("trials_sha256") != sha256(trials_path)):
        raise ValueError("A complete exploratory grid with matching hashes is required")
    if CHECKPOINT_SHA256 is not None and manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Grid must use the registered checkpoint for this task")
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    training_path = directory / "training_provenance.json"
    training = json.loads(training_path.read_text())
    archived = directory / "source_snapshot"
    task_hash = hashlib.sha256(b"".join((archived / p).read_bytes() for p in TASK_FILES)).hexdigest()
    collector_hash = hashlib.sha256(b"".join((archived / p).read_bytes() for p in SOURCE_FILES)).hexdigest()
    if (sha256(training_path) != manifest.get("training_provenance_sha256")
            or training.get("seed") != TRAINING_SEED or training.get("fresh_training") is not True
            or training.get("task_sha256") != task_hash
            or task_hash != manifest.get("task_sha256")
            or collector_hash != manifest.get("collector_sha256")):
        raise ValueError("Archived source/training identity mismatch")
    grid_gaits = set(manifest.get("gaits") or GAITS)
    expected = {c for c in conditions() if c[0] in grid_gaits}
    keys = ("gait", "speed", "period", "step_width", "command_df")
    if {tuple(e[k] for k in keys) for e in manifest["conditions"]} != expected:
        raise ValueError("Manifest does not contain the full fixed physical grid")
    names = {e["filename"] for e in manifest["conditions"]}
    if set(complete["archive_sha256"]) != names or {p.name for p in directory.glob("*.npz")} != names:
        raise ValueError("Archive set does not match the completed manifest")
    for name, digest in complete["archive_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Archive hash mismatch: {name}")
    trials = pd.read_csv(trials_path)
    grouped = trials.groupby(list(keys))
    if set(grouped.groups) != expected or len(trials) != len(expected) * TRIALS:
        raise ValueError("Trial table is not the complete fixed grid")
    for key, group in grouped:
        if len(group) != TRIALS or set(group.seed) != set(range(GRID_SEED, GRID_SEED + TRIALS)):
            raise ValueError(f"Missing or duplicated matched trials at {key}")
    # Recompute endpoints from hashed raw measurements; a CSV edit cannot create
    # a passing condition or change the plotted values.
    recomputed = []
    for entry in manifest["conditions"]:
        key = tuple(entry[k] for k in keys)
        with np.load(directory / entry["filename"], allow_pickle=False) as payload:
            expected_command = [key[1], key[4], key[3], key[2], GAITS.index(key[0])]
            if (str(payload["checkpoint_sha256"]) != checkpoint_sha256
                    or str(payload["task_sha256"]) != task_hash
                    or str(payload["collector_sha256"]) != collector_hash
                    or not np.array_equal(payload["seeds"], np.arange(GRID_SEED, GRID_SEED + TRIALS))
                    or not np.allclose(payload["command"], expected_command, atol=1e-7, rtol=0)):
                raise ValueError(f"Archive identity/condition mismatch: {entry['filename']}")
            validate_measurement(payload, key)
            recomputed.extend(summarize(payload, key, GRID_SEED))
    verified = pd.DataFrame(recomputed)
    sort_keys = list(keys) + ["seed"]
    pd.testing.assert_frame_equal(
        trials.sort_values(sort_keys).reset_index(drop=True),
        verified.sort_values(sort_keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
    return manifest, trials


def summarize_cells(trials):
    rows = []
    keys = ("gait", "speed", "period", "step_width", "command_df")
    for key, group in trials.groupby(list(keys)):
        valid = group.energy_trial_valid.astype(bool)
        rate = float(valid.mean())
        finite = np.isfinite(group.positive_mechanical_cot) & ~group.any_failure.astype(bool)
        rows.append(dict(zip(keys, key), trials=len(group),
                         valid_energy_trials=int(valid.sum()), energy_valid_rate=rate,
                         energy_condition_valid=int(rate >= .90),
                         positive_mechanical_cot_median=(
                             float(group.loc[valid, "positive_mechanical_cot"].median())
                             if rate >= .90 else np.nan),
                         # Descriptive: every non-failed trial with finite CoT,
                         # regardless of the command/topology/periodicity gates.
                         finite_cot_trials=int(finite.sum()),
                         positive_mechanical_cot_median_all=(
                             float(group.loc[finite, "positive_mechanical_cot"].median())
                             if finite.any() else np.nan),
                         periodic_rate=float(group.periodic_orbit_gate_pass.mean()),
                         compliance_rate=float(group.compliant.mean()),
                         achieved_df_mean=float(group.achieved_df.mean()),
                         out_of_training_support=int(
                             not min(DUTIES_BY_GAIT[key[0]]) - 1e-9 <= key[4]
                             <= max(DUTIES_BY_GAIT[key[0]]) + 1e-9)))
    return pd.DataFrame(rows)


def summarize_factors(cells):
    """Require every width, then weight all five widths equally."""
    rows = []
    for (gait, speed, duty), group in cells.groupby(["gait", "speed", "command_df"]):
        complete = len(group) == 5 and set(group.step_width) == set(WIDTHS)
        energy_valid = complete and bool(group.energy_condition_valid.all())
        rows.append(dict(gait=gait, speed=speed, command_df=duty,
                         valid_widths=int(group.energy_condition_valid.sum()), widths=5,
                         positive_mechanical_cot_median=(
                             float(group.positive_mechanical_cot_median.median())
                             if energy_valid else np.nan),
                         positive_mechanical_cot_median_all=(
                             float(group.positive_mechanical_cot_median_all.median())
                             if complete and group.positive_mechanical_cot_median_all.notna().all()
                             else np.nan),
                         periodic_rate=float(group.periodic_rate.mean()) if complete else np.nan,
                         compliance_rate=float(group.compliance_rate.mean()) if complete else np.nan))
    return pd.DataFrame(rows)


def measured_mesh(axis, data, metric, color, duties=DUTIES, alpha=.72, faces_only=False):
    grid = data.pivot(index="speed", columns="command_df", values=metric).reindex(
        index=SPEEDS, columns=list(duties))
    z = grid.to_numpy()
    x, y = np.meshgrid(duties, SPEEDS)
    faces = []
    for i in range(len(SPEEDS) - 1):
        for j in range(len(duties) - 1):
            vertices = [(x[a, b], y[a, b], z[a, b])
                        for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))]
            if np.isfinite(vertices).all():
                faces.append(vertices)
    if faces:
        axis.add_collection3d(Poly3DCollection(
            faces, facecolors=color, edgecolors=color, linewidths=.7, alpha=alpha))
    finite = np.isfinite(z)
    if faces_only:
        return int((~finite).sum())
    axis.scatter(x[finite], y[finite], z[finite], c=color, s=22, depthshade=False)
    # Only connect adjacent measured vertices. Never bridge an invalid vertex.
    for i in range(len(SPEEDS)):
        for j in range(len(duties) - 1):
            if finite[i, j:j + 2].all():
                axis.plot(x[i, j:j + 2], y[i, j:j + 2], z[i, j:j + 2], color=color)
    for j in range(len(duties)):
        for i in range(len(SPEEDS) - 1):
            if finite[i:i + 2, j].all():
                axis.plot(x[i:i + 2, j], y[i:i + 2, j], z[i:i + 2, j], color=color)
    return int((~finite).sum())


def plot_surfaces(data, output, stem, title, per_width=False):
    columns = list(WIDTHS) if per_width else [None]
    # Unified layout: CoT left, periodicity right, viewed so the duty-factor
    # slope reads left to right with speed receding into depth.
    side_by_side = TASK_VARIANT in ("unified", "specialist") and not per_width
    view = dict(elev=20, azim=-72) if TASK_VARIANT in ("unified", "specialist") else dict(elev=24, azim=-52)
    fig = plt.figure(figsize=(15, 6.8) if side_by_side else (22 if per_width else 11, 9))
    metrics = [("positive_mechanical_cot_median", "Positive mechanical CoT"),
               ("periodic_rate", "Periodicity of realized motion (not χ)")]
    descriptive_plane = TASK_VARIANT in ("unified", "specialist")
    if descriptive_plane:
        metrics[0] = ("positive_mechanical_cot_median",
                      "Positive mechanical CoT (plane: all trials; markers: gate-valid)")
    cot = (data.positive_mechanical_cot_median_all if descriptive_plane
           else data.positive_mechanical_cot_median).dropna()
    top = max(.1, float(cot.max()) * 1.12) if len(cot) else 1.
    for row, (metric, label) in enumerate(metrics):
        for col, width in enumerate(columns):
            ax = (fig.add_subplot(1, 2, row + 1, projection="3d") if side_by_side else
                  fig.add_subplot(2, len(columns), row * len(columns) + col + 1, projection="3d"))
            selected = data if width is None else data[np.isclose(data.step_width, width)]
            missing = 0
            for gait, color in GAIT_COLORS.items():
                if row == 0 and descriptive_plane:
                    measured_mesh(ax, selected[selected.gait == gait],
                                  "positive_mechanical_cot_median_all", color,
                                  DUTIES_BY_GAIT[gait], alpha=.30, faces_only=True)
                missing += measured_mesh(ax, selected[selected.gait == gait], metric, color,
                                         DUTIES_BY_GAIT[gait])
            short = ("Positive mechanical CoT" if row == 0 else "Periodicity")
            ax.set(xlabel="Commanded duty factor", ylabel="Speed (m/s)",
                   zlabel=short if TASK_VARIANT in ("unified", "specialist") else label,
                   xlim=(min(DUTIES) - .01, max(DUTIES) + .01), ylim=(.24, .41),
                   zlim=(0, top) if row == 0 else (0, 1.05))
            ax.set_xticks(DUTIES)
            ax.set_yticks(SPEEDS)
            ax.view_init(**view)
            if TASK_VARIANT in ("unified", "specialist"):
                ax.set_title(f"Width {width:.2f} m" if width is not None else label, fontsize=11)
            else:
                ax.set_title((f"Width {width:.2f} m" if width is not None else label)
                             + (f" • {missing} invalid vertices omitted" if missing else ""), fontsize=10)
    fig.suptitle(title, fontsize=15, y=.98)
    fig.legend(handles=[Patch(facecolor=GAIT_COLORS[gait], edgecolor=GAIT_COLORS[gait],
                              label=gait.title(), alpha=.72) for gait in GAITS],
               loc="upper center", bbox_to_anchor=(.5, .93 if side_by_side else .958), ncol=2,
               frameon=True, fancybox=False, framealpha=1., edgecolor="#777777",
               fontsize=12, handlelength=2.5, columnspacing=2.5)
    footer = ("Frozen seed-2 paper policy • flat ground • period 0.48 s • exploratory, one policy • no χ estimates.\n"
              + ("Each width shown separately; invalid CoT vertices and adjacent faces omitted."
                 if per_width else "CoT shown only with all five widths valid; equal-width aggregation. Periodicity includes all trials.")
              + "\nWalk DF < 0.75 was outside training support. Periodicity does not establish requested gait/DF compliance.")
    if TEXT["footer"]:
        fig.text(.5, .025, footer, ha="center", fontsize=9)
    if side_by_side:
        fig.subplots_adjust(left=.02, right=.96, bottom=.06, top=.84, wspace=.10)
    else:
        fig.subplots_adjust(left=.01, right=.94, bottom=.10, top=.88, hspace=.25, wspace=.12)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, nargs="+",
                        help="Grid directory, or one per gait for the specialists")
    parser.add_argument("--output", type=Path, default=ROOT / TEXT["output"])
    parser.add_argument("--task", choices=("paper", "unified", "specialist"), default="paper")
    args = parser.parse_args()
    loaded = [load_results(directory) for directory in args.input]
    manifest = loaded[0][0]
    trials = pd.concat([item[1] for item in loaded], ignore_index=True)
    if len(args.input) > 1:
        seen = [set(m.get("gaits") or GAITS) for m, _ in loaded]
        if any(a & b for i, a in enumerate(seen) for b in seen[i + 1:]):
            raise ValueError("Each gait must come from exactly one grid")
    cells = summarize_cells(trials)
    factors = summarize_factors(cells)
    args.output.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.output / "surface_cells.csv", index=False)
    factors.to_csv(args.output / "surface_factors.csv", index=False)
    plot_surfaces(factors, args.output, "figure_7_paper_style_rl_surfaces", TEXT["title"])
    plot_surfaces(cells, args.output, "rl_surfaces_by_width", TEXT["title_width"], per_width=True)
    (args.output / "provenance.json").write_text(json.dumps(dict(
        input=[str(d.resolve()) for d in args.input],
        checkpoint_sha256=[m["checkpoint_sha256"] for m, _ in loaded],
        completion_sha256=[sha256(d / "SURFACE_COMPLETE") for d in args.input],
        valid_energy_cells=int(cells.energy_condition_valid.sum()), total_cells=len(cells),
        plot_script_sha256=sha256(Path(__file__)), paper_claims_allowed=False), indent=2))
    (args.output / "README.md").write_text(
        TEXT["readme_head"]
        + "The upper panel is positive mechanical CoT; the lower panel is phase-one periodicity, "
        "a diagnostic of realized motion rather than a convergence estimate. "
        "Periodicity does not establish execution of the requested gait or duty factor; "
        "see the companion walking_df_command_fidelity figure. χ remains unmeasured here. "
        f"Each of {len(cells)} cells has 32 matched resets, 12 total cycles, and measurement over the last four. "
        "Physical failures at any time invalidate trial endpoints. Energy cells need at least 90% "
        "valid trials, including command, contact-topology, and periodicity gates. "
        "CoT summaries take the median of four cycles, then the median of valid trials. "
        "The aggregate takes the median of all five width summaries and requires all five widths valid; "
        "missing cells are not interpolated. "
        "The companion figure separates widths. All trial denominators are retained.\n")
    print(json.dumps(dict(output=str(args.output), valid_energy_cells=int(cells.energy_condition_valid.sum()),
                          total_cells=len(cells))))


if __name__ == "__main__":
    main()
