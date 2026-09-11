"""Explain gaps in old-policy surfaces using measured command fidelity."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_policy_surfaces import load_results
from policy_surface_data import DUTIES, GAITS, ROOT, SPEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "PAPER_GRAPHS/old_policy_walk_df")
    args = parser.parse_args()
    _, trials = load_results(args.input)
    groups = trials.groupby(["gait", "speed", "command_df"], as_index=False).agg(
        achieved_df=("achieved_df", "mean"),
        command_compliance=("compliant", "mean"),
        periodicity=("periodic_orbit_gate_pass", "mean"),
        trials=("seed", "size"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for col, gait in enumerate(GAITS):
        group = groups[groups.gait == gait]
        for row, (metric, label, bounds, cmap) in enumerate((
                ("achieved_df", "Mean achieved duty factor", (.45, .80), "viridis"),
                ("command_compliance", "Command/contact/energy compliance", (0, 1), "RdYlGn"))):
            axis = axes[row, col]
            values = group.pivot(index="speed", columns="command_df", values=metric).reindex(
                index=SPEEDS, columns=DUTIES).to_numpy()
            im = axis.imshow(values, vmin=bounds[0], vmax=bounds[1], cmap=cmap, aspect="auto")
            for i, j in np.ndindex(values.shape):
                axis.text(j, i, f"{values[i, j]:.3f}" if row == 0 else f"{values[i, j]:.0%}",
                          ha="center", va="center", color="black" if row == 1 else "white", fontsize=11)
            axis.set_xticks(range(3), [f"{d:g}" for d in DUTIES])
            axis.set_yticks(range(4), [f"{s:.2f}" for s in SPEEDS])
            axis.set(xlabel="Commanded duty factor", ylabel="Speed (m/s)",
                     title=f"{gait.title()} — {label}")
            fig.colorbar(im, ax=axis, shrink=.8)
    fig.suptitle("Original paper policy: achieved duty factor and command compliance", fontsize=14)
    fig.supxlabel("160 trials per cell (5 widths × 32 resets). Walk DF < 0.75 was absent from training.\n"
                  "Periodic motion alone does not establish execution of the requested gait.", fontsize=10)
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"walking_df_command_fidelity.{extension}", dpi=200)
    groups.to_csv(args.output / "walking_df_command_fidelity.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
