#!/usr/bin/env python3
"""Build publication figures from the audited flat-ground evaluation outputs.

The script deliberately keeps command-fidelity evidence separate from the
invalid return-map estimates.  It never plots the suppressed V4/V5 chi values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TROT = ROOT / "results/validation_v4_seed2_v030_p048_trot_20260911/summary.csv"
DEFAULT_WALK = ROOT / "results/validation_v4_seed2_v030_p048_walk_20260911/summary.csv"
DEFAULT_STABILITY = ROOT / "results/stability_v4_seed2_descriptive_20260911"
DEFAULT_PILOT = ROOT / "results/stability_v5_estimator_pilot_seed1100000_20260911/stability_references.csv"
DEFAULT_OUTPUT = ROOT / "PAPER_GRAPHS"

COLORS = {
    "trot_050": "#7A5195",
    "trot_0625": "#374C80",
    "trot_075": "#EF5675",
    "walk_075": "#2A9D8F",
    "pass": "#2A9D8F",
    "diagnostic": "#F4A261",
    "missing": "#B8BEC9",
    "failed": "#D1495B",
    "ink": "#243447",
}


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def condition_label(gait: str, duty_factor: float) -> str:
    if gait == "walk":
        return "Walk, DF 0.75"
    return f"Trot, DF {duty_factor:g}"


def condition_color(gait: str, duty_factor: float) -> str:
    if gait == "walk":
        return COLORS["walk_075"]
    if np.isclose(duty_factor, 0.5):
        return COLORS["trot_050"]
    if np.isclose(duty_factor, 0.625):
        return COLORS["trot_0625"]
    return COLORS["trot_075"]


def load_nominal(trot_path: Path, walk_path: Path) -> pd.DataFrame:
    frames = []
    for gait, path in (("trot", trot_path), ("walk", walk_path)):
        frame = pd.read_csv(path)
        frame["gait"] = gait
        frames.append(frame)
    nominal = pd.concat(frames, ignore_index=True)
    needed = {
        "gait",
        "command_df",
        "step_width",
        "body_achieved_width",
        "achieved_df",
        "forward_speed",
        "lateral_rmse",
        "heading_rmse_rad",
        "combined_cell_pass",
        "n",
    }
    missing = needed.difference(nominal.columns)
    if missing:
        raise ValueError(f"Nominal summaries are missing columns: {sorted(missing)}")
    return nominal.sort_values(["gait", "command_df", "step_width"]).reset_index(drop=True)


def make_command_figure(nominal: pd.DataFrame, output: Path) -> dict[str, float | int]:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    groups = list(nominal.groupby(["gait", "command_df"], sort=True))

    ax = axes[0, 0]
    for (gait, df), group in groups:
        ax.plot(
            group["step_width"],
            group["body_achieved_width"],
            marker="o",
            lw=1.8,
            ms=5,
            color=condition_color(gait, df),
            label=condition_label(gait, df),
        )
    ax.plot([0.08, 0.52], [0.08, 0.52], ls="--", lw=1.2, color="#5B6573", label="Ideal")
    ax.set(xlim=(0.08, 0.52), ylim=(0.08, 0.52), xlabel="Commanded full stance width (m)", ylabel="Achieved body-frame width (m)")
    ax.set_title("a  Stance-width commands are realized", loc="left", fontweight="bold")

    ax = axes[0, 1]
    offsets = {"trot": -0.006, "walk": 0.006}
    for (gait, df), group in groups:
        x = group["command_df"].to_numpy() + offsets[gait]
        ax.scatter(x, group["achieved_df"], s=38, color=condition_color(gait, df), edgecolor="white", linewidth=0.6)
    ax.plot([0.47, 0.78], [0.47, 0.78], ls="--", lw=1.2, color="#5B6573")
    ax.set(xlim=(0.47, 0.78), ylim=(0.47, 0.78), xlabel="Commanded duty factor", ylabel="Achieved duty factor")
    ax.set_xticks([0.50, 0.625, 0.75])
    ax.set_title("b  Duty-factor commands are realized", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for (gait, df), group in groups:
        ax.plot(group["step_width"], group["lateral_rmse"], marker="o", lw=1.8, ms=5, color=condition_color(gait, df))
    ax.axhline(0.10, ls="--", color=COLORS["failed"], lw=1.2, label="Frozen limit (0.10 m)")
    ax.set(xlim=(0.08, 0.52), ylim=(0, 0.105), xlabel="Commanded full stance width (m)", ylabel="Lateral RMSE (m)")
    ax.set_title("c  Robot stays near the straight line", loc="left", fontweight="bold")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for (gait, df), group in groups:
        ax.plot(group["step_width"], group["heading_rmse_rad"], marker="o", lw=1.8, ms=5, color=condition_color(gait, df))
    ax.axhline(0.10, ls="--", color=COLORS["failed"], lw=1.2, label="Frozen limit (0.10 rad)")
    ax.set(xlim=(0.08, 0.52), ylim=(0, 0.105), xlabel="Commanded full stance width (m)", ylabel="Heading RMSE (rad)")
    ax.set_title("d  Heading remains forward", loc="left", fontweight="bold")
    ax.legend(loc="upper right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Flat-ground command validation: 1,280 held-out rollouts", fontsize=14, fontweight="bold", y=1.035)
    fig.text(0.5, 0.005, "Seed-2 V4 evaluation; speed = 0.30 m/s, period = 0.48 s; each point summarizes 64 held-out rollouts.", ha="center", color="#4D5968", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_figure(fig, output, "figure_1_command_fidelity")

    return {
        "cells": int(len(nominal)),
        "rollouts": int(nominal["n"].sum()),
        "passing_cells": int(nominal["combined_cell_pass"].astype(bool).sum()),
        "max_width_abs_error_m": float((nominal["body_achieved_width"] - nominal["step_width"]).abs().max()),
        "max_df_abs_error": float((nominal["achieved_df"] - nominal["command_df"]).abs().max()),
        "max_speed_abs_error_mps": float((nominal["forward_speed"] - 0.30).abs().max()),
        "max_lateral_rmse_m": float(nominal["lateral_rmse"].max()),
        "max_heading_rmse_rad": float(nominal["heading_rmse_rad"].max()),
    }


def make_periodicity_figure(references: pd.DataFrame, output: Path) -> pd.DataFrame:
    radius = references[np.isclose(references["perturbation_h"], 0.05)].copy()
    summary = (
        radius.groupby(["gait", "command_df"], as_index=False)
        .agg(references=("periodic_orbit_gate_pass", "size"), passed=("periodic_orbit_gate_pass", "sum"), periodic_rate=("periodic_orbit_gate_pass", "mean"))
    )
    summary["label"] = [condition_label(g, d) for g, d in zip(summary["gait"], summary["command_df"])]
    colors = [condition_color(g, d) for g, d in zip(summary["gait"], summary["command_df"])]

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    x = np.arange(len(summary))
    bars = ax.bar(x, 100 * summary["periodic_rate"], color=colors, width=0.68)
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.7, f"{100*row.periodic_rate:.1f}%\n({row.passed}/{row.references})", ha="center", va="bottom", fontsize=9)
    ax.axhline(90, ls="--", lw=1.2, color=COLORS["failed"], label="Frozen coverage target (90%)")
    ax.set_xticks(x, summary["label"])
    ax.set_ylim(0, 113)
    ax.set_ylabel("References passing phase-one periodicity gate (%)")
    ax.set_title("Low-duty-factor trot is the only periodicity bottleneck", loc="left", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.5, -0.01, "Diagnostic only: this gate is a prerequisite for the return map and is not the paper's convergence metric χ.", ha="center", color=COLORS["failed"], fontsize=9.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, "figure_2_periodicity_diagnostic")
    return summary


def make_energy_coverage_figure(energy: pd.DataFrame, output: Path) -> pd.DataFrame:
    coverage = (
        energy.groupby(["gait", "command_df", "speed"], as_index=False)
        .agg(valid_widths=("energy_condition_valid", "sum"), widths=("energy_condition_valid", "size"))
    )
    coverage["coverage"] = coverage["valid_widths"] / coverage["widths"]
    row_order = [("trot", 0.50), ("trot", 0.625), ("trot", 0.75), ("walk", 0.75)]
    speeds = sorted(coverage["speed"].unique())
    matrix = np.zeros((len(row_order), len(speeds)))
    counts = np.zeros_like(matrix, dtype=int)
    for i, (gait, df) in enumerate(row_order):
        for j, speed in enumerate(speeds):
            row = coverage[(coverage["gait"] == gait) & np.isclose(coverage["command_df"], df) & np.isclose(coverage["speed"], speed)].iloc[0]
            matrix[i, j] = row["coverage"]
            counts[i, j] = int(row["valid_widths"])

    fig, ax = plt.subplots(figsize=(8.7, 5.1))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.67 else COLORS["ink"]
            ax.text(j, i, f"{counts[i,j]}/5", ha="center", va="center", color=color, fontweight="bold", fontsize=10)
    ax.set_xticks(range(len(speeds)), [f"{s:.2f}" for s in speeds])
    ax.set_yticks(range(len(row_order)), [condition_label(g, d) for g, d in row_order])
    ax.set_xlabel("Commanded speed (m/s)")
    ax.set_ylabel("Gait condition")
    ax.set_title("Mechanical-CoT validity coverage is factor dependent", loc="left", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of stance widths valid")
    fig.text(0.5, -0.01, "Only 65/80 cells are valid; missing low-DF trot cells prevent unbiased efficiency correlations.", ha="center", color=COLORS["failed"], fontsize=9.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, "figure_3_energy_validity_coverage")
    return coverage


def make_estimator_figure(pilot: pd.DataFrame, output: Path) -> dict[str, float | int]:
    pilot = pilot.sort_values("perturbation_h")
    h = pilot["perturbation_h"].to_numpy()
    noise_ratio = pilot["zero_clone_noise_fraction_of_h"].to_numpy()
    translation = pilot["translation_identity_residual_max"].to_numpy()
    topology = pilot["hybrid_topology_gate_pass"].astype(bool).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    ax.plot(h, noise_ratio, marker="o", ms=6, lw=2, color=COLORS["failed"])
    ax.axhline(0.05, ls="--", lw=1.3, color=COLORS["ink"], label="Required maximum")
    ax.set(xscale="log", yscale="log", xlabel="Finite-difference radius h", ylabel="Zero-clone divergence / h")
    ax.set_xticks(h, [f"{x:g}" for x in h])
    ax.set_title("a  Hidden-state noise exceeds every radius", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    for x, y, passed in zip(h, translation, topology):
        ax.scatter(x, y, s=70, marker="o" if passed else "X", color=COLORS["diagnostic"] if passed else COLORS["failed"], edgecolor="white", linewidth=0.7, zorder=3)
    ax.plot(h, translation, lw=1.7, color=COLORS["diagnostic"], alpha=0.8)
    ax.axhline(0.02, ls="--", lw=1.3, color=COLORS["ink"], label="Required maximum")
    ax.set(xscale="log", yscale="log", xlabel="Finite-difference radius h", ylabel="Translation-identity residual")
    ax.set_xticks(h, [f"{x:g}" for x in h])
    ax.set_title("b  Translation check also fails", loc="left", fontweight="bold")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["diagnostic"], markeredgecolor="white", markersize=8, label="Topology preserved"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["failed"], markeredgecolor="white", markersize=8, label="Topology changed"),
    ]
    ax.legend(handles=legend, frameon=False, loc="upper right")

    fig.suptitle("Return-map validity audit: no χ estimate is releasable", fontsize=14, fontweight="bold")
    fig.text(0.5, -0.015, "The pilot validates the exposed-state fork but fails reproducibility and symmetry; raw χ values are intentionally suppressed.", ha="center", color=COLORS["failed"], fontsize=9.3, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    save_figure(fig, output, "figure_4_return_map_validity")
    return {
        "radii_tested": int(len(pilot)),
        "valid_chi_estimates": int(pilot["reference_valid"].astype(bool).sum()),
        "zero_clone_output_norm": float(pilot["zero_clone_output_max_error"].max()),
        "min_noise_over_h": float(noise_ratio.min()),
        "max_noise_over_h": float(noise_ratio.max()),
        "topology_passing_radii": int(topology.sum()),
        "translation_passing_radii": int((translation <= 0.02).sum()),
    }


def make_claim_status_figure(output: Path) -> pd.DataFrame:
    rows = [
        ("Policy executes width, DF and gait\nwhile walking straight", "Supported", "20/20 cells; 1,280/1,280 rollouts"),
        ("Higher DF improves convergence", "Diagnostic only", "Periodicity: 61.25% at trot DF 0.50; 100% otherwise"),
        ("Narrower stance worsens convergence", "Missing", "No valid χ-by-width comparison"),
        ("Speed has weak convergence effect", "Missing", "0.25–0.40 m/s ran; χ invalid, so no usable comparison"),
        ("Matched walk and trot converge similarly", "Missing", "No valid χ-by-gait comparison"),
        ("DF and speed affect efficiency; gait is weak", "Suppressed", "65/80 valid CoT cells; missingness depends on factors"),
        ("Width has little efficiency effect", "Suppressed", "Incomplete matched CoT pairs"),
    ]
    status = pd.DataFrame(rows, columns=["claim", "status", "evidence"])
    status_colors = {
        "Supported": COLORS["pass"],
        "Diagnostic only": COLORS["diagnostic"],
        "Suppressed": COLORS["failed"],
        "Missing": COLORS["missing"],
    }

    fig, ax = plt.subplots(figsize=(13.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, "Evidence status against the paper's claims", fontsize=16, fontweight="bold", va="top")
    ax.text(0.02, 0.885, "Claim", fontweight="bold", color="#596575")
    ax.text(0.43, 0.885, "Status", fontweight="bold", color="#596575", ha="center")
    ax.text(0.56, 0.885, "Current evidence", fontweight="bold", color="#596575")

    row_top = 0.85
    row_height = 0.105
    for idx, row in status.iterrows():
        y_top = row_top - idx * row_height
        y_mid = y_top - row_height / 2
        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.012, y_top - row_height), 0.976, row_height, facecolor="#F6F8FA", edgecolor="none"))
        ax.text(0.02, y_mid, row["claim"], ha="left", va="center", fontsize=10, fontweight="bold")
        color = status_colors[row["status"]]
        ax.add_patch(
            FancyBboxPatch(
                (0.355, y_mid - 0.025),
                0.15,
                0.05,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=color,
                edgecolor="none",
            )
        )
        label_color = COLORS["ink"] if row["status"] == "Missing" else "white"
        ax.text(0.43, y_mid, row["status"], ha="center", va="center", color=label_color, fontweight="bold", fontsize=9.2)
        ax.text(0.56, y_mid, row["evidence"], ha="left", va="center", fontsize=9.4, color=COLORS["ink"])

    fig.text(0.5, 0.025, "Flat-ground, forward-only scope; nominal motor gains; one trained seed. χ claims require a valid estimator and independent policies.", ha="center", fontsize=9, color="#4D5968")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.985, bottom=0.07)
    save_figure(fig, output, "figure_5_paper_claim_status")
    return status


def make_factor_space_figure(
    references: pd.DataFrame, energy: pd.DataFrame, output: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = energy.copy()
    energy["valid"] = energy["energy_condition_valid"].astype(bool)
    energy_rows = []
    for (gait, duty_factor, speed), group in energy.groupby(["gait", "command_df", "speed"]):
        valid = group[group["valid"]]
        energy_rows.append(
            {
                "gait": gait,
                "command_df": duty_factor,
                "speed": speed,
                "valid_widths": int(len(valid)),
                "total_widths": int(len(group)),
                "positive_mechanical_cot_median": float(valid["positive_mechanical_cot_median"].median()) if len(valid) else np.nan,
            }
        )
    cot = pd.DataFrame(energy_rows).sort_values(["gait", "command_df", "speed"])

    radius = references[np.isclose(references["perturbation_h"], 0.05)]
    periodicity = (
        radius.groupby(["gait", "command_df", "speed"], as_index=False)
        .agg(references=("reference", "size"), periodic_rate=("periodic_orbit_gate_pass", "mean"))
        .sort_values(["gait", "command_df", "speed"])
    )

    fig = plt.figure(figsize=(13.0, 6.2))
    ax_cot = fig.add_subplot(1, 2, 1, projection="3d")
    for (gait, duty_factor), group in cot.groupby(["gait", "command_df"], sort=True):
        group = group.sort_values("speed")
        valid = group["positive_mechanical_cot_median"].notna()
        color = condition_color(gait, duty_factor)
        sizes = 45 + 85 * group.loc[valid, "valid_widths"] / group.loc[valid, "total_widths"]
        ax_cot.plot(
            group.loc[valid, "speed"],
            group.loc[valid, "command_df"],
            group.loc[valid, "positive_mechanical_cot_median"],
            color=color,
            lw=1.8,
            alpha=0.85,
        )
        ax_cot.scatter(
            group.loc[valid, "speed"],
            group.loc[valid, "command_df"],
            group.loc[valid, "positive_mechanical_cot_median"],
            s=sizes,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
        )
        missing = ~valid
        if missing.any():
            ax_cot.scatter(
                group.loc[missing, "speed"],
                group.loc[missing, "command_df"],
                np.full(int(missing.sum()), 0.185),
                marker="X",
                s=85,
                color=COLORS["failed"],
                depthshade=False,
            )
    ax_cot.set(
        xlabel="Speed (m/s)",
        ylabel="Duty factor",
        zlabel="Positive mechanical CoT",
        xlim=(0.235, 0.415),
        ylim=(0.48, 0.77),
        zlim=(0.18, 0.42),
    )
    ax_cot.set_xticks([0.25, 0.30, 0.35, 0.40])
    ax_cot.set_yticks([0.50, 0.625, 0.75])
    ax_cot.view_init(elev=24, azim=-58)
    ax_cot.set_title("a  Mechanical CoT (valid cells only)", loc="left", fontweight="bold", pad=12)

    ax_periodic = fig.add_subplot(1, 2, 2, projection="3d")
    for (gait, duty_factor), group in periodicity.groupby(["gait", "command_df"], sort=True):
        group = group.sort_values("speed")
        color = condition_color(gait, duty_factor)
        ax_periodic.plot(
            group["speed"],
            group["command_df"],
            100 * group["periodic_rate"],
            marker="o",
            ms=6,
            lw=2,
            color=color,
            label=condition_label(gait, duty_factor),
        )
    ax_periodic.set(
        xlabel="Speed (m/s)",
        ylabel="Duty factor",
        zlabel="Phase-one periodicity (%)",
        xlim=(0.235, 0.415),
        ylim=(0.48, 0.77),
        zlim=(0, 105),
    )
    ax_periodic.set_xticks([0.25, 0.30, 0.35, 0.40])
    ax_periodic.set_yticks([0.50, 0.625, 0.75])
    ax_periodic.zaxis.labelpad = 1
    ax_periodic.view_init(elev=24, azim=-58)
    ax_periodic.set_title("b  Periodicity prerequisite (not χ)", loc="left", fontweight="bold", pad=12)

    legend_handles = [
        Line2D([0], [0], marker="o", color=COLORS["trot_050"], lw=2, label="Trot, DF 0.50"),
        Line2D([0], [0], marker="o", color=COLORS["trot_0625"], lw=2, label="Trot, DF 0.625"),
        Line2D([0], [0], marker="o", color=COLORS["trot_075"], lw=2, label="Trot, DF 0.75"),
        Line2D([0], [0], marker="o", color=COLORS["walk_075"], lw=2, label="Walk, DF 0.75"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["failed"], markeredgecolor=COLORS["failed"], label="Invalid CoT cell"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5, frameon=False)
    fig.suptitle("Speed × duty-factor view of the available evidence", fontsize=15, fontweight="bold", y=1.01)
    fig.text(
        0.5,
        0.015,
        "Descriptive seed-2 data. CoT missingness depends on the factors; periodicity is a convergence prerequisite, not the paper's χ metric.",
        ha="center",
        color=COLORS["failed"],
        fontsize=9.2,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.01, right=0.93, top=0.86, bottom=0.10, wspace=0.08)
    save_figure(fig, output, "figure_6_speed_df_cot_periodicity_3d")
    return cot, periodicity


def make_paper_style_surfaces(
    cot: pd.DataFrame, periodicity: pd.DataFrame, output: Path
) -> None:
    """Render response surfaces in the layout used by the reference paper."""

    trot_cot = cot[cot["gait"] == "trot"].pivot(
        index="speed", columns="command_df", values="positive_mechanical_cot_median"
    ).sort_index().sort_index(axis=1)
    trot_periodicity = periodicity[periodicity["gait"] == "trot"].pivot(
        index="speed", columns="command_df", values="periodic_rate"
    ).sort_index().sort_index(axis=1)
    walk_cot = cot[cot["gait"] == "walk"].sort_values("speed")
    walk_periodicity = periodicity[periodicity["gait"] == "walk"].sort_values("speed")

    fig = plt.figure(figsize=(12.5, 8.1))
    ax_cot = fig.add_subplot(2, 1, 1, projection="3d")
    duty_grid, speed_grid = np.meshgrid(
        trot_cot.columns.to_numpy(), trot_cot.index.to_numpy()
    )
    ax_cot.plot_surface(
        duty_grid,
        speed_grid,
        trot_cot.to_numpy(),
        color="#0072BD",
        edgecolor="#0072BD",
        linewidth=0.65,
        alpha=0.82,
        antialiased=True,
    )
    ax_cot.plot(
        walk_cot["command_df"],
        walk_cot["speed"],
        walk_cot["positive_mechanical_cot_median"],
        color="#D95319",
        marker="s",
        ms=6,
        lw=3,
        label="Walk (measured at DF 0.75 only)",
    )
    missing = trot_cot.isna().stack()
    for (speed, duty_factor), is_missing in missing.items():
        if is_missing:
            ax_cot.scatter(
                duty_factor,
                speed,
                0.185,
                marker="X",
                s=70,
                color=COLORS["failed"],
                depthshade=False,
            )
    ax_cot.set(
        xlabel="Duty factor",
        ylabel="Speed (m/s)",
        zlabel="Positive mechanical CoT",
        xlim=(0.48, 0.77),
        ylim=(0.24, 0.41),
        zlim=(0.18, 0.42),
    )
    ax_cot.set_xticks([0.50, 0.625, 0.75])
    ax_cot.set_yticks([0.25, 0.30, 0.35, 0.40])
    ax_cot.view_init(elev=24, azim=-52)
    ax_cot.set_title("a  Cost of transport", loc="left", fontsize=12, fontweight="bold")

    ax_periodic = fig.add_subplot(2, 1, 2, projection="3d")
    duty_grid, speed_grid = np.meshgrid(
        trot_periodicity.columns.to_numpy(), trot_periodicity.index.to_numpy()
    )
    ax_periodic.plot_surface(
        duty_grid,
        speed_grid,
        trot_periodicity.to_numpy(),
        color="#0072BD",
        edgecolor="#0072BD",
        linewidth=0.65,
        alpha=0.82,
        antialiased=True,
    )
    ax_periodic.plot(
        walk_periodicity["command_df"],
        walk_periodicity["speed"],
        walk_periodicity["periodic_rate"],
        color="#D95319",
        marker="s",
        ms=6,
        lw=3,
    )
    ax_periodic.set(
        xlabel="Duty factor",
        ylabel="Speed (m/s)",
        zlabel="Periodicity rate",
        xlim=(0.48, 0.77),
        ylim=(0.24, 0.41),
        zlim=(0.45, 1.03),
    )
    ax_periodic.set_xticks([0.50, 0.625, 0.75])
    ax_periodic.set_yticks([0.25, 0.30, 0.35, 0.40])
    ax_periodic.view_init(elev=24, azim=-52)
    ax_periodic.set_title(
        "b  Convergence prerequisite: phase-one periodicity (not χ)",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    handles = [
        Patch(facecolor="#0072BD", edgecolor="#0072BD", label="Trot"),
        Patch(facecolor="#D95319", edgecolor="#D95319", label="Walk"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=COLORS["failed"], markeredgecolor=COLORS["failed"], label="Invalid/missing CoT"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True,
               fancybox=False, framealpha=1., edgecolor="#777777",
               fontsize=11, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("RL response surfaces in the paper's factor-space format", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.018,
        "Measured seed-2 data only: no walk surface was inferred from one duty-factor row, and no invalid χ estimate is displayed.",
        ha="center",
        color=COLORS["failed"],
        fontsize=9.3,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.02, right=0.94, top=0.90, bottom=0.075, hspace=0.20)
    save_figure(fig, output, "figure_7_paper_style_rl_surfaces")


def write_readme(output: Path, command: dict, estimator: dict) -> None:
    text = f"""# Paper-claim figure set

These figures are generated from the audited seed-2 flat-ground evaluations.
All PNG files are directly in this folder; the matching PDFs are vector exports.

| Figure | What it establishes | Scientific limit |
|---|---|---|
| `figure_1_command_fidelity` | The policy realizes commanded stance width and duty factor while tracking 0.30 m/s and staying straight. All {command['passing_cells']}/{command['cells']} cells passed ({command['rollouts']} rollouts). | This validates the treatment variables; it is not a convergence result. |
| `figure_2_periodicity_diagnostic` | Low-DF trot is the only stratum with reduced phase-one periodicity (61.25% versus 100%). | Directionally agrees with the paper's duty-factor claim, but periodicity is not chi. |
| `figure_3_energy_validity_coverage` | Shows exactly where mechanical-CoT data pass the validity gate. | Only 65/80 cells pass and missingness is factor dependent, so efficiency correlations are suppressed. |
| `figure_4_return_map_validity` | Explains why the return-map result is withheld: zero-clone noise and translation symmetry fail at every tested radius. | {estimator['valid_chi_estimates']} valid chi estimates; raw values must not be shown as policy evidence. |
| `figure_5_paper_claim_status` | Slide-ready map from each paper claim to the evidence currently available. | A single trained seed cannot establish population-level paper replication. |
| `figure_6_speed_df_cot_periodicity_3d` | Three-axis view of speed, duty factor, mechanical CoT, and the periodicity prerequisite. | CoT is descriptive with factor-dependent missingness; periodicity is not chi. |
| `figure_7_paper_style_rl_surfaces` | Paper-style response-surface layout using the RL measurements. | Trot forms a measured surface; walk has only one duty-factor row, and periodicity replaces unavailable chi only as a labeled diagnostic. |

## Direct quantitative result

- Maximum stance-width error across cell means: {command['max_width_abs_error_m']:.4f} m.
- Maximum duty-factor error across cell means: {command['max_df_abs_error']:.4f}.
- Maximum speed error from 0.30 m/s: {command['max_speed_abs_error_mps']:.4f} m/s.
- Maximum lateral RMSE: {command['max_lateral_rmse_m']:.4f} m, against the 0.10 m limit.
- Maximum heading RMSE: {command['max_heading_rmse_rad']:.4f} rad, against the 0.10 rad limit.

## Source data

- `results/validation_v4_seed2_v030_p048_trot_20260911/summary.csv`
- `results/validation_v4_seed2_v030_p048_walk_20260911/summary.csv`
- `results/stability_v4_seed2_descriptive_20260911/stability_references.csv`
- `results/stability_v4_seed2_descriptive_20260911/paper_energy_conditions.csv`
- `results/stability_v5_estimator_pilot_seed1100000_20260911/stability_references.csv`

Regenerate with `python scripts/make_paper_figures.py`. No simulator or GPU is used.
"""
    (output / "README.md").write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trot-summary", type=Path, default=DEFAULT_TROT)
    parser.add_argument("--walk-summary", type=Path, default=DEFAULT_WALK)
    parser.add_argument("--stability-dir", type=Path, default=DEFAULT_STABILITY)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    nominal = load_nominal(args.trot_summary, args.walk_summary)
    references = pd.read_csv(args.stability_dir / "stability_references.csv")
    energy = pd.read_csv(args.stability_dir / "paper_energy_conditions.csv")
    pilot = pd.read_csv(args.pilot)

    command_metrics = make_command_figure(nominal, output)
    periodicity = make_periodicity_figure(references, output)
    energy_coverage = make_energy_coverage_figure(energy, output)
    estimator_metrics = make_estimator_figure(pilot, output)
    claim_status = make_claim_status_figure(output)
    cot_factor_space, periodicity_factor_space = make_factor_space_figure(references, energy, output)
    make_paper_style_surfaces(cot_factor_space, periodicity_factor_space, output)

    nominal.to_csv(output / "figure_1_command_fidelity_data.csv", index=False)
    periodicity.to_csv(output / "figure_2_periodicity_data.csv", index=False)
    energy_coverage.to_csv(output / "figure_3_energy_coverage_data.csv", index=False)
    pilot.to_csv(output / "figure_4_estimator_validity_data.csv", index=False)
    claim_status.to_csv(output / "figure_5_claim_status_data.csv", index=False)
    cot_factor_space.to_csv(output / "figure_6_cot_speed_df_data.csv", index=False)
    periodicity_factor_space.to_csv(output / "figure_6_periodicity_speed_df_data.csv", index=False)
    metrics = {"command_fidelity": command_metrics, "estimator_validity": estimator_metrics}
    (output / "key_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_readme(output, command_metrics, estimator_metrics)
    print(f"Wrote paper figures to {output}")


if __name__ == "__main__":
    main()
