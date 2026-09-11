"""Analyze and plot the push-free empirical gait return-map experiment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.stability import (
    analyze_stability, evaluation_source_hash)

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()

import json
manifest = json.loads(
    (args.directory / "stability_manifest.json").read_text())
if manifest["evaluation_sha256"] != evaluation_source_hash(ROOT):
    raise ValueError(
        "Current evaluator source differs from the run; use its source snapshot")
analyze_stability(args.directory)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

conditions_path = args.directory / "paper_conditions.csv"
if not conditions_path.exists():
    raise ValueError("No central-h conditions were produced")
table = pd.read_csv(conditions_path)
valid = table[table.condition_valid.astype(bool)].copy()
metric = "chi_orbital_augmented_median"


def save(fig, name):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.directory / f"{name}.{extension}", dpi=180)
    plt.close(fig)


def invalid_message(ax):
    ax.text(.5, .5, "No conditions passed all preregistered gates",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def scope_note(ax, text):
    ax.text(.99, .01, text, ha="right", va="bottom", fontsize=8,
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"})


fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
for ax, gait in zip(axes[0], ("trot", "walk")):
    subset = valid[valid.gait == gait]
    if subset.empty:
        invalid_message(ax)
        continue
    grouped = subset.groupby(["command_df", "step_width"], as_index=False)[metric].median()
    for width, group in grouped.groupby("step_width"):
        ax.plot(group.command_df, group[metric], "o-", label=f"{width:.2f} m")
    ax.set(title=gait.title(), xlabel="Achieved-command gate: duty factor",
           ylabel=r"Orbital convergence $\chi_{orb}$ (lower is better)")
    ax.legend(title="Stance width")
    scope_note(ax, f"Valid cells only: {len(subset)}/{int((table.gait == gait).sum())}")
save(fig, "paper_convergence_vs_duty_factor")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
for ax, gait in zip(axes[0], ("trot", "walk")):
    subset = valid[valid.gait == gait]
    if subset.empty:
        invalid_message(ax)
        continue
    grouped = subset.groupby(["step_width", "command_df"], as_index=False)[metric].median()
    for duty, group in grouped.groupby("command_df"):
        ax.plot(group.step_width, group[metric], "o-", label=f"DF {duty:.3f}")
    ax.axvline(.30, color="black", ls=":", alpha=.5, label="Nominal comparison")
    ax.set(title=gait.title(), xlabel="Commanded full stance width (m)",
           ylabel=r"Orbital convergence $\chi_{orb}$ (lower is better)")
    ax.legend()
    scope_note(ax, f"Valid cells only: {len(subset)}/{int((table.gait == gait).sum())}")
save(fig, "paper_convergence_vs_stance_width")

fig, ax = plt.subplots(figsize=(7, 4))
if valid.empty:
    invalid_message(ax)
else:
    grouped = valid.groupby(["speed", "gait"], as_index=False)[metric].median()
    for gait, group in grouped.groupby("gait"):
        ax.plot(group.speed, group[metric], "o-", label=gait.title())
    ax.set(xlabel="Commanded forward speed (m/s)",
           ylabel=r"Orbital convergence $\chi_{orb}$ (lower is better)")
    ax.legend()
    scope_note(ax, f"Valid cells only: {len(valid)}/{len(table)}")
save(fig, "paper_convergence_vs_speed")

fig, ax = plt.subplots(figsize=(6, 5))
if valid.empty:
    invalid_message(ax)
else:
    ax.scatter(
        valid["chi_orbital_augmented_median"],
        valid["chi_augmented_median"], alpha=.7)
    low = min(
        valid["chi_orbital_augmented_median"].min(),
        valid["chi_augmented_median"].min())
    high = max(
        valid["chi_orbital_augmented_median"].max(),
        valid["chi_augmented_median"].max())
    ax.plot([low, high], [low, high], "k:", alpha=.5)
    ax.set(
        xlabel=r"Primary orbital $\chi_{orb}$ (46D)",
        ylabel=r"Unquotiented diagnostic $\chi_{full}$ (48D)")
    scope_note(ax, f"Valid cells only: {len(valid)}/{len(table)}")
save(fig, "paper_orbital_vs_full_diagnostic")

fig, ax = plt.subplots(figsize=(7, 4))
rates = table.groupby(["gait", "command_df"], as_index=False).agg(
    command_gate_rate=("command_gate_pass_rate", "mean"),
    orbit_gate_rate=("periodic_orbit_gate_pass_rate", "mean"),
    translation_gate_rate=("translation_symmetry_gate_pass_rate", "mean"),
    finite_difference_gate_rate=("finite_difference_gate_pass_rate", "mean"))
if rates.empty:
    invalid_message(ax)
else:
    labels = [f"{row.gait}\nDF {row.command_df:g}" for row in rates.itertuples()]
    x = range(len(rates))
    width = .20
    ax.bar([i-1.5*width for i in x], rates.command_gate_rate, width,
           label="Command")
    ax.bar([i-.5*width for i in x], rates.orbit_gate_rate, width,
           label="Periodic orbit")
    ax.bar([i+.5*width for i in x], rates.translation_gate_rate, width,
           label="Translation symmetry")
    ax.bar([i+1.5*width for i in x], rates.finite_difference_gate_rate,
           width, label="Finite difference")
    ax.set(xticks=list(x), xticklabels=labels, ylabel="Reference pass rate",
           ylim=(0, 1.05))
    ax.legend()
    scope_note(ax, f"All planned cells shown: {len(table)}")
save(fig, "paper_validity_gates")

energy_table = pd.read_csv(args.directory / "paper_energy_conditions.csv")
energy_valid = energy_table[
    energy_table.energy_condition_valid.astype(bool)].copy()
energy_metric = "positive_mechanical_cot_median"

fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
for ax, gait in zip(axes[0], ("trot", "walk")):
    subset = energy_valid[energy_valid.gait == gait]
    if subset.empty:
        invalid_message(ax)
        continue
    grouped = subset.groupby(["command_df", "speed"], as_index=False)[
        energy_metric].median()
    for speed, group in grouped.groupby("speed"):
        ax.plot(group.command_df, group[energy_metric], "o-",
                label=f"{speed:.2f} m/s")
    ax.set(title=gait.title(), xlabel="Commanded duty factor",
           ylabel="Positive mechanical CoT (lower is better)")
    ax.legend(title="Speed")
    scope_note(
        ax, f"Valid cells only: {len(subset)}/{int((energy_table.gait == gait).sum())}\n"
        "Positive mechanical proxy; electrical loss excluded")
save(fig, "paper_efficiency_vs_duty_factor_and_speed")

fig, axes = plt.subplots(1, 2, figsize=(12, 4), squeeze=False)
for ax, gait in zip(axes[0], ("trot", "walk")):
    subset = energy_valid[energy_valid.gait == gait]
    if subset.empty:
        invalid_message(ax)
        continue
    grouped = subset.groupby(["step_width", "command_df"], as_index=False)[
        energy_metric].median()
    for duty, group in grouped.groupby("command_df"):
        ax.plot(group.step_width, group[energy_metric], "o-",
                label=f"DF {duty:.3f}")
    ax.axvline(.30, color="black", ls=":", alpha=.5,
               label="Nominal comparison")
    ax.set(title=gait.title(), xlabel="Commanded full stance width (m)",
           ylabel="Positive mechanical CoT (lower is better)")
    ax.legend()
    scope_note(
        ax, f"Valid cells only: {len(subset)}/{int((energy_table.gait == gait).sum())}\n"
        "Positive mechanical proxy; electrical loss excluded")
save(fig, "paper_efficiency_vs_stance_width")

fig, ax = plt.subplots(figsize=(6, 5))
paired = valid.merge(
    energy_valid,
    on=["speed", "command_df", "step_width", "period", "gait"],
    how="inner", suffixes=("_chi", "_energy"))
if paired.empty:
    invalid_message(ax)
else:
    for gait, group in paired.groupby("gait"):
        ax.scatter(group[metric], group[energy_metric], label=gait.title(),
                   alpha=.75)
    ax.set(xlabel=r"Orbital convergence $\chi_{orb}$ (lower is better)",
           ylabel="Positive mechanical CoT (lower is better)")
    ax.legend()
    scope_note(
        ax, f"Paired valid cells: {len(paired)}/{len(table)}\n"
        "Descriptive; widths >=0.10 m; no beam/push\n"
        "No paper high-speed crossover test")
save(fig, "paper_convergence_efficiency_relationship")

print(table.to_string(index=False))
print(energy_table.to_string(index=False))
