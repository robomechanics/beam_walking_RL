import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/rml2/Documents/thomas_practice/beam_walking_RL/results")
OUT = ROOT / "perturbation_results_20260913"
RUNS = {"current policy": "validation_tight3_perturbed_20260913",
        "perturbation-trained": "validation_tight3perturb_perturbed_20260913"}
COLORS = {"current policy": "#2f6db3", "perturbation-trained": "#d8792a"}
DT = .02

def load(run):
    cells = {}
    for f in sorted((ROOT / run).glob("trial_*.npz")):
        p = f.stem.split("_"); w, df = float(p[1][1:]), float(p[2][1:])
        with np.load(f) as d:
            cells[(w, df)] = {k: d[k] for k in ("valid", "failure", "success", "perturbation_envelope", "perturbation_force_w")}
    return cells

data = {name: load(run) for name, run in RUNS.items()}

# 1. Survival curve vs episode time (all cells pooled), with push envelope.
fig, ax = plt.subplots(figsize=(6.4, 4))
ax2 = ax.twinx()
tmax = 0
for name, cells in data.items():
    ends, fails = [], []
    for c in cells.values():
        valid, failure = c["valid"].astype(bool), c["failure"].astype(bool)
        for i in range(valid.shape[1]):
            last = np.flatnonzero(valid[:, i])[-1]
            ends.append(last); fails.append(failure[last, i])
    ends, fails = np.array(ends), np.array(fails)
    t = np.arange(0, ends.max() + 2) * DT; tmax = max(tmax, t[-1])
    alive = np.array([1 - ((ends < k) & fails).mean() for k in range(len(t))])
    ax.plot(t, 100 * alive, color=COLORS[name], lw=2, label=name)
    ax.text(t[-1], 100 * alive[-1] + 1.5, f"{100*alive[-1]:.0f}%", color=COLORS[name], ha="right", fontsize=9)
env = .1 + .9 * np.clip(np.arange(int(tmax / DT) + 1) * DT / 10., 0, 1)
ax2.fill_between(np.arange(len(env)) * DT, 0, 25 * env, color="#999999", alpha=.15, lw=0)
ax2.plot(np.arange(len(env)) * DT, 25 * env, color="#777777", lw=1, ls="--", label="push bound")
ax.set_xlabel("episode time (s)"); ax.set_ylabel("trials not fallen (%)"); ax.set_ylim(0, 105); ax.set_xlim(0, tmax)
ax2.set_ylabel("push force bound (N)"); ax2.set_ylim(0, 26.25)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", frameon=True); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "survival_vs_time.png", dpi=160)

# 2. Success rate by width, one panel per DF.
dfs = sorted({k[1] for k in data["current policy"]}); widths = sorted({k[0] for k in data["current policy"]})
fig, axes = plt.subplots(1, len(dfs), figsize=(9.6, 3.4), sharey=True)
for ax, df in zip(axes, dfs):
    for name, cells in data.items():
        ys = []
        for w in widths:
            c = cells[(w, df)]; valid = c["valid"].astype(bool)
            last = np.array([np.flatnonzero(valid[:, i])[-1] for i in range(valid.shape[1])])
            idx = (last, np.arange(valid.shape[1]))
            ys.append(100 * (c["success"][idx] & ~c["failure"][idx]).mean())
        ax.plot(widths, ys, "-o", color=COLORS[name], lw=2, label=name)
    ax.set_title(f"DF {df:g}"); ax.set_xlabel("step width (m)"); ax.set_ylim(-3, 103); ax.grid(alpha=.3)
axes[0].set_ylabel("success rate (%)"); axes[0].legend(frameon=True, fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "success_by_width_per_df.png", dpi=160)

# 3. Histogram of push force at the failure step.
fig, ax = plt.subplots(figsize=(6.4, 3.6))
bins = np.linspace(0, 25, 26)
for name, cells in data.items():
    forces = []
    for c in cells.values():
        valid, failure = c["valid"].astype(bool), c["failure"].astype(bool)
        f = np.linalg.norm(c["perturbation_force_w"], axis=-1)
        for i in range(valid.shape[1]):
            last = np.flatnonzero(valid[:, i])[-1]
            if failure[last, i]:
                forces.append(f[max(last - 5, 0):last + 1, i].max())
    ax.hist(forces, bins=bins, color=COLORS[name], alpha=.6, label=f"{name} ({len(forces)} falls)")
ax.set_xlabel("largest push in the 0.1 s before falling (N)"); ax.set_ylabel("falls"); ax.legend(frameon=True); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "push_force_at_failure.png", dpi=160)
print("ok")
