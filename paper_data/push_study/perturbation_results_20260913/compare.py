"""Compare nominal vs perturbed evaluation outcomes for the old and new policies."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/rml2/Documents/thomas_practice/beam_walking_RL/results")
RUNS = {
    ("current policy", "nominal"): "validation_tight3_nominal_20260913",
    ("current policy", "perturbed"): "validation_tight3_perturbed_20260913",
    ("perturbation-trained", "nominal"): "validation_tight3perturb_nominal_20260913",
    ("perturbation-trained", "perturbed"): "validation_tight3perturb_perturbed_20260913",
}

def outcomes(path):
    with np.load(path) as d:
        valid, failure, success = d["valid"].astype(bool), d["failure"].astype(bool), d["success"].astype(bool)
        env = d["perturbation_envelope"] if "perturbation_envelope" in d else None
        body = d["body"]
    trials = valid.shape[1]
    last = np.array([np.flatnonzero(valid[:, i])[-1] for i in range(trials)])
    idx = (last, np.arange(trials))
    failed = failure[idx]; succeeded = success[idx] & ~failed
    dist = body[idx][:, 0] + .65  # forward travel from reset
    out = {"trials": trials, "success": int(succeeded.sum()), "failure": int(failed.sum()),
           "timeout": int(trials - failed.sum() - succeeded.sum()),
           "mean_travel_m": float(dist.mean()),
           "mean_travel_at_failure_m": float(dist[failed].mean()) if failed.any() else None,
           "mean_time_at_failure_s": float((last[failed] * .02).mean()) if failed.any() else None}
    if env is not None and failed.any():
        out["mean_envelope_at_failure"] = float(env[idx][failed].mean())
    return out

rows = {}
for (policy, cond), run in RUNS.items():
    d = ROOT / run
    if not (d / "evaluation_complete.json").exists():
        continue
    for f in sorted(d.glob("trial_*.npz")):
        parts = f.stem.split("_")
        w, df = float(parts[1][1:]), float(parts[2][1:])
        rows[(policy, cond, w, df)] = outcomes(f)

json.dump({" | ".join(map(str, k)): v for k, v in rows.items()},
          open(ROOT / "perturbation_results_20260913/outcomes.json", "w"), indent=1)

lines = ["| policy | condition | width | DF | success | failure | timeout | travel m | fail time s |", "|---|---|---|---|---|---|---|---|---|"]
for (policy, cond, w, df), o in rows.items():
    ft = "" if o["mean_time_at_failure_s"] is None else f"{o['mean_time_at_failure_s']:.1f}"
    lines.append(f"| {policy} | {cond} | {w:.2f} | {df:.3f} | {o['success']} | {o['failure']} | {o['timeout']} | {o['mean_travel_m']:.2f} | {ft} |")
agg = {}
for (policy, cond, w, df), o in rows.items():
    a = agg.setdefault((policy, cond), {"success": 0, "trials": 0})
    a["success"] += o["success"]; a["trials"] += o["trials"]
lines += ["", "| policy | condition | success rate |", "|---|---|---|"]
for (policy, cond), a in agg.items():
    lines.append(f"| {policy} | {cond} | {a['success']}/{a['trials']} = {100*a['success']/a['trials']:.0f}% |")
(ROOT / "perturbation_results_20260913/RESULTS.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Figure: success rate per width, one line per (policy, condition).
widths = sorted({k[2] for k in rows})
fig, ax = plt.subplots(figsize=(6.4, 4))
styles = {("current policy", "nominal"): ("tab:blue", "--"), ("current policy", "perturbed"): ("tab:blue", "-"),
          ("perturbation-trained", "nominal"): ("tab:orange", "--"), ("perturbation-trained", "perturbed"): ("tab:orange", "-")}
for key, (color, ls) in styles.items():
    ys = []
    for w in widths:
        cells = [o for k, o in rows.items() if k[:2] == key and k[2] == w]
        ys.append(100 * sum(c["success"] for c in cells) / sum(c["trials"] for c in cells) if cells else np.nan)
    if not all(np.isnan(ys)):
        ax.plot(widths, ys, ls, color=color, marker="o", label=f"{key[0]}, {key[1]}")
ax.set_xlabel("commanded step width (m)"); ax.set_ylabel("success rate (%)"); ax.set_ylim(-2, 102)
ax.legend(frameon=True); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(ROOT / "perturbation_results_20260913/success_rate_by_width.png", dpi=160)
