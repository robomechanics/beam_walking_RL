# Paper-aligned evaluation outputs

Run the stability collector only with a frozen checkpoint whose task hash and
training provenance match current source. A confirmatory policy must be a fresh,
parent-free 1,800-iteration run; evaluation accepts only its final requested
checkpoint at iteration 1,799. The five-policy ensemble must use the same frozen
training source and environment count, while checkpoint hashes, seeds, lineage
IDs, and archived provenance hashes must be distinct. The confirmatory grid is
frozen at 80 physical command cells: period 0.48 s; speeds 0.25, 0.30, 0.35,
and 0.40 m/s; full body-frame stance widths 0.10, 0.20, 0.30, 0.40, and 0.50 m;
trot duty factors 0.50, 0.625, and 0.75; and four-beat walk at duty factor
0.75. The three finite-difference sizes produce 240 return-map archives per
policy. It uses flat ground, no pushes, and nominal motor gains.

Before this grid is collected, every one of the 20 nominal command cells must
pass the frozen V4 readiness screen using 64 matched held-out validation seeds.
V4 uniquely matches each same-leg, same-direction contact transition to its
scheduled event within 20 ms and evaluates complete noncircular cycles. At the
policy's operational 5 N contact threshold, per-cell contact compliance must be
at least 0.80 and the mean per-trial complete-cycle topology fraction must be at
least 0.90. The 2 N and 10 N reconstructions are reported as threshold-sensitivity
diagnostics and do not gate readiness. This corrects the V3 fixed-cycle-bin
boundary artifact found during seed-2 validation; it does not change the policy,
commands, or training.

The seed-2 V4 rerun and any following seed-2 stability/CoT collection are
readiness and descriptive evidence only because the evaluator correction was
chosen after inspecting seed-2 validation traces. The V4 stability/CoT block at
seed 1,000,000 has also been inspected and is development data. V5 uses seed
1,100,000 for its pilot and reserves the untouched block beginning at 2,000,000
for a later frozen protocol. Paper-level inference requires the stated five
fresh independent PPO policies.

The collector checks GPU ownership, VRAM, and system RAM before Isaac Lab starts.
With eight references the frozen three-radius grid uses 2,328 environments:
eight simultaneous 291-member master stencils. Each master stencil contains a
baseline, two zero-offset controls, and all positive/negative 48-coordinate
perturbations. V5 copies one reference member's exposed phase-zero state,
previous action, and contact caches into every member immediately before the
fork, then applies only the requested tangent perturbation. It archives both the
requested state and immediate simulator readback and aborts on a normalized
error above 0.001. Pre-restore parallel-clone spread is diagnostic; zero-clone
output divergence remains the hidden-history noise gate.

The prior V4 collector used a different settled state for each member and
therefore produced invalid finite differences. Its 0/80 valid chi conditions
must not be interpreted as policy instability, and V4 archives are explicitly
ineligible for paper-claim release. Before another grid, V5 runs one development
condition with h=0.005, 0.01, 0.025, 0.05, and 0.10. Those radii diagnose hybrid
topology, zero-clone noise, and adjacent-h agreement. The extra radii receive no
frozen-gate pass. If the original trio is unsuitable, the diagnostic may inform
a separately versioned and preregistered protocol before the untouched seed
block is used. V5 claim release stays disabled, and no numeric threshold is
relaxed.

Regenerate tables and plots without the GPU with
scripts/analyze_stability.py. Aggregate at least five distinct PPO policies
with scripts/analyze_stability_ensemble.py.

| Artifact | Meaning |
|---|---|
| stability_manifest.json | Exact grid, metric, gates, identities, and capacity |
| condition NPZ files | Simultaneous stencils, zero controls, contacts, command and straightness traces |
| return_maps.npz | All 48D/46D augmented and 36D/34D physical matrices, spectra, and worst directions |
| return_map_index.json | Map keys and retained coordinate labels |
| stability_references.csv | Per-reference chi, translation residuals, straightness, and validity gates |
| stability_summary.csv | Perturbation-size condition summaries |
| finite_difference_pairs.csv | Adjacent-radius matrix and chi differences for every requested h |
| paper_conditions.csv | Preregistered h=0.05 condition table |
| paper_claims.json | Strictly gated single-policy descriptive contrasts |
| policy_effects.csv | Within-policy matched log-ratios across PPO seeds |
| ensemble_claims.json | Five-policy bootstrap directional/equivalence decisions |
| energy condition NPZ files | Four complete cycles of 200 Hz applied torque, joint velocity, and gait-validity traces |
| paper_energy_conditions.csv | Positive-mechanical-CoT results for all 80 physical cells |
| paper figures | Orbital correlations, CoT correlations, full-map comparison, and validity plots |

The preregistered outcome is 46D augmented orbital
chi_orb = sigma_max(Phi_orb), after removing neutral global x/y translation.
The unquotiented 48D value is diagnostic. The x/y identity-mode gate must pass
before any orbital result is valid.

The matched tests are: trot DF 0.75 over 0.50 should have a chi ratio below one;
width 0.10 over 0.30 should have a ratio above one; speed 0.40 over 0.25 and
walk over trot at DF 0.75 must have bootstrap 95% ratio intervals fully within
0.80–1.20. That margin is our preregistered operational definition, not a bound
from the paper. Report only whether qualitative alignment passed under the
translation-reduced learned-controller analogue. A paper trajectory optimizer
that “failed to converge” failed to produce a feasible nominal trajectory; that
is separate from PPO learning convergence and from the closed-loop chi measured
here.

The efficiency endpoint is positive mechanical cost of transport:

`CoT+ = sum(max(tau_applied * qdot_left, 0)) * 0.005 / (m * g * delta_x)`.

The clipped applied torque and left-endpoint joint velocity are sampled at
200 Hz over the final four complete phase-locked settling cycles. Robot mass,
gravity, actuator effort limits, joint order, integration rule, raw samples,
work, displacement, and per-cycle CoT are archived. Absolute-work CoT is a
diagnostic. No electrical-loss coefficient is invented, so this is a positive
mechanical proxy rather than the paper's exact electromechanical efficiency
measure. Energy validity uses command, straightness, contact topology,
termination, and positive-progress gates independently of finite-difference
numerical validity.

Across five independent PPO seeds, the preregistered energy tests are: the trot
log CoT ratio for DF 0.75 over 0.50 should increase from 0.25 to 0.40 m/s;
width 0.10 over 0.30 and walk over trot at matched DF 0.75 must have
policy-bootstrap 95% ratio intervals wholly inside 0.80--1.20. This speed range
tests a local interaction and cannot establish the paper's high-speed
crossover. Beam traversal and impulse rejection remain outside the flat-ground,
no-push experiment.
