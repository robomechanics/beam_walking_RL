# Original-policy walking duty-factor extension

The user explicitly selected the same older policy that produced the existing
paper figures, not the separate adaptive policy. The checkpoint is
`results/paper_ppo_forcefix_nominalgains_seed2_3072_20260911/model_1799.pt`, SHA-256
`ef5dae663b9650a3197e589b406101d4a450282e1e358c2af895d0505912127a`.
No training or task-source change is part of this request.

## Measurement scope

This is exploratory generalization: walk DF 0.50 and 0.625 were absent from
this policy's training distribution. Keep the original paper-facing walk-at-0.75
restriction and old figures intact. New artifacts live in
`PAPER_GRAPHS/old_policy_walk_df/`. Do not pool adaptive or other checkpoint data.

- Gaits: trot and four-phase walking schedule.
- Duty factors: 0.50, 0.625, 0.75 for each gait.
- Speeds: 0.25, 0.30, 0.35, 0.40 m/s.
- Full body-frame stance widths: 0.10, 0.20, 0.30, 0.40, 0.50 m.
- Period: 0.48 s, 50 Hz control and 200 Hz physics.
- Flat ground, no pushes, nominal motor gains, deterministic inference.
- 120 physical cells, 32 matched independently seeded resets each (3,840 trials).
- 10% Bernoulli calibrated grounded reset mixture; report actual count.
- Grid seed block 1,310,000--1,310,031; smoke block 1,300,000--1,300,031.
- Twelve total cycles; discard first eight and measure last four complete cycles.

Record applied torque and left-endpoint joint velocity at 200 Hz, positive and
absolute mechanical CoT, complete force/contact histories, command fidelity,
straightness, physical failures, and five phase-zero augmented states. The
periodicity diagnostic is the maximum of four consecutive phase-zero differences,
normalized with existing state scales and reduced only by global x/y. Threshold
is 0.02 orbital RMS. No finite-difference forks or chi estimates are produced.

Physical failures are latched over all twelve cycles. Automatic failure/crossing
resets are disabled only to prevent episode splicing; failed trials remain in
every denominator and cannot yield valid energy or periodicity endpoints. An
unexpected reset or nonfinite instrument trace aborts collection.

An energy trial must pass command/straightness, 5 N schedule-centered topology,
positive-displacement energy validity, periodicity, and absence of any failure.
Each energy cell needs at least 90% valid trials. CoT aggregation takes the median
over four cycles, then valid trials. The aggregate speed/DF figure requires all
five widths valid before taking the median over their summaries. The companion
figure shows each width separately. Surface faces touching invalid vertices are
omitted. Periodicity includes every trial and is explicitly not chi.

## Reviewed execution

1. Independent science, physics and evaluation council review of actual new files.
2. Run `scripts/run_old_policy_surfaces.py` with the council-reviewed combined
   SHA-256; the runner refuses changed sources and new-file overwrites.
3. Wait for the pre-existing adaptive/deployment queue and fresh capacity. Never
   stop those jobs or start a duplicate deployment queue.
4. Six-cell smoke: both gaits and all three DFs at width/speed 0.30.
5. Require matching resets, exact phase/schedule/force chronology, expected array
   shapes, finite measurements and matching completion/archive hashes. Recompute
   smoke summaries from raw data. Trained trot DF 0.625/0.75 must retain at least
   90% compliance; unseen walk success is not a smoke gate.
6. Repeat GPU/process/RAM check, collect full grid, verify all archives and
   recompute every trial endpoint before plotting.

Full source snapshots, exact checkpoint/training identity, CSVs, NPZs, capacity
records, completion hashes and reviews are retained. Single-policy results do not
release paper claims; the unresolved chi estimator remains a separate issue.
