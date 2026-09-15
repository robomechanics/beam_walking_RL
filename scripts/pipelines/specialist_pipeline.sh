#!/bin/bash
# Narrow-stance pipeline, stance width 0.05-0.30 m, self-collision on, no domain
# randomization. One trot and one walk specialist are the reported controllers.
#   1  train the trot and walk specialists
#   2  command fidelity at every period
#   3  duty-contrast check over all fidelity runs (recorded, never stops the run)
#   4  robustness: success under disturbances at every speed, period, width and duty
#   5  period sweeps, both gaits
#   6  duty-factor selector network from success under disturbance, fresh-seed
#      validation, promotion
#   7  paper figures
#   8  export CSVs and the paper figures into paper_data/specialist_<tag>
# Documented in PIPELINE.md at the repo root. Resumable: completed stages are
# skipped. Run from anywhere:
#   setsid nohup bash scripts/pipelines/specialist_pipeline.sh > results/pipeline.log 2>&1 &
# Set REPO_ROOT when running a copy of this script from another directory.
cd "${REPO_ROOT:-$(dirname "$0")/../..}" || exit 1
PYTHON_BIN=${PYTHON_BIN:-$HOME/anaconda3/envs/isaaclab/bin/python}
P="env -u PYTHONPATH -u AMENT_PREFIX_PATH $PYTHON_BIN"
TAG=${TAG:-narrow_20260914}
S=results/${TAG}.status
W=".05 .10 .15 .20 .25 .30"
: >> $S
while pgrep -f "scripts/(evaluate_policy|collect_policy_surfaces|unified_specialist|narrow_specialist)" >/dev/null; do sleep 20; done

run() { local name=$1; shift; echo "=== STAGE $name $(date -u +%FT%TZ)"
  if "$@"; then echo "$name ok $(date -u +%FT%TZ)" >> $S; return 0
  else echo "$name FAILED $(date -u +%FT%TZ)" >> $S; return 1; fi; }
wait_for_ram() { local waited=0
  while [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -lt 14800 ]; do
    sync; sleep 30; waited=$((waited+30)); [ $waited -ge 1200 ] && return 0; done; }
ck() { echo results/narrow_specialist_$1_seed5_3072_${TAG}/model_1799.pt; }
periods_of() { [ "$1" = walk ] && echo ".40 .48 .54" || echo ".36 .40 .48 .54"; }
SPEEDS=".25 .30 .35 .40"
v3() { awk -v s="$1" 'BEGIN { printf "%03d", s * 100 + .5 }'; }   # .30 -> 030
# Evaluated duty factors: trot every 0.05 from 0.50 (0.75 needs a period of at
# least 0.40 s for five swing ticks), walk 0.75-0.90.
dfs_of() { # gait period
  if [ "$1" = walk ]; then echo ".75 .80 .85 .90"
  elif [ "$2" = .36 ]; then echo ".50 .55 .60 .65 .70"
  else echo ".50 .55 .60 .65 .70 .75"; fi; }
# Empty a training output directory. A trainer stopped by a signal can leave its
# watcher alive, and the watcher's final report would land in the emptied
# directory and make the retry refuse to start. Stop that watcher first.
fresh() { local d waited
  for d; do
    pkill -f "watch_training.py --run $PWD/$d " 2>/dev/null
    waited=0
    while pgrep -f "watch_training.py --run $PWD/$d " >/dev/null && [ $waited -lt 60 ]; do
      sleep 1; waited=$((waited+1)); done
    rm -rf "$d"
  done; }

# 1. Training.
for g in trot walk; do
  [ -f "$(ck $g)" ] && continue
  fresh results/narrow_specialist_${g}_smoke_${TAG} results/narrow_specialist_${g}_seed5_3072_${TAG}
  run smoke_$g $P scripts/narrow_specialist_experiment.py smoke --gait $g --num_envs 64 --steps 96 --headless \
    --output results/narrow_specialist_${g}_smoke_${TAG} || exit 1
  wait_for_ram
  run train_$g $P scripts/narrow_specialist_experiment.py train --gait $g --headless \
    --output results/narrow_specialist_${g}_seed5_3072_${TAG} || exit 1
done

# Evaluations below run trot and walk side by side. Each is an independent
# process with its own seeds, so running two at once changes timing only.
# Before each start, wait for the host memory the evaluation check needs, and
# retry a job whose start is refused.
eval_mem() { local waited=0
  while [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -lt 9000 ]; do
    sleep 20; waited=$((waited+20)); [ $waited -ge 900 ] && return 0; done; }
attempt() { # completion-file stage-name command...
  local marker=$1 name=$2; shift 2
  local outdir; outdir=$(dirname "$marker")
  for try in 1 2 3; do
    [ -f "$marker" ] && return 0
    rm -rf "$outdir"; mkdir -p "$(dirname "$outdir")"; eval_mem
    run "$name" "$@" && [ -f "$marker" ] && return 0
    sleep 60
  done
  return 1
}
# Run job lines from stdin (each a function call) on two workers. A worker claims
# a job by creating its lock directory, so every job runs once.
pool() { # name
  local locks=results/.pool_${TAG}_$1 jobs a b
  jobs=$(cat)
  [ -z "$jobs" ] && return 0
  rm -rf "$locks"; mkdir -p "$locks"
  worker() { local line i=0
    while IFS= read -r line; do
      i=$((i+1))
      mkdir "$locks/$i" 2>/dev/null || continue
      eval "$line" < /dev/null   # a job must not read the remaining job lines
    done <<< "$jobs"; }
  worker & a=$!
  sleep 45
  worker & b=$!
  wait $a $b
  rm -rf "$locks"; }

# 2. Command fidelity at every period.
fid() { # gait period
  local g=$1 p=$2
  local out; out=$(printf "results/validation_narrow_%s_v030_p%.2f_%s" "$g" "$p" "$TAG")
  [ -f "$out/analysis_done" ] && return 0
  attempt "$out/evaluation_complete.json" "fidelity_${g}_p${p}" \
    $P scripts/evaluate_policy.py --task narrow_specialist --checkpoint "$(ck $g)" \
    --gait $g --period $p --step_widths $W --dfs $(dfs_of $g $p) --output "$out" --headless || return 1
  run "analyze_${g}_p${p}" $P scripts/analyze_beam.py "$out" && touch "$out/analysis_done"
}
( for p in $(periods_of trot); do fid trot $p; done ) & A=$!
sleep 45
( for p in $(periods_of walk); do fid walk $p; done ) & B=$!
wait $A $B

# 3. Duty-contrast check over all fidelity runs, recorded in
#    results/<tag>_fidelity_gate.json. The pipeline continues either way.
GATE_RUNS=""
for d in results/validation_narrow_*_${TAG}; do GATE_RUNS="$GATE_RUNS $(basename $d | sed "s/validation_narrow_//;s/_${TAG}//")=$d"; done
run fidelity_gate $P scripts/narrow_fidelity_gate.py --criterion contrast --runs $GATE_RUNS \
  --output results/${TAG}_fidelity_gate.json

# 4. Robustness: success under random base disturbances at every speed, period,
#    width and duty factor, on two workers. The selector's labels come from
#    these runs; figure 3 uses 0.30 m/s.
push_dir() { # gait speed period force
  local level=""; [ "$4" != 25 ] && level="_f$4"
  printf "results/push_robustness_narrow_%s_v%s_p%.2f%s_%s" "$1" "$(v3 $2)" "$3" "$level" "$TAG"; }
robustness_job() { # gait speed period force
  local out; out=$(push_dir $1 $2 $3 $4)
  attempt "$out/evaluation_complete.json" "push_robustness_${1}_v$(v3 $2)_p${3}_f${4}" \
    $P scripts/evaluate_policy.py --task narrow_specialist --checkpoint "$(ck $1)" \
    --gait $1 --speed $2 --period $3 --step_widths $W --dfs $(dfs_of $1 $3) --perturbation \
    --perturbation_max_force $4 --perturbation_max_torque $(python3 -c "print($4 * 3 / 25)") \
    --output "$out" --headless; }
for v in $SPEEDS; do for g in trot walk; do for p in $(periods_of $g); do
  echo "robustness_job $g $v $p 25"; done; done; done | pool robustness

# 4b. Where every duty factor succeeds in at least 95% of trials at 0.20-0.30 m
#     for every period at 0.30 m/s, repeat that gait's 0.30 m/s robustness with
#     disturbances up to 50 N (6 N m) so the wider stances are not at full success.
at_ceiling() { # gait
  python3 - "$1" $(for p in $(periods_of $1); do echo "$(push_dir $1 .30 $p 25)/perturbation_summary.json"; done) <<'PY'
import json, sys
rates = [c["success"] / c["trials"] for path in sys.argv[2:]
         for c in json.load(open(path))["conditions"].values() if c["step_width"] >= .195]
sys.exit(0 if rates and min(rates) >= .95 else 1)
PY
}
for g in trot walk; do
  at_ceiling $g && for p in $(periods_of $g); do echo "robustness_job $g .30 $p 50"; done
done | pool robustness_50n

# 5. Period sweeps, trot and walk side by side.
sweep() { local g=$1
  local out=results/narrow_surfaces_sweep_${g}_${TAG}/grid
  attempt "$out/SURFACE_COMPLETE" "surface_sweep_$g" \
    $P scripts/collect_policy_surfaces.py --task narrow_specialist --checkpoint "$(ck $g)" \
    --gaits $g --period-sweep --output "$out" --headless
}
sweep trot & A=$!
sleep 45
sweep walk & B=$!
wait $A $B

# 6. Duty-factor selector network. Labels come from success under disturbance at
#    every speed, period and width (stage 4). Every selection is run again under
#    the same disturbances on the held-out test split, then the selector is
#    promoted. A new fit discards the previous validation runs.
FIT=results/narrow_robust_selector_${TAG}
PROMOTED_DIR=results/narrow_robust_selector_promoted_${TAG}
if [ ! -f $FIT/robust_duty_selector.pt ]; then
  rm -rf $FIT $PROMOTED_DIR results/narrow_robust_selector_validation_*_${TAG}
  run selector_fit $P scripts/fit_robust_duty_selector.py --tag $TAG --output $FIT
fi
if [ -f $FIT/robust_duty_selector.pt ]; then
  selector_validation_job() { # gait speed period widths duties (comma-separated)
    local out; out=$(printf "results/narrow_robust_selector_validation_%s_v%s_p%.2f_%s" $1 "$(v3 $2)" $3 $TAG)
    attempt "$out/evaluation_complete.json" "selector_validation_${1}_v$(v3 $2)_p${3}" \
      $P scripts/evaluate_policy.py --task narrow_specialist --checkpoint "$(ck $1)" \
      --gait $1 --speed $2 --period $3 --step_widths ${4//,/ } --dfs ${5//,/ } --perturbation \
      --perturbation_max_force 25 --perturbation_max_torque 3 --split test --seed 1000000 \
      --output "$out" --headless; }
  $P scripts/validate_robust_duty_selector.py --selector $FIT/robust_duty_selector.pt --tag $TAG --print-jobs \
    | grep '^selector_validation_job ' | pool selector_validation
  [ -f $PROMOTED_DIR/validated_robust_duty_selector.pt ] || run selector_promote \
    $P scripts/validate_robust_duty_selector.py --selector $FIT/robust_duty_selector.pt --tag $TAG --output $PROMOTED_DIR
fi

# 7. Paper figures: every figure pools the periods at which every duty factor ran.
#    Nothing is pinned to 0.48 s.
FIG=PAPER_GRAPHS/narrow_specialist/paper_figures
run figures_paper $P scripts/make_narrow_paper_figures.py --tag $TAG --output $FIG
echo "figures_done $(date -u +%FT%TZ)" >> $S

# 8. Export.
OUT=paper_data/specialist_${TAG}
mkdir -p $OUT/trials $OUT/policies $OUT/docs
for g in trot walk; do
  cp results/narrow_surfaces_sweep_${g}_${TAG}/grid/surface_trials.csv $OUT/trials/grid_speed_width_duty_BY_PERIOD_${g}.csv 2>/dev/null
done
for d in results/validation_narrow_*_${TAG}; do
  n=$(basename $d | sed "s/validation_narrow_//;s/_${TAG}//")
  cp $d/trials.csv $OUT/trials/fidelity_${n}.csv 2>/dev/null; cp $d/summary.csv $OUT/trials/fidelity_${n}_summary.csv 2>/dev/null
done
for d in results/push_robustness_narrow_*_${TAG}; do
  n=$(basename $d | sed "s/push_robustness_narrow_//;s/_${TAG}//")
  cp $d/perturbation_summary.json $OUT/trials/robustness_${n}.json 2>/dev/null
done
for d in results/narrow_specialist_*_seed5_3072_${TAG}; do
  n=$(basename $d); mkdir -p $OUT/policies/$n
  cp $d/provenance.json $d/agent.yaml $d/env.yaml $OUT/policies/$n/ 2>/dev/null
  sha256sum $d/model_1799.pt | sed "s|$d/||" > $OUT/policies/$n/model_1799.sha256
done
# The selector fit (labels and selections behind figure 5), its fresh-seed
# validation runs, and the promotion report.
rm -rf $OUT/selector_fit $OUT/selector_promoted $OUT/trials/selector_validation_*
[ -f $FIT/robust_duty_selector.pt ] && cp -r $FIT $OUT/selector_fit
[ -d $PROMOTED_DIR ] && cp -r $PROMOTED_DIR $OUT/selector_promoted
for d in results/narrow_robust_selector_validation_*_${TAG}; do
  [ -f $d/perturbation_summary.json ] || continue
  n=$(basename $d | sed "s/narrow_robust_selector_validation_//;s/_${TAG}//")
  cp $d/perturbation_summary.json $OUT/trials/selector_validation_${n}.json
done
cp results/${TAG}_fidelity_gate.json $OUT/docs/ 2>/dev/null
rm -rf $OUT/figures; cp -r $FIG $OUT/figures
echo "narrow_pipeline_done $(date -u +%FT%TZ)" >> $S
echo "=== NARROW PIPELINE DONE $(date -u +%FT%TZ)"
