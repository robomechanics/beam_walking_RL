#!/bin/bash
# Narrow-stance pipeline, stance width 0.05-0.30 m, self-collision on, no domain
# randomization. The reported controller is push-trained, for realism.
# Serialized: one GPU job at a time.
#   1  train clean trot and walk specialists (the starting point)
#   2  push fine-tune each, warm started from its clean parent
#   3  command fidelity of the push-trained policies at every period
#   4  duty-contrast gate over all fidelity runs (stops the pipeline on failure)
#   5  period sweeps, both gaits
#   6  selector fit, validation per gait, promotion
#   7  figures: per-period surfaces, per-period trends, selector, fidelity
#   8  export CSVs and figures into paper_data/specialist_narrow_20260914
# Documented in PIPELINE.md at the repo root. Resumable: completed stages are
# skipped. Run from anywhere:
#   setsid nohup bash scripts/pipelines/specialist_pipeline.sh > results/pipeline.log 2>&1 &
cd "$(dirname "$0")/../.."
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
clean() { echo results/narrow_specialist_$1_seed5_3072_${TAG}/model_1799.pt; }
FINAL=model_1199.pt   # push fine-tune segments are 1,200 updates each
seg1() { echo results/narrow_specialist_push_$1_from_seed5_3072_${TAG}; }
seg2() { echo results/narrow_specialist_push_$1_seg2_3072_${TAG}; }
ck() { echo $(seg2 $1)/$FINAL; }
periods_of() { [ "$1" = walk ] && echo ".40 .48 .54" || echo ".36 .48 .54"; }
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

# 1. Clean training.
for g in trot walk; do
  [ -f "$(clean $g)" ] && continue
  fresh results/narrow_specialist_${g}_smoke_${TAG} results/narrow_specialist_${g}_seed5_3072_${TAG}
  run smoke_$g $P scripts/narrow_specialist_experiment.py smoke --gait $g --num_envs 64 --steps 96 --headless \
    --output results/narrow_specialist_${g}_smoke_${TAG} || exit 1
  wait_for_ram
  run train_$g $P scripts/narrow_specialist_experiment.py train --gait $g --headless \
    --output results/narrow_specialist_${g}_seed5_3072_${TAG} || exit 1
done

# 2. Push fine-tunes: two consecutive 1,200-update segments per gait (2,400
#    updates). Narrow stance under pushes learns slowly: after one segment a
#    third of trot episodes still ended in a fall and reward was still rising.
for g in trot walk; do
  if [ ! -f "$(seg1 $g)/$FINAL" ]; then
    fresh results/narrow_specialist_push_${g}_smoke_${TAG} "$(seg1 $g)"
    run push_smoke_$g $P scripts/narrow_specialist_push_experiment.py smoke --gait $g --perturbation \
      --num_envs 64 --steps 96 --headless --output results/narrow_specialist_push_${g}_smoke_${TAG} || exit 1
    wait_for_ram
    run push_train_$g $P scripts/narrow_specialist_push_experiment.py train --gait $g --perturbation \
      --initialize_from "$(clean $g)" --headless --output "$(seg1 $g)" || exit 1
  fi
  if [ ! -f "$(ck $g)" ]; then
    fresh "$(seg2 $g)"
    wait_for_ram
    run push_train2_$g $P scripts/narrow_specialist_push_experiment.py train --gait $g --perturbation \
      --initialize_from "$(seg1 $g)/$FINAL" --headless --output "$(seg2 $g)" || exit 1
  fi
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

# 3. Fidelity of the push-trained policies at every period.
fid() { # gait period
  local g=$1 p=$2
  local out; out=$(printf "results/validation_narrow_%s_v030_p%.2f_%s" "$g" "$p" "$TAG")
  local dfs=""; [ "$g" = trot ] && [ "$p" = ".36" ] && dfs="--dfs .50 .625"
  [ -f "$out/analysis_done" ] && return 0
  attempt "$out/evaluation_complete.json" "fidelity_${g}_p${p}" \
    $P scripts/evaluate_policy.py --task narrow_specialist --checkpoint "$(ck $g)" \
    --gait $g --period $p --step_widths $W $dfs --output "$out" --headless || return 1
  run "analyze_${g}_p${p}" $P scripts/analyze_beam.py "$out" && touch "$out/analysis_done"
}
( for p in $(periods_of trot); do fid trot $p; done ) & A=$!
sleep 45
( for p in $(periods_of walk); do fid walk $p; done ) & B=$!
wait $A $B

# 4. Duty-contrast gate. Push training is expected to shift duty somewhat;
#    the gate stops the pipeline only if the duty contrast itself collapses.
GATE_RUNS=""
for d in results/validation_narrow_*_${TAG}; do GATE_RUNS="$GATE_RUNS $(basename $d | sed "s/validation_narrow_//;s/_${TAG}//")=$d"; done
if ! run fidelity_gate $P scripts/narrow_fidelity_gate.py --criterion contrast --runs $GATE_RUNS \
    --output results/${TAG}_fidelity_gate.json; then
  echo "pipeline_stopped_at_fidelity_gate $(date -u +%FT%TZ)" >> $S
  exit 3
fi

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

# 6. Selector: fit, then both validations side by side, then promotion.
ST=results/narrow_surfaces_sweep_trot_${TAG}/grid
SW=results/narrow_surfaces_sweep_walk_${TAG}/grid
FIT=results/narrow_selector_${TAG}
if [ -f $ST/SURFACE_COMPLETE ] && [ -f $SW/SURFACE_COMPLETE ]; then
  [ -f $FIT/unified_duty_selector.pt ] || { rm -rf $FIT; run selector_fit $P scripts/fit_unified_duty_selector.py \
    $ST/surface_trials.csv $SW/surface_trials.csv --output $FIT; }
  if [ -f $FIT/unified_duty_selector.pt ]; then
    validate() { local g=$1
      attempt "results/narrow_selector_validation_${g}_${TAG}/SURFACE_COMPLETE" "selector_validation_$g" \
        $P scripts/collect_policy_surfaces.py --task narrow_specialist --checkpoint "$(ck $g)" \
        --gaits $g --selector-checkpoint $FIT/unified_duty_selector.pt \
        --output results/narrow_selector_validation_${g}_${TAG} --headless
    }
    validate trot & A=$!
    sleep 45
    validate walk & B=$!
    wait $A $B
    run selector_promote $P scripts/validate_unified_selector.py --selector $FIT/unified_duty_selector.pt \
      --validation results/narrow_selector_validation_trot_${TAG} results/narrow_selector_validation_walk_${TAG} \
      --output results/narrow_selector_promoted_${TAG}
  fi
fi

# 7. No push study: dropped at the mentor's direction. The push fine-tunes
#    above remain, as the realistic controller.
#    Figures: every period gets its own panel or file; nothing is pinned to 0.48 s.
FIG=PAPER_GRAPHS/narrow_specialist
mkdir -p $FIG
if [ -f $ST/SURFACE_COMPLETE ] && [ -f $SW/SURFACE_COMPLETE ]; then
  run figures_period_surfaces $P scripts/make_specialist_period_surfaces.py --input $ST $SW \
    --output $FIG/period_surfaces --title "Narrow-stance specialists"
  for p in .36 .40 .48 .54; do
    run figures_trends_p$p $P scripts/make_specialist_trend_figures.py --input $ST $SW --period $p \
      --output $FIG/trends/p$p
  done
fi
mkdir -p $FIG/selector $FIG/fidelity
cp $FIT/*.png $FIT/*.pdf $FIG/selector/ 2>/dev/null
cp results/narrow_selector_promoted_${TAG}/*.png results/narrow_selector_promoted_${TAG}/*.pdf $FIG/selector/ 2>/dev/null
for d in results/validation_narrow_*_${TAG}; do
  n=$(basename $d | sed "s/validation_narrow_//;s/_${TAG}//"); mkdir -p $FIG/fidelity/$n
  cp $d/*.png $d/*.pdf $FIG/fidelity/$n/ 2>/dev/null
done
echo "figures_done $(date -u +%FT%TZ)" >> $S

# 8. CSV export.
OUT=paper_data/specialist_narrow_20260914
mkdir -p $OUT/trials $OUT/policies $OUT/docs
for g in trot walk; do
  cp results/narrow_surfaces_sweep_${g}_${TAG}/grid/surface_trials.csv $OUT/trials/grid_speed_width_duty_BY_PERIOD_${g}.csv 2>/dev/null
  cp results/narrow_selector_validation_${g}_${TAG}/surface_trials.csv $OUT/trials/selector_validation_${g}.csv 2>/dev/null
done
for d in results/validation_narrow_*_${TAG}; do
  n=$(basename $d | sed "s/validation_narrow_//;s/_${TAG}//")
  cp $d/trials.csv $OUT/trials/fidelity_${n}.csv 2>/dev/null; cp $d/summary.csv $OUT/trials/fidelity_${n}_summary.csv 2>/dev/null
done
for d in results/narrow_specialist_*_3072_${TAG}; do
  n=$(basename $d); mkdir -p $OUT/policies/$n
  cp $d/provenance.json $d/agent.yaml $d/env.yaml $OUT/policies/$n/ 2>/dev/null
  for m in $d/model_1799.pt $d/model_1199.pt; do
    [ -f $m ] && sha256sum $m | sed "s|$d/||" > $OUT/policies/$n/$(basename $m .pt).sha256
  done
done
[ -d $FIT ] && cp -r $FIT $OUT/selector_fit
[ -d results/narrow_selector_promoted_${TAG} ] && cp -r results/narrow_selector_promoted_${TAG} $OUT/selector_promoted
cp results/${TAG}_fidelity_gate.json $OUT/docs/ 2>/dev/null
rm -rf $OUT/figures; cp -r $FIG $OUT/figures
echo "narrow_pipeline_done $(date -u +%FT%TZ)" >> $S
echo "=== NARROW PIPELINE DONE $(date -u +%FT%TZ)"
