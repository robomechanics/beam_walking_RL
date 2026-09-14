#!/bin/bash
set -e
cd /home/rml2/Documents/thomas_practice/beam_walking_RL
P=/home/rml2/anaconda3/envs/isaaclab/bin/python
OLD=results/narrow_beam_tight3_center10_from_tight2_3072_20260912/model_1799.pt
NEW_DIR=results/narrow_beam_tight3_perturb_from_tight3_3072_20260913
EVAL="--num_envs 64 --step_widths 0.05 0.06 0.08 0.10 --dfs 0.5 0.625 0.75 --gait trot --deployment_profile nominal --headless"
PERT="--perturbation"
stage() { echo "=== STAGE $1 $(date -u +%FT%TZ)"; }
stage smoke
$P scripts/beam_experiment.py smoke --num_envs 64 --steps 96 --headless $PERT --output results/perturbation_smoke_20260913
stage eval_old_perturbed
$P scripts/evaluate_policy.py --checkpoint $OLD --output results/validation_tight3_perturbed_20260913 $EVAL $PERT
stage eval_old_nominal
$P scripts/evaluate_policy.py --checkpoint $OLD --output results/validation_tight3_nominal_20260913 $EVAL
stage train_perturbed
$P scripts/beam_experiment.py train --deployment_dr --initialize_from $OLD --num_envs 3072 --iterations 1800 --seed 3 --headless --min_step_width 0.05 --width_tolerance 0.02 --lateral_heading_gain 1.0 --heading_cost_weight 3.0 --heading_cost_on_observation --centering_scale 0.05 --centering_weight 10.0 $PERT --output $NEW_DIR
stage eval_new_perturbed
$P scripts/evaluate_policy.py --checkpoint $NEW_DIR/model_1799.pt --output results/validation_tight3perturb_perturbed_20260913 $EVAL $PERT
stage eval_new_nominal
$P scripts/evaluate_policy.py --checkpoint $NEW_DIR/model_1799.pt --output results/validation_tight3perturb_nominal_20260913 $EVAL
stage done
