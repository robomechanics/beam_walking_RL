"""Analyze measured nominal flat-ground rollouts without launching Isaac Sim."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.analysis import (
    NOMINAL_SCHEMA, analyze, nominal_evaluation_source_hash,
    nominal_study_readiness,
)

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
parser.add_argument("--study-with", type=Path)
args = parser.parse_args()
manifest = json.loads((args.directory / "evaluation_manifest.json").read_text())
if (manifest.get("schema") == NOMINAL_SCHEMA
        and manifest.get("evaluator_sha256")
        != nominal_evaluation_source_hash(ROOT)):
    raise ValueError(
        "Current nominal evaluator differs from this run; use its source snapshot")
analyze(args.directory)
if args.study_with is not None:
    other_manifest = json.loads(
        (args.study_with / "evaluation_manifest.json").read_text())
    if (other_manifest.get("schema") == NOMINAL_SCHEMA
            and other_manifest.get("evaluator_sha256")
            != nominal_evaluation_source_hash(ROOT)):
        raise ValueError(
            "Current nominal evaluator differs from paired run; use its source snapshot")
    analyze(args.study_with)
    report = nominal_study_readiness(args.directory, args.study_with)
    (args.directory / "study_readiness.json").write_text(
        json.dumps(report, indent=2))
    print("STUDY_READINESS", json.dumps(report))
