"""Analyze measured nominal flat-ground rollouts without launching Isaac Sim."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.analysis import (
    NOMINAL_SCHEMA, analyze, nominal_evaluation_source_hash,
)

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
manifest = json.loads((args.directory / "evaluation_manifest.json").read_text())
if (manifest.get("schema") == NOMINAL_SCHEMA
        and manifest.get("evaluator_sha256")
        != nominal_evaluation_source_hash(ROOT)):
    raise ValueError(
        "Current nominal evaluator differs from this run; use its source snapshot")
analyze(args.directory)
