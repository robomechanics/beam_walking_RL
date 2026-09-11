"""Aggregate matched paper effects across at least five independent PPO policies."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))
from beam_walking.experiment.stability import (
    analyze_policy_ensemble, evaluation_source_hash)

parser = argparse.ArgumentParser()
parser.add_argument("directories", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
import json
current_hash = evaluation_source_hash(ROOT)
for directory in args.directories:
    manifest = json.loads(
        (directory / "stability_manifest.json").read_text())
    if manifest.get("evaluation_sha256") != current_hash:
        raise ValueError(
            f"Current evaluator source differs from {directory}; "
            "use that run's source snapshot")
report = analyze_policy_ensemble(args.directories, args.output)
print(report)
