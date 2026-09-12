"""Query a validated high-duty walk selector."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.high_duty_walk import HIGH_DUTY_WALK_PERIOD  # noqa: E402
from beam_walking.experiment.high_duty_walk_selector import (  # noqa: E402
    load_selector,
    predict_supported_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--step-width", type=float, required=True)
    parser.add_argument("--speed", type=float, required=True)
    args = parser.parse_args()
    model, payload = load_selector(args.checkpoint)
    context = {"step_width": args.step_width, "speed": args.speed,
               "period": HIGH_DUTY_WALK_PERIOD, "gait": "walk"}
    result = predict_supported_rows(model, payload, [context])[0]
    result["schema"] = payload["schema"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
