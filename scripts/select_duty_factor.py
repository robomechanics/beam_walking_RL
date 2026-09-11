"""Query a trained flat-ground duty-factor selector."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/beam_walking"))

from beam_walking.experiment.duty_selector import (  # noqa: E402
    load_selector,
    predict_supported_rows,
)
from beam_walking.experiment.protocol import GAITS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--step-width", type=float, required=True)
    parser.add_argument("--speed", type=float, required=True)
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--gait", choices=GAITS, required=True)
    args = parser.parse_args()
    if not .10 <= args.step_width <= .50:
        parser.error("Step width must be in [0.10, 0.50] m")
    if not .25 <= args.speed <= .40:
        parser.error("Speed must be in [0.25, 0.40] m/s")
    ticks = round(args.period / .02)
    if ticks not in range(18, 28) or abs(ticks * .02 - args.period) > 1e-7:
        parser.error("Period must be 0.36--0.54 s in 0.02 s increments")
    model, checkpoint = load_selector(args.checkpoint)
    context = {
        "step_width": args.step_width, "speed": args.speed,
        "period": args.period, "gait": args.gait,
    }
    result = predict_supported_rows(model, checkpoint, [context])[0]
    result["schema"] = checkpoint["schema"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
