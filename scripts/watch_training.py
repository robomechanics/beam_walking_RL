"""Read-only CPU monitoring of PPO logs; reward plateaus are not convergence."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
LIMITATION = ("A reward plateau cannot establish convergence without held-out gait-command "
              "compliance and locomotion evaluation. This watcher does not tune rewards or modify training.")
SCHEMA = "ppo-training-watch-v1"


def load_json_progress(path):
    """JSONL records: {iteration: integer, scalars: {TensorBoard-tag: number}}.

    Ignore an unfinished final line while a writer is appending. Reject malformed
    completed records rather than silently presenting a partial history as current.
    """
    path = Path(path)
    if not path.exists():
        return {}, ["Progress file does not exist yet"]
    lines = path.read_text().splitlines(keepends=True)
    series = {}
    warnings = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                warnings.append("Ignored an unfinished final JSONL record")
                continue
            raise ValueError(f"Malformed progress JSON at line {index + 1}") from None
        step = record.get("iteration")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"Invalid iteration at line {index + 1}")
        scalars = record.get("scalars")
        if not isinstance(scalars, dict):
            raise ValueError(f"Missing scalar mapping at line {index + 1}")
        for tag, value in scalars.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Non-numeric scalar {tag} at line {index + 1}")
            if not math.isfinite(value):
                warnings.append(f"Non-finite scalar {tag} at iteration {step}")
                continue
            series.setdefault(tag, {})[step] = float(value)
    return {tag: sorted(values.items()) for tag, values in series.items()}, warnings


def load_tensorboard(directory):
    """Read event scalars lazily without importing torch or a simulator."""
    files = sorted(Path(directory).glob("events.out.tfevents.*"))
    if not files:
        return {}, ["No TensorBoard event files found yet"]
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("Install/use a Python environment with tensorboard, or supply --progress-json") from error
    series = {}
    warnings = []
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                if not math.isfinite(event.value):
                    warnings.append(f"Non-finite scalar {tag} at iteration {event.step}")
                    continue
                values = series.setdefault(tag, {})
                previous = values.get(event.step)
                if previous is None or event.wall_time >= previous[0]:
                    values[event.step] = (event.wall_time, float(event.value))
    return {tag: sorted((step, value[1]) for step, value in values.items())
            for tag, values in series.items()}, warnings


def summarize_series(points, window=20, tolerance=.02):
    """Compare disjoint recent/prior windows of logged updates, never timesteps."""
    if window < 2 or tolerance <= 0:
        raise ValueError("window must be >=2 and tolerance positive")
    if not points:
        return {"samples": 0, "plateau_indicated": None}
    points = sorted(dict(points).items())
    recent = points[-window:]
    result = {"samples": len(points), "latest_iteration": points[-1][0],
              "latest": points[-1][1], "recent_mean": statistics.mean(v for _, v in recent),
              "recent_std": statistics.pstdev(v for _, v in recent),
              "recent_iterations": [recent[0][0], recent[-1][0]],
              "plateau_indicated": None}
    if len(points) < 2 * window:
        result["reason"] = f"Need {2 * window} logged values for two complete windows"
        return result
    prior = points[-2 * window:-window]
    prior_mean = statistics.mean(v for _, v in prior)
    scale = max(abs(prior_mean), abs(result["recent_mean"]), 1.)
    delta = result["recent_mean"] - prior_mean
    xmean = statistics.mean(x for x, _ in recent)
    denominator = sum((x - xmean) ** 2 for x, _ in recent)
    slope = sum((x - xmean) * (v - result["recent_mean"]) for x, v in recent) / denominator
    drift = slope * (recent[-1][0] - recent[0][0])
    result.update(prior_mean=prior_mean, prior_iterations=[prior[0][0], prior[-1][0]],
                  mean_change=delta, relative_mean_change=delta / scale,
                  recent_fitted_change=drift,
                  plateau_indicated=(abs(delta) <= tolerance * scale
                                     and abs(drift) <= tolerance * scale
                                     and result["recent_std"] <= .1 * scale))
    return result


def metric_group(tag):
    lower = tag.lower()
    gait_tag = lower.removeprefix("episode/")
    if gait_tag in {f"gait/{name}" for name in ["contact_accuracy", "stance_recall", "swing_recall",
                                               "foot_lateral_rmse", "body_foot_lateral_rmse", "heading_rmse",
                                               "forward_speed", "speed_mae", "lateral_rmse",
                                               "body_lateral_velocity_rmse", "body_yaw_rate_rmse"]}:
        return "gait"
    if lower in {"train/mean_reward", "train/mean_reward/time", "mean_reward"}:
        return "reward"
    if lower in {"train/mean_episode_length", "train/mean_episode_length/time", "mean_episode_length"}:
        return "episode_length"
    if "termination" in lower:
        if "time_out" in lower or "timeout" in lower:
            return "timeout"
        if "crossing" in lower or "success" in lower:
            return "crossing"
        return "failure"
    return None


def inspect_process(pid=None, log_file=None):
    result = {"pid": pid, "process_state": "not_checked", "exit_code": None, "error_lines": []}
    if pid is not None:
        if pid <= 0:
            raise ValueError("pid must be positive")
        try:
            os.kill(pid, 0)
            result["process_state"] = "present"
        except ProcessLookupError:
            result["process_state"] = "exited_or_missing"
        except PermissionError:
            result["process_state"] = "present_permission_restricted"
        # A zombie still responds to kill(pid, 0), but is no longer training.
        status = Path(f"/proc/{pid}/stat")
        try:
            state = status.read_text().rsplit(")", 1)[1].split()[0]
            if state == "Z":
                result["process_state"] = "exited_zombie"
        except (OSError, IndexError):
            pass
    if log_file is not None:
        log_file = Path(log_file)
        if log_file.exists():
            with log_file.open("rb") as stream:
                stream.seek(max(0, log_file.stat().st_size - 65536))
                tail = stream.read().decode(errors="replace")
            pattern = re.compile(r"Traceback \(most recent call last\)|CUDA out of memory|"
                                 r"RuntimeError:|Error executing job|Segmentation fault|Fatal Python error")
            result["error_lines"] = [line for line in tail.splitlines() if pattern.search(line)][-10:]
            result["log_modified_unix"] = log_file.stat().st_mtime
        else:
            result["log_status"] = "not_found_yet"
    result["note"] = "Process presence and log errors do not establish successful completion; exit status is unavailable."
    return result


def build_report(series, warnings=(), window=20, tolerance=.02, exploratory=False):
    metrics = {group: {} for group in ["reward", "episode_length", "failure", "timeout", "crossing", "gait"]}
    for tag, points in series.items():
        group = metric_group(tag)
        if group:
            metrics[group][tag] = summarize_series(points, window, tolerance)
    # Only iteration-indexed tags support an update count. */time is wall-clock indexed.
    reward = metrics["reward"].get("Train/mean_reward", metrics["reward"].get("mean_reward"))
    iterations = [points[-1][0] for tag, points in series.items() if points and not tag.lower().endswith("/time")]
    return {"schema": SCHEMA, "generated_unix": time.time(),
            "classification": "exploratory_only" if exploratory else "training_diagnostic_not_final_evaluation",
            "latest_logged_iteration": max(iterations) if iterations else None,
            "window_logged_updates": window, "relative_plateau_tolerance": tolerance,
            "reward_plateau_indicated": reward.get("plateau_indicated") if reward else None,
            "metrics": metrics, "unavailable_metric_groups": [k for k, v in metrics.items() if not v],
            "warnings": list(warnings), "limitation": LIMITATION,
            "gait_metric_note": "Gait scalars are training-batch diagnostics, not held-out complete-cycle command compliance."}


class ReportSchedule:
    """Monotonic cadence: initial, hourly (by default), and exactly one final.

    Missed deadlines yield one current report, never fabricated historical reports.
    Even an already-exited PID gets an initial report followed by a final report.
    """
    def __init__(self, interval=3600.):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("report interval must be finite and positive")
        self.interval = interval
        self.next_due = None
        self.finished = False

    def next_kind(self, now, stopping=False, process_state="present"):
        if self.finished:
            return None
        if self.next_due is None:
            self.next_due = now + self.interval
            return "initial"
        if stopping or process_state in {"exited_or_missing", "exited_zombie"}:
            self.finished = True
            return "final"
        if now >= self.next_due:
            self.next_due += (math.floor((now - self.next_due) / self.interval) + 1) * self.interval
            return "periodic"
        return None


def save_report(report, output):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.suffix != ".json":
        raise ValueError("Report output must be a .json path within this repository")
    if any(part in {".git", ".codex", ".agents"} for part in output.relative_to(ROOT).parts):
        raise ValueError("Report output cannot be inside repository metadata")
    if output.exists():
        try:
            existing = json.loads(output.read_text())
        except (ValueError, OSError):
            raise ValueError("Refusing to replace an existing non-watcher file") from None
        if not isinstance(existing, dict) or existing.get("schema") != SCHEMA:
            raise ValueError("Refusing to replace an existing non-watcher file")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".watch-", delete=False) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, output)


def publish_report(report, output, archive=False):
    """Archive scheduled snapshots beside latest; never replace a training log."""
    if archive:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(report["generated_unix"]))
        output_path = Path(output)
        archive_path = output_path.with_name(f"{output_path.stem}_{timestamp}_{time.time_ns()}_{report['report_kind']}.json")
        save_report(report, archive_path)
        report["archive"] = str(archive_path.resolve())
    save_report(report, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Directory containing TensorBoard event files")
    source.add_argument("--progress-json", type=Path, help="Append-only JSONL progress file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=.02)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Poll until interrupted; default is one report")
    parser.add_argument("--interval", type=float, default=20., help="Polling seconds, 1 through 30")
    parser.add_argument("--report_interval", "--report-interval", type=float, default=3600.,
                        help="Seconds between full reports during --watch (default: hourly)")
    parser.add_argument("--final-report", action="store_true", help="Label a one-shot fallback report final and archive it")
    args = parser.parse_args(argv)
    if args.window < 2 or not 0 < args.tolerance or not math.isfinite(args.tolerance):
        parser.error("--window must be >=2 and --tolerance finite and positive")
    if not 1 <= args.interval <= 30:
        parser.error("--interval must be between 1 and 30 seconds")
    if not math.isfinite(args.report_interval) or args.report_interval <= 0:
        parser.error("--report_interval must be finite and positive")
    if args.watch and args.final_report:
        parser.error("--final-report is a one-shot fallback; do not combine it with --watch")
    source_path = (args.run or args.progress_json).resolve()
    exploratory = args.exploratory or "train_42" in source_path.parts
    schedule = ReportSchedule(args.report_interval)
    stop = threading.Event()
    stop_reason = [None]
    previous_handlers = {}

    def request_stop(signum, _frame):
        stop_reason[0] = signal.Signals(signum).name
        stop.set()

    if args.watch:
        for signum in [signal.SIGTERM, signal.SIGINT]:
            previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        while True:
            process = inspect_process(args.pid, args.log_file)
            kind = schedule.next_kind(time.monotonic(), stop.is_set(), process["process_state"])
            if args.final_report:
                kind = "final"
            if kind is not None:
                try:
                    series, warnings = load_tensorboard(args.run) if args.run else load_json_progress(args.progress_json)
                except (OSError, ValueError, RuntimeError) as error:
                    series, warnings = {}, [f"Progress read failed: {type(error).__name__}: {error}"]
                report = build_report(series, warnings, args.window, args.tolerance, exploratory)
                report.update(source=str(source_path), process=process, report_kind=kind,
                              report_interval_seconds=args.report_interval,
                              final_reason=(stop_reason[0] or process["process_state"]) if kind == "final" else None)
                publish_report(report, args.output, archive=args.watch or args.final_report)
                print(json.dumps({"output": str(args.output.resolve()), "report_kind": kind,
                                  "iteration": report["latest_logged_iteration"],
                                  "reward_plateau_indicated": report["reward_plateau_indicated"],
                                  "classification": report["classification"], "limitation": LIMITATION}), flush=True)
            if not args.watch or kind == "final":
                break
            # Detect stop promptly and never sleep for an entire reporting interval.
            if process["process_state"] not in {"exited_or_missing", "exited_zombie"}:
                stop.wait(args.interval)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
