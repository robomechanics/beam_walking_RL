"""Read-only resource check before starting an Isaac Sim process."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def check_capacity(mode, num_envs, device="cuda:0", video=False):
    if not device.startswith("cuda"):
        raise RuntimeError("This experiment requires the tested CUDA/Isaac Sim execution path")
    index = int(device.split(":")[1]) if ":" in device else 0
    query = subprocess.run([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    name, total, used, free, utilization, temperature = [x.strip() for x in query.stdout.strip().split(",")]
    available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
    # The measured 1024-env training allocation was ~3.2 GiB. Reserve an additional
    # margin; this is a startup check, not a guarantee against later external load.
    required_mib = (4096 if mode in ["train", "benchmark"] else 2048) + (2048 if video else 0)
    if num_envs > 1024:
        required_mib += (num_envs - 1024) * 2
    record = {"checked_at": datetime.now(timezone.utc).isoformat(), "gpu": name,
        "total_mib": int(total), "used_mib": int(used), "free_mib": int(free),
        "utilization_percent": int(utilization), "temperature_c": int(temperature),
        "required_free_mib": required_mib, "available_system_mib": available_kib // 1024}
    print("GPU_CAPACITY", json.dumps(record), flush=True)
    if int(free) < required_mib or available_kib < 2 * 1024 * 1024:
        raise RuntimeError(f"Insufficient free memory to start {mode}; resource check: {record}")
    return record
