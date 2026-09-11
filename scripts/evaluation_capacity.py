"""Evaluation-only capacity gate with the user's narrow RustDesk exception."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import subprocess

from gpu_capacity import _compute_processes

RUSTDESK_PROCESS_LIMIT_MIB = 512
RUSTDESK_TOTAL_LIMIT_MIB = 512


def classify_compute_processes(processes, current_user):
    """Allow only the current user's small RustDesk CUDA context."""
    allowed, blocked = [], []
    for process in processes:
        name = Path(process.get("process_name", "")).name.lower()
        command = process.get("command", "").lower()
        is_rustdesk = name == "rustdesk" and "rustdesk" in command
        owned = process.get("owner") == current_user
        small = process.get("used_gpu_memory_mib", RUSTDESK_PROCESS_LIMIT_MIB + 1) <= RUSTDESK_PROCESS_LIMIT_MIB
        (allowed if is_rustdesk and owned and small else blocked).append(process)
    if sum(item["used_gpu_memory_mib"] for item in allowed) > RUSTDESK_TOTAL_LIMIT_MIB:
        blocked.extend(allowed)
        allowed = []
    return allowed, blocked


def check_evaluation_capacity(num_envs, device="cuda:0", video=False):
    if not device.startswith("cuda"):
        raise RuntimeError("This evaluation requires CUDA/Isaac Sim")
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    index = int(device.split(":")[1]) if ":" in device else 0
    query = subprocess.run([
        "nvidia-smi", f"--id={index}",
        "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    name, total, used, free, utilization, temperature = [
        value.strip() for value in query.stdout.strip().split(",")]
    available_kib = next(
        int(line.split()[1])
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:"))
    available_system_mib = available_kib // 1024
    required_mib = 4096 + max(0, num_envs - 1024) * 2 + (2048 if video else 0)
    required_system_mib = 8192 + num_envs * 2 + (2048 if video else 0)
    processes = _compute_processes(index)
    current_user = pwd.getpwuid(os.getuid()).pw_name
    allowed, blocked = classify_compute_processes(processes, current_user)
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checker_pid": os.getpid(), "gpu_index": index, "gpu": name,
        "total_mib": int(total), "used_mib": int(used), "free_mib": int(free),
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
        "required_free_mib": required_mib,
        "available_system_mib": available_system_mib,
        "required_system_mib": required_system_mib,
        "compute_processes": processes,
        "allowed_rustdesk_processes": allowed,
        "blocked_compute_processes": blocked,
        "exclusive_compute_device": len(processes) == 0,
        "rustdesk_exception_applied": bool(allowed),
        "rustdesk_process_limit_mib": RUSTDESK_PROCESS_LIMIT_MIB,
        "rustdesk_total_limit_mib": RUSTDESK_TOTAL_LIMIT_MIB,
    }
    print("GPU_CAPACITY", json.dumps(record), flush=True)
    if blocked:
        raise RuntimeError(
            "GPU has a compute process outside the narrow RustDesk exception; "
            f"no process was stopped: {blocked}")
    if int(free) < required_mib or available_system_mib < required_system_mib:
        raise RuntimeError(
            f"Insufficient free memory to start evaluation: {record}")
    return record
