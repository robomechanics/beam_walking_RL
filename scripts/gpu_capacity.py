"""Read-only GPU/RAM ownership and capacity gate before Isaac Sim starts."""
import json
import os
import pwd
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _compute_processes(index):
    query = subprocess.run([
        "nvidia-smi", f"--id={index}",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True)
    processes = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        pid_text, name, memory = [value.strip() for value in line.split(",", 2)]
        pid = int(pid_text)
        try:
            stat = Path(f"/proc/{pid}/status").read_text()
            uid = int(next(
                row.split()[1] for row in stat.splitlines()
                if row.startswith("Uid:")))
            owner = pwd.getpwuid(uid).pw_name
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(
                b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, ProcessLookupError, KeyError):
            owner, command = "exited", ""
        processes.append({
            "pid": pid, "owner": owner, "process_name": name,
            "command": command, "used_gpu_memory_mib": int(memory),
        })
    return processes


def check_capacity(mode, num_envs, device="cuda:0", video=False):
    if not device.startswith("cuda"):
        raise RuntimeError(
            "This experiment requires the tested CUDA/Isaac Sim execution path")
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

    # A 1024-environment Isaac training process measured about 3.2 GiB VRAM.
    # Reserve 4 GiB through 1024 envs, then 2 MiB per additional env, plus
    # a separate renderer margin. This deliberately overestimates evaluation.
    required_mib = 4096 + max(0, num_envs - 1024) * 2
    if mode in ("train", "benchmark"):
        required_mib = max(required_mib, 6144)
    if video:
        required_mib += 2048
    required_system_mib = 8192 + num_envs * 2 + (2048 if video else 0)
    compute_processes = _compute_processes(index)
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checker_pid": os.getpid(),
        "gpu_index": index, "gpu": name,
        "total_mib": int(total), "used_mib": int(used),
        "free_mib": int(free),
        "utilization_percent": int(utilization),
        "temperature_c": int(temperature),
        "required_free_mib": required_mib,
        "available_system_mib": available_system_mib,
        "required_system_mib": required_system_mib,
        "compute_processes": compute_processes,
        "exclusive_compute_device": len(compute_processes) == 0,
    }
    print("GPU_CAPACITY", json.dumps(record), flush=True)
    if compute_processes:
        raise RuntimeError(
            "GPU already has active compute processes; no process was stopped: "
            f"{compute_processes}")
    if int(free) < required_mib or available_system_mib < required_system_mib:
        raise RuntimeError(
            f"Insufficient free memory to start {mode}; resource check: {record}")
    return record
