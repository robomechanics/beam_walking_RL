"""Read-only GPU/RAM ownership and capacity gate before Isaac Sim starts."""
import json
import os
import pwd
import subprocess
from datetime import datetime, timezone
from pathlib import Path

RUSTDESK_PROCESS_LIMIT_MIB = 512
RUSTDESK_TOTAL_LIMIT_MIB = 512


def classify_compute_processes(processes, current_user):
    """Allow only the current user's bounded RustDesk CUDA context."""
    allowed, blocked = [], []
    for process in processes:
        name = Path(process.get("process_name", "")).name.lower()
        command = process.get("command", "").lower()
        is_rustdesk = name == "rustdesk" and "rustdesk" in command
        owned = process.get("owner") == current_user
        small = (
            process.get("used_gpu_memory_mib", RUSTDESK_PROCESS_LIMIT_MIB + 1)
            <= RUSTDESK_PROCESS_LIMIT_MIB
        )
        (allowed if is_rustdesk and owned and small else blocked).append(process)
    if sum(item["used_gpu_memory_mib"] for item in allowed) > RUSTDESK_TOTAL_LIMIT_MIB:
        blocked.extend(allowed)
        allowed = []
    return allowed, blocked


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
    current_user = pwd.getpwuid(os.getuid()).pw_name
    if mode in ("smoke", "train", "benchmark"):
        allowed, blocked = classify_compute_processes(compute_processes, current_user)
    else:
        allowed, blocked = [], list(compute_processes)
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
        "allowed_rustdesk_processes": allowed,
        "blocked_compute_processes": blocked,
        "exclusive_compute_device": len(compute_processes) == 0,
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
            f"Insufficient free memory to start {mode}; resource check: {record}")
    return record
