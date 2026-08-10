"""Host/load gates and explicit system prerequisite checks."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_threshold() -> float:
    return max(4.0, (os.cpu_count() or 1) * 0.5)


def wait_for_load(
    *,
    timeout_seconds: int = 900,
    force_busy: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    threshold = load_threshold()
    started = time.monotonic()
    samples: list[dict[str, float]] = []
    while True:
        one_minute = float(os.getloadavg()[0]) if hasattr(os, "getloadavg") else 0.0
        samples.append({"elapsed_seconds": time.monotonic() - started, "one_minute": one_minute})
        if one_minute < threshold:
            return {"passed": True, "forced": False, "threshold": threshold, "samples": samples}
        elapsed = time.monotonic() - started
        if force_busy:
            return {"passed": False, "forced": True, "threshold": threshold, "samples": samples}
        if elapsed >= timeout_seconds:
            return {"passed": False, "forced": False, "threshold": threshold, "samples": samples}
        retry_seconds = min(10.0, timeout_seconds - elapsed)
        if progress is not None:
            progress(
                f"System load {one_minute:.2f} is above {threshold:.2f}; "
                f"retrying in {retry_seconds:.0f}s"
            )
        time.sleep(retry_seconds)


def host_record() -> dict[str, Any]:
    system = platform.system()
    machine = platform.machine().lower()
    memory = _physical_memory()
    official_platform = (system == "Darwin" and machine == "arm64") or (
        system == "Linux" and machine in {"x86_64", "amd64"}
    )
    return {
        "os": system,
        "os_release": platform.release(),
        "architecture": machine,
        "cpu": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory,
        "official_v1_platform": official_platform,
    }


def prerequisites() -> dict[str, Any]:
    programs = {}
    for executable in ("uv", "ffmpeg", "ffprobe"):
        path = shutil.which(executable)
        programs[executable] = {
            "available": path is not None,
            "version": _program_version(executable) if path else None,
        }
    return {"passed": all(item["available"] for item in programs.values()), "programs": programs}


def _program_version(executable: str) -> str:
    argument = "--version" if executable == "uv" else "-version"
    process = subprocess.run(
        [executable, argument],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return process.stdout.splitlines()[0] if process.stdout else "unknown"


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        process = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        value = process.stdout.strip()
        if value:
            return value
        process = subprocess.run(
            ["sysctl", "-n", "hw.model"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return process.stdout.strip() or "unknown"
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            if line.casefold().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _physical_memory() -> int | None:
    if platform.system() == "Darwin":
        process = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        try:
            return int(process.stdout.strip())
        except ValueError:
            return None
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
