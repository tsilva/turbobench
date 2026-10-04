"""Offline invocation of one isolated provider runner."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from turbobench.model import ResolvedProvider
from turbobench.util import read_json, write_json

_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(?:/[A-Za-z0-9_.+@%:,=-]+){2,}")
RUNNER_TIMEOUT_SECONDS = 900


def _runner_timeout_seconds(request: dict[str, Any]) -> int:
    """Bound workers by requested work, keeping the existing small-job floor."""
    steps = len(request.get("actions") or [])
    if not steps:
        return RUNNER_TIMEOUT_SECONDS
    repetitions = int(request.get("repetitions", 1))
    warmup = int(request.get("warmup_steps", 0))
    snapshot = 2 * int(request.get("snapshot_suffix_steps", 0))
    constructor = (request.get("execution_spec") or {}).get("constructor") or {}
    frame_skip = int(request.get("frame_skip", constructor.get("frame_skip", 1)))
    frames = int(request.get("shape", 1)) * frame_skip * (
        steps * repetitions + warmup + snapshot
    )
    # A 64-lane, 3,000-decision, three-repeat job at skip 2 retains 15 minutes.
    # Larger captures/counts scale that finite budget; timing itself is unchanged.
    baseline_frames = 64 * 2 * (3000 * 3 + 500)
    return RUNNER_TIMEOUT_SECONDS * max(1, (frames + baseline_frames - 1) // baseline_frames)


def invoke_runner(
    provider: ResolvedProvider,
    request: dict[str, Any],
    *,
    log_path: Path,
) -> dict[str, Any]:
    if not provider.runtime_python:
        raise RuntimeError(f"provider {provider.provider} has no prepared runtime")
    timeout_seconds = _runner_timeout_seconds(request)
    source_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="turbobench-request-") as raw_temp:
        temporary = Path(raw_temp)
        if source_root.name != "src":
            # An installed CLI must not shadow the provider's dependency lock
            # by putting its entire site-packages directory first on sys.path.
            isolated_source = temporary / "harness"
            shutil.copytree(
                Path(__file__).resolve().parent,
                isolated_source / "turbobench",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            source_root = isolated_source
        request_path = temporary / "request.json"
        response_path = temporary / "response.json"
        write_json(request_path, request)
        environment = _offline_environment(source_root)
        process = subprocess.Popen(
            [
                provider.runtime_python,
                "-I",
                "-c",
                _ISOLATED_ENTRYPOINT,
                str(source_root),
                str(request_path),
                str(response_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=os.name != "nt",
        )
        try:
            stdout, _stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stdout = _terminate_runner(process)
            redacted_log = _ABSOLUTE_PATH.sub("<redacted-path>", stdout or "")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(redacted_log, encoding="utf-8")
            raise RuntimeError(
                f"{provider.provider} runner exceeded {timeout_seconds} seconds; "
                f"see {log_path.name}"
            ) from None
        except BaseException:
            _terminate_runner(process)
            raise
        redacted_log = _ABSOLUTE_PATH.sub("<redacted-path>", stdout or "")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(redacted_log, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(
                f"{provider.provider} runner failed with exit {process.returncode}; see {log_path.name}"
            )
        if not response_path.is_file():
            raise RuntimeError(f"{provider.provider} runner produced no response")
        return read_json(response_path)


def _terminate_runner(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        stdout, _stderr = process.communicate()
    return stdout or ""


def _offline_environment(source_root: Path) -> dict[str, str]:
    keep = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TEMP",
        "TMP",
        "DYLD_LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "SDL_VIDEODRIVER",
    }
    environment = {key: value for key, value in os.environ.items() if key in keep}
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_NO_INDEX": "1",
            "UV_OFFLINE": "1",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "",
            "TURBOBENCH_NETWORK_DISABLED": "1",
        }
    )
    return environment


_ISOLATED_ENTRYPOINT = """
import sys
sys.path.insert(0, sys.argv[1])
from turbobench.runner import main
raise SystemExit(main(sys.argv[2:]))
"""
