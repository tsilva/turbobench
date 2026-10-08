from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from turbobench import runner_client


def policy_request(shape: int) -> dict:
    return {
        "operation": "benchmark",
        "shape": shape,
        "actions": [[0]] * 3000,
        "repetitions": 3,
        "warmup_steps": 500,
        "execution_spec": {"constructor": {"frame_skip": 2}},
    }


def test_short_requests_keep_existing_deadline_and_long_captures_scale():
    assert runner_client._runner_timeout_seconds({"operation": "contract"}) == 900
    assert runner_client._runner_timeout_seconds(policy_request(64)) == 900
    assert runner_client._runner_timeout_seconds(policy_request(256)) == 3600
    longer = policy_request(64)
    longer["actions"] *= 2
    assert runner_client._runner_timeout_seconds(longer) > 900
    wider_skip = policy_request(64)
    wider_skip["execution_spec"]["constructor"]["frame_skip"] = 8
    assert runner_client._runner_timeout_seconds(wider_skip) == 3600


def test_large_worker_can_finish_but_small_worker_is_still_terminated(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_client, "RUNNER_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(
        runner_client,
        "_ISOLATED_ENTRYPOINT",
        "import pathlib, sys, time; time.sleep(1.5); "
        "pathlib.Path(sys.argv[3]).write_text('{\"finished\": true}')",
    )
    provider = SimpleNamespace(runtime_python=sys.executable, provider="deadline-test")
    with pytest.raises(RuntimeError, match="exceeded 1 seconds"):
        runner_client.invoke_runner(provider, policy_request(64), log_path=tmp_path / "small.log")
    result = runner_client.invoke_runner(
        provider, policy_request(256), log_path=tmp_path / "large.log"
    )
    assert result == {"finished": True}
