from __future__ import annotations

import hashlib
from pathlib import Path

from turbobench import assets
from turbobench.assets import discover_assets
from turbobench.profiles import get_profile
from turbobench.system import host_record, load_threshold, wait_for_load
from turbobench.util import find_portability_violations, redact


def test_load_gate_and_force_busy_are_separate(monkeypatch) -> None:
    monkeypatch.setattr("turbobench.system.os.cpu_count", lambda: 8)
    monkeypatch.setattr("turbobench.system.os.getloadavg", lambda: (3.9, 0.0, 0.0))
    assert load_threshold() == 4.0
    assert wait_for_load(timeout_seconds=0)["passed"]
    monkeypatch.setattr("turbobench.system.os.getloadavg", lambda: (5.0, 0.0, 0.0))
    forced = wait_for_load(timeout_seconds=0, force_busy=True)
    assert not forced["passed"] and forced["forced"]


def test_load_gate_reports_busy_retries(monkeypatch) -> None:
    loads = iter(((5.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    messages: list[str] = []
    monkeypatch.setattr("turbobench.system.os.cpu_count", lambda: 8)
    monkeypatch.setattr("turbobench.system.os.getloadavg", lambda: next(loads))
    monkeypatch.setattr("turbobench.system.time.sleep", lambda _seconds: None)

    result = wait_for_load(timeout_seconds=10, progress=messages.append)

    assert result["passed"]
    assert messages == ["System load 5.00 is above 4.00; retrying in 10s"]


def test_redaction_removes_secrets_hostnames_and_absolute_paths(tmp_path: Path) -> None:
    asset_path = str(tmp_path / "roms" / "game.nes")
    value = {
        "api_key": "super-secret",
        "hostname": "workstation.local",
        "asset_path": asset_path,
        "message": f"installed from {tmp_path / 'cache' / 'provider' / 'source'}",
    }
    redacted = redact(value)
    assert redacted["api_key"] == "<redacted>"
    assert "workstation" not in redacted["hostname"]
    assert str(tmp_path) not in redacted["asset_path"]
    assert str(tmp_path) not in redacted["message"]
    assert find_portability_violations(redacted) == []
    assert find_portability_violations(value)


def test_host_record_never_contains_hostname() -> None:
    record = host_record()
    assert "hostname" not in record and "node" not in record
    assert record["os"] and record["architecture"]


def test_asset_discovery_merges_state_catalogs_without_exposing_paths(
    tmp_path: Path, monkeypatch
) -> None:
    profile = get_profile("breakout/start-v1")
    root_one = tmp_path / "one" / profile.game
    root_two = tmp_path / "two" / profile.game
    root_one.mkdir(parents=True)
    root_two.mkdir(parents=True)
    rom = root_one / "rom.a26"
    rom.write_bytes(b"canonical-test-rom")
    (root_one / "data.json").write_text('{"info":{"score":{},"lives":{}}}')
    state = root_two / "Start.state"
    state.write_bytes(b"canonical-test-state")
    info = root_two / "data.json"
    info.write_text('{"info":{"score":{},"lives":{},"ball_y":{}}}')
    scenario = root_two / "scenario.json"
    scenario.write_text("{}")
    monkeypatch.setattr(assets, "BREAKOUT_ROM_SHA256", hashlib.sha256(rom.read_bytes()).hexdigest())
    monkeypatch.setitem(
        assets.STATE_SHA256[profile.id], "Start", hashlib.sha256(state.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("TURBOBENCH_ASSET_ROOT", str(tmp_path / "one"))
    monkeypatch.setenv("RETRO_DATA_PATH", str(tmp_path / "two"))
    private, portable = discover_assets(profile)
    assert portable["available"]
    assert set(private["state_paths"]) == {"Start"}
    assert private["info_schema_path"] == str(info.resolve())
    assert private["scenario_path"] == str(scenario.resolve())
    assert not find_portability_violations(portable)
