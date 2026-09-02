from __future__ import annotations

import json
from pathlib import Path

import pytest

from turbobench.cli import build_parser, main


def test_cli_exposes_benchmark_and_parity_commands(capsys) -> None:
    parser = build_parser()
    help_text = parser.format_help()
    for command in (
        "doctor",
        "providers",
        "profiles",
        "compare",
        "parity",
        "verify",
        "verify-parity",
        "report",
        "promo",
    ):
        assert command in help_text
    assert "publish" not in help_text
    assert main(["providers", "list"]) == 0
    assert "env-supermariobrosnes-turbo-emu" in capsys.readouterr().out
    assert main(["profiles", "list"]) == 0
    assert "vizdoom/basic-v1" in capsys.readouterr().out


def test_verify_and_report_commands_use_self_verified_bundle(fake_bundle: Path, capsys) -> None:
    assert main(["verify", str(fake_bundle)]) == 0
    assert '"passed": true' in capsys.readouterr().out
    assert main(["report", str(fake_bundle)]) == 0
    assert "Shape-local results" in capsys.readouterr().out


def test_verify_command_returns_failure_for_tampering(fake_bundle: Path, capsys) -> None:
    (fake_bundle / "report.md").write_text("tampered\n", encoding="utf-8")
    assert main(["verify", str(fake_bundle)]) == 1
    assert "artifact hash mismatch" in capsys.readouterr().out


def test_compare_streams_progress_to_stderr_and_keeps_json_on_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    output = tmp_path / "bundle"

    def fake_run(profile, left, right, bundle, options):
        assert profile == "supermario/world1-v1"
        assert left.provider == "env-supermariobrosnes-turbo-emu"
        assert right.provider == "env-stableretro-turbo"
        options.report_progress("Resolving package artifacts")
        return bundle.resolve(), {
            "validity": {"passed": True},
            "claim": {"status": "official"},
            "comparison": {"outcome": "inconclusive"},
            "promo": {"requested": False, "eligible": False, "generated": False},
        }

    monkeypatch.setattr("turbobench.cli.run_comparison", fake_run)
    assert main(
        [
            "compare",
            "supermario/world1-v1",
            "--left",
            "env-supermariobrosnes-turbo-emu@latest",
            "--right",
            "env-stableretro-turbo@latest",
            "--output",
            str(output),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert captured.err == "turbobench: Resolving package artifacts\n"
    assert json.loads(captured.out)["bundle"] == str(output.resolve())
