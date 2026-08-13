from __future__ import annotations

import os
from pathlib import Path

import pytest

from turbobench.bundle import finalize_manifest, verify_bundle
from turbobench.engine import ComparisonOptions, run_comparison_resolved
from turbobench.profiles import get_profile
from turbobench.resolution import fake_resolved
from turbobench.runtime import prepare_runtime
from turbobench.util import canonical_json_hash, read_json, write_json


def test_fake_provider_end_to_end_bundle_has_separate_statuses(fake_bundle: Path) -> None:
    verification = verify_bundle(fake_bundle)
    assert verification["passed"]
    result = read_json(fake_bundle / "result.json")
    assert result["schema"] == "turbobench.result/v1"
    assert result["tool"]["distribution"] == "turbobench-cli"
    assert not result["validity"]["passed"]
    assert result["claim"]["status"] == "diagnostic"
    assert result["comparison"]["outcome"] == "right_faster"
    assert not result["promo"]["eligible"]
    assert set(result["comparison"]["shapes"]) == {"1"}
    assert not any(str(value).startswith("/") for value in result["commands"])
    for report in result["turbo_contract"].values():
        assert report["passed"]
        assert any(check["name"] == "constructor common order" for check in report["checks"])


def test_manifest_id_and_all_required_bundle_entries(fake_bundle: Path) -> None:
    manifest = read_json(fake_bundle / "manifest.json")
    assert manifest["tool"]["distribution"] == "turbobench-cli"
    assert manifest["bundle_id"] == canonical_json_hash({**manifest, "bundle_id": ""})
    paths = {item["path"] for item in manifest["artifacts"]}
    assert {"result.json", "profile.toml", "resolved-lock.json", "report.md", "chart.svg"} <= paths
    assert all((fake_bundle / name).is_dir() for name in ("raw", "verification", "media"))
    reversal = read_json(fake_bundle / "verification" / "order-reversal.json")
    assert reversal["providers"] == {"left": "fake-fast", "right": "fake-slow"}
    assert reversal["shapes"]["1"]["raw_evidence_reused"]
    assert reversal["shapes"]["1"]["statistics"]["outcome"] == "left_faster"


def test_verification_allows_git_to_drop_empty_media_directory(
    fake_bundle: Path,
) -> None:
    (fake_bundle / "media").rmdir()

    verification = verify_bundle(fake_bundle)

    assert verification["passed"]


def test_tampering_and_unrecorded_files_are_detected(fake_bundle: Path) -> None:
    result_path = fake_bundle / "result.json"
    result_path.write_text(result_path.read_text() + " ", encoding="utf-8")
    assert not verify_bundle(fake_bundle)["passed"]


def test_status_consistency_detects_failed_validity_marked_official(fake_bundle: Path) -> None:
    result = read_json(fake_bundle / "result.json")
    result["claim"]["status"] = "official"
    (fake_bundle / "manifest.json").unlink()
    write_json(fake_bundle / "result.json", result)
    finalize_manifest(fake_bundle)
    verification = verify_bundle(fake_bundle)
    assert not verification["passed"]
    assert any("failed validity" in item for item in verification["errors"])


def test_resume_reuses_completed_trace_and_invocations(
    fake_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = fake_bundle
    partial = final.with_name(final.name + ".partial")
    os.replace(final, partial)
    profile = get_profile("supermario/canonical-v1")
    left = prepare_runtime(fake_resolved("fake-slow", speed=1.0))
    right = prepare_runtime(fake_resolved("fake-fast", speed=2.0))

    def unexpected(*args, **kwargs):
        raise AssertionError("completed runner evidence should have been resumed")

    monkeypatch.setattr("turbobench.engine.invoke_runner", unexpected)
    bundle, _result = run_comparison_resolved(
        profile,
        left,
        right,
        final,
        ComparisonOptions(quick=True),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    assert bundle == final
    assert verify_bundle(final)["passed"]


def test_portability_scan_rejects_absolute_paths(fake_bundle: Path) -> None:
    result = read_json(fake_bundle / "result.json")
    result["tool"]["source_path"] = "/private/cache/provider/source.py"
    (fake_bundle / "manifest.json").unlink()
    write_json(fake_bundle / "result.json", result)
    finalize_manifest(fake_bundle)
    verification = verify_bundle(fake_bundle)
    assert any("portable output violation" in item for item in verification["errors"])


def test_quick_comparison_reports_stage_and_pair_progress(tmp_path: Path) -> None:
    profile = get_profile("supermario/canonical-v1")
    left = prepare_runtime(fake_resolved("fake-slow", speed=1.0))
    right = prepare_runtime(fake_resolved("fake-fast", speed=2.0))
    messages: list[str] = []

    bundle, _result = run_comparison_resolved(
        profile,
        left,
        right,
        tmp_path / "progress-bundle",
        ComparisonOptions(quick=True, progress=messages.append),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )

    assert messages[0] == "Starting supermario/canonical-v1: shapes 1, 100 benchmark steps"
    assert "Correctness for shape 1: passed" in messages
    assert "Shape 1 warmup: running left provider" in messages
    assert "Shape 1, pair 2/2 (BA): running left provider" in messages
    assert "Self-verifying result bundle" in messages
    assert messages[-1] == f"Comparison complete: {bundle}"
