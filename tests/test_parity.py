from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from turbobench.cli import main
from turbobench.parity import (
    ParityOptions,
    _finalize_parity_manifest,
    _requires_ram,
    parity_gate_for_benchmark,
    run_parity_resolved,
    verify_parity_receipt,
)
from turbobench.parity_profiles import get_parity_profile
from turbobench.profiles import get_profile
from turbobench.resolution import fake_resolved, with_runtime
from turbobench.runtime import prepare_runtime


@pytest.fixture
def parity_receipt(tmp_path: Path) -> Path:
    profile = get_profile("supermario/world1-v2")
    parity_profile = get_parity_profile(profile.id)
    authority = prepare_runtime(
        with_runtime(fake_resolved("stable-retro"), version="1.0.1")
    )
    candidate = prepare_runtime(fake_resolved("env-supermariobrosnes-turbo-emu"))
    receipt, result = run_parity_resolved(
        parity_profile,
        profile,
        authority,
        candidate,
        tmp_path / "parity-receipt",
        ParityOptions(steps=4, shapes=(1,)),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    assert result["passed"]
    return receipt


def test_parity_receipt_verifies_exact_checks(parity_receipt: Path) -> None:
    result = verify_parity_receipt(parity_receipt)

    assert result["passed"]
    assert result["artifact_count"] > 0


def test_ram_evidence_is_required_only_where_both_adapters_expose_it() -> None:
    stable = fake_resolved("stable-retro")
    stable_turbo = fake_resolved("env-stableretro-turbo")
    breakout = fake_resolved("env-breakoutatari2600-turbo-native")
    vizdoom = fake_resolved("vizdoom")
    vizdoom_turbo = fake_resolved("env-vizdoom-turbo")

    assert _requires_ram(get_profile("supermario/world1-v2"), stable, stable_turbo)
    assert _requires_ram(get_profile("breakout/start-v2"), stable, stable_turbo)
    assert not _requires_ram(get_profile("breakout/start-v2"), stable, breakout)
    assert not _requires_ram(get_profile("vizdoom/basic-v2"), vizdoom, vizdoom_turbo)


def test_benchmark_reuse_requires_same_artifacts_and_covered_shapes(
    parity_receipt: Path,
) -> None:
    profile = get_profile("supermario/world1-v1")
    authority = prepare_runtime(
        with_runtime(fake_resolved("stable-retro"), version="1.0.1")
    )
    candidate = prepare_runtime(fake_resolved("env-supermariobrosnes-turbo-emu"))

    accepted = parity_gate_for_benchmark(
        parity_receipt, profile, (candidate, authority), (1,)
    )
    rejected = parity_gate_for_benchmark(
        parity_receipt, profile, (candidate, authority), (16,)
    )
    too_long = parity_gate_for_benchmark(
        parity_receipt,
        replace(profile, correctness_steps=257),
        (candidate, authority),
        (1,),
    )

    assert accepted["passed"]
    assert not rejected["passed"]
    assert "parity receipt does not cover every benchmark lane shape" in rejected["errors"]
    assert not too_long["passed"]
    assert "parity receipt benchmark workload is too short for shape 1" in too_long["errors"]


def test_canonical_gate_rejects_short_or_non_pypi_receipt(
    parity_receipt: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(
        [
            "verify-parity",
            str(parity_receipt),
            "--require-canonical",
            "--require-provider",
            "env-supermariobrosnes-turbo-emu",
        ]
    ) == 1
    output = json.loads(capsys.readouterr().out)
    assert "receipt does not contain every canonical parity shape" in output["errors"]
    assert "canonical receipt does not pin the PyPI semantic authority" in output["errors"]
    assert "canonical receipt candidate must be an exact local distribution artifact" in output["errors"]


def test_receipt_with_failed_parity_check_never_verifies(
    parity_receipt: Path,
    tmp_path: Path,
) -> None:
    failed = tmp_path / "failed-receipt"
    shutil.copytree(parity_receipt, failed)
    (failed / "manifest.json").unlink()
    result_path = failed / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["passed"] = False
    for check in result["checks"].values():
        check["passed"] = False
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _finalize_parity_manifest(failed)

    verification = verify_parity_receipt(failed)

    assert not verification["passed"]
    assert "one or more exact semantic checks failed" in verification["errors"]
