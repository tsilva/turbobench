from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from turbobench.cli import main
from turbobench.profiles import get_profile
from turbobench.resolution import fake_resolved, with_runtime
from turbobench.runtime import prepare_runtime
from turbobench.semantic_oracle import (
    OracleOptions,
    _finalize_oracle_manifest,
    run_oracle_resolved,
    verify_oracle_receipt,
)


@pytest.fixture
def semantic_receipt(tmp_path: Path) -> Path:
    profile = get_profile("supermario/canonical-v2")
    authority = prepare_runtime(
        with_runtime(fake_resolved("stable-retro"), version="1.0.1")
    )
    candidate = prepare_runtime(fake_resolved("fake-candidate"))
    receipt, result = run_oracle_resolved(
        profile,
        authority,
        candidate,
        tmp_path / "semantic-receipt",
        OracleOptions(steps=4, shapes=(1,)),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    assert result["passed"]
    return receipt


def test_semantic_receipt_verifies_exact_checks(semantic_receipt: Path) -> None:
    result = verify_oracle_receipt(semantic_receipt)

    assert result["passed"]
    assert result["artifact_count"] > 0


def test_canonical_gate_rejects_short_or_non_pypi_receipt(
    semantic_receipt: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(
        [
            "verify-oracle",
            str(semantic_receipt),
            "--require-canonical",
            "--require-provider",
            "supermariobrosnes-turbo",
        ]
    ) == 1
    output = json.loads(capsys.readouterr().out)
    assert "receipt does not contain every canonical oracle shape" in output["errors"]
    assert "canonical receipt does not pin the PyPI semantic authority" in output["errors"]
    assert "canonical receipt candidate must be an installed PyPI release" in output["errors"]
    assert "required provider is absent: supermariobrosnes-turbo" in output["errors"]


def test_receipt_with_failed_semantic_check_never_verifies(
    semantic_receipt: Path,
    tmp_path: Path,
) -> None:
    failed = tmp_path / "failed-receipt"
    shutil.copytree(semantic_receipt, failed)
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
    _finalize_oracle_manifest(failed)

    verification = verify_oracle_receipt(failed)

    assert not verification["passed"]
    assert "one or more exact semantic checks failed" in verification["errors"]
