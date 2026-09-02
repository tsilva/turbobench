from __future__ import annotations

from pathlib import Path

import pytest

from turbobench.engine import ComparisonOptions, run_comparison_resolved
from turbobench.profiles import get_profile
from turbobench.resolution import fake_resolved
from turbobench.runtime import prepare_runtime


@pytest.fixture
def fake_bundle(tmp_path: Path) -> Path:
    profile = get_profile("supermario/world1-v1")
    left = prepare_runtime(fake_resolved("fake-slow", speed=1.0))
    right = prepare_runtime(fake_resolved("fake-fast", speed=2.0))
    bundle, _result = run_comparison_resolved(
        profile,
        left,
        right,
        tmp_path / "bundle",
        ComparisonOptions(quick=True),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    return bundle
