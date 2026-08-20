from __future__ import annotations

import os
from pathlib import Path

import pytest

from turbobench.engine import ComparisonOptions, run_comparison
from turbobench.providers import load_providers, parse_provider_ref

pytestmark = pytest.mark.acceptance


REAL_PAIRS = (
    ("supermario/canonical-v1", "env-supermariobrosnes-turbo-emu", "stable-retro"),
    ("supermario/canonical-v1", "env-supermariobrosnes-turbo-emu", "env-stableretro-turbo"),
    ("supermario/canonical-v1", "env-stableretro-turbo", "stable-retro"),
    ("breakout/start-v1", "env-breakoutatari2600-turbo-native", "env-stableretro-turbo"),
    ("breakout/start-v1", "env-breakoutatari2600-turbo-native", "stable-retro"),
    ("breakout/start-v1", "env-stableretro-turbo", "stable-retro"),
    ("vizdoom/basic-v1", "env-vizdoom-turbo", "vizdoom"),
)


@pytest.mark.parametrize(("profile", "left", "right"), REAL_PAIRS)
def test_real_provider_pair_acceptance(
    profile: str, left: str, right: str, tmp_path: Path
) -> None:
    if os.environ.get("TURBOBENCH_RUN_REAL_ACCEPTANCE") != "1":
        pytest.skip("set TURBOBENCH_RUN_REAL_ACCEPTANCE=1 with canonical assets")
    providers = load_providers()
    bundle, result = run_comparison(
        profile,
        parse_provider_ref(left, providers),
        parse_provider_ref(right, providers),
        tmp_path / f"{left}-versus-{right}",
        ComparisonOptions(promo=True),
    )
    assert result["validity"]["passed"]
    assert result["claim"]["status"] == "official"
    assert result["promo"]["generated"]
    assert (bundle / "media" / "comparison.mp4").is_file()
    assert (bundle / "media" / "comparison.gif").is_file()
