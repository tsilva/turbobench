from __future__ import annotations

from pathlib import Path

from turbobench.bundle import verify_bundle
from turbobench.engine import (
    ComparisonOptions,
    generate_promo_for_bundle,
    run_comparison_resolved,
)
from turbobench.profiles import get_profile
from turbobench.resolution import fake_resolved
from turbobench.runtime import prepare_runtime
from turbobench.util import read_json


def test_fake_provider_e2e_generates_verified_watermarked_mp4_and_gif(tmp_path: Path) -> None:
    profile = get_profile("vizdoom/basic-v1")
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
    messages: list[str] = []
    result = generate_promo_for_bundle(bundle, diagnostic=True, progress=messages.append)
    assert result["promo"]["generated"]
    assert result["promo"]["diagnostic_watermark"]
    assert (bundle / "media" / "comparison.mp4").is_file()
    assert (bundle / "media" / "comparison.gif").is_file()
    media = read_json(bundle / "media" / "media-manifest.json")
    assert media["diagnostic_watermark"]
    assert media["outputs"]["mp4"]["probe"]["streams"][0]["codec_name"] == "h264"
    assert media["outputs"]["mp4"]["probe"]["streams"][0]["pix_fmt"] == "yuv420p"
    assert media["outputs"]["gif"]["probe"]["streams"][0]["width"] == 640
    assert verify_bundle(bundle)["passed"]
    assert messages[0] == "Verifying source bundle"
    assert "Generating promotional MP4 and GIF" in messages
    assert messages[-1] == f"Promotional media complete: {bundle}"
