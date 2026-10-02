from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from turbobench.bundle import verify_bundle
from turbobench.engine import ComparisonOptions, run_comparison_resolved
from turbobench.profiles import get_profile
from turbobench.proofs import finalize_proof, require_proof, validate_document
from turbobench.resolution import fake_resolved
from turbobench.runtime import harness_source_hash, prepare_runtime
from turbobench.stats import paired_statistics
from turbobench.util import canonical_json_hash, read_json, write_json
from turbobench.workflow import _extract, machine_identity, validate_request


def test_smoke_is_four_single_measurements_without_warmups(tmp_path: Path) -> None:
    root, result = run_comparison_resolved(
        get_profile("supermario/world1-v1"),
        prepare_runtime(fake_resolved("fake-left", speed=1)),
        prepare_runtime(fake_resolved("fake-right", speed=2)),
        tmp_path / "smoke",
        ComparisonOptions(smoke=True),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    assert set(result["comparison"]["shapes"]) == {"1", "2"}
    assert result["claim"]["status"] == "diagnostic"
    assert result["comparison"]["outcome"] == "inconclusive"
    assert result["sampling"] == {"mode": "smoke", "pairs": 1, "repetitions": 1, "warmup_pairs": 0}
    assert not list((root / "raw").rglob("warmup*.json"))
    invocations = list((root / "raw").rglob("pair-*.json"))
    assert len(invocations) == 4
    for invocation in invocations:
        data = read_json(invocation)
        assert data["repetitions"] == len(data["sps"]) == 1
        assert data["warmup_steps"] == 0
    assert verify_bundle(root)["passed"]
    for shape in result["comparison"]["shapes"].values():
        assert shape["statistics"]["bootstrap"] is None
    result["claim"]["status"] = "official"
    write_json(root / "result.json", result)
    from turbobench.bundle import finalize_manifest

    finalize_manifest(root)
    assert not verify_bundle(root)["passed"]


@pytest.mark.parametrize(
    "options",
    [
        ComparisonOptions(smoke=True, quick=True),
        ComparisonOptions(smoke=True, steps=4),
        ComparisonOptions(smoke=True, shapes=(1, 2)),
        ComparisonOptions(smoke=True, promo=True),
    ],
)
def test_smoke_rejects_overrides(tmp_path: Path, options: ComparisonOptions) -> None:
    provider = prepare_runtime(fake_resolved("fake", speed=1))
    with pytest.raises(ValueError, match="cannot be combined"):
        run_comparison_resolved(
            get_profile("supermario/world1-v1"), provider, provider, tmp_path / "bad", options
        )


def test_single_repetition_not_accepted_as_normal_statistics() -> None:
    pairs = [{"left_sps": [1.0], "right_sps": [2.0]}]
    with pytest.raises(ValueError, match="three"):
        paired_statistics(pairs, require_official_design=False)
    assert paired_statistics(pairs, require_official_design=False, smoke=True)["bootstrap"] is None


def request(profile: str = "breakout/firstwall-policy-v1") -> dict:
    actions = [["BUTTON"]] * 1802
    contract = {"frame_skip": 2, "action_sha256": canonical_json_hash(actions)}
    result = {
        "schema": "turbobench.comparison-request/v1",
        "request_id": "",
        "profile": profile,
        "left": "stable-retro@1.0.1",
        "right": "env-breakoutatari2600-turbo-native@0.5.13",
        "python_minor": "3.14",
        "smoke": True,
        "policy_id": "test",
        "actions": actions,
        "policy_contract": contract,
        "render_machine": {"machine_sha256": "other-host", "hardware": {}},
        "harness_sha256": harness_source_hash(),
    }
    result["request_id"] = canonical_json_hash(result)
    return result


def test_request_is_strict_versioned_and_cadence_bound() -> None:
    valid = request()
    validate_request(valid)
    valid["schema"] = "turbobench.comparison-request/v100"
    with pytest.raises(ValueError, match="unsupported"):
        validate_document(valid)
    valid = request()
    valid["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_document(valid)
    valid = request("breakout/start-v1")
    with pytest.raises(ValueError, match="cadence"):
        validate_request(valid)


def test_remote_archive_cannot_escape(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as tar:
        entry = tarfile.TarInfo("../escape")
        entry.size = 1
        tar.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        _extract(archive, tmp_path / "output")


def test_distinct_host_measurement_proof_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from turbobench import workflow

    req = request()
    monkeypatch.setattr(
        workflow, "machine_identity", lambda: {"machine_sha256": "measurement-host", "hardware": {}}
    )
    req["left"] = "stable-retro@1.0+speed.1"
    req["right"] = "env-breakoutatari2600-turbo-native@1.0+speed.2"
    req["request_id"] = canonical_json_hash({**req, "request_id": ""})
    profile = get_profile(req["profile"])
    root, _result = run_comparison_resolved(
        profile,
        prepare_runtime(fake_resolved("stable-retro", speed=1)),
        prepare_runtime(fake_resolved("env-breakoutatari2600-turbo-native", speed=2)),
        tmp_path / "measurement",
        ComparisonOptions(smoke=True, measurement_only=True, workflow_request=req),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    assert verify_bundle(root)["passed"]
    assert not (root / "chart.svg").exists()
    assert not (root / "report.md").exists()
    assert not list(root.rglob("*.rgb"))
    proof = require_proof(root)
    write_json(root / "benchmark-machine.json", req["render_machine"])
    finalize_proof(
        root, proof["schema"], {**proof["bindings"], "benchmark_machine": req["render_machine"]}
    )
    assert not verify_bundle(root)["passed"]


def test_same_machine_refused_before_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from turbobench.workflow import measurement_worker

    req = request()
    req["render_machine"] = machine_identity()
    req["request_id"] = canonical_json_hash({**req, "request_id": ""})
    write_json(tmp_path / "request.json", req)
    with pytest.raises(ValueError, match="different machines"):
        measurement_worker(tmp_path / "request.json", tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_platform_transport_conversion_preserves_declared_palette_and_policy_values() -> None:
    import numpy as np

    from turbobench.runner import _canonical_stella_rgb, _linux_stella_training_transport

    linux = np.asarray(
        [
            [
                [0, 0, 0],
                [142, 142, 142],
                [205, 73, 73],
                [198, 108, 58],
                [180, 122, 48],
                [162, 162, 42],
                [72, 160, 72],
                [66, 72, 205],
                [66, 158, 130],
            ]
        ],
        dtype=np.uint8,
    )
    training = _linux_stella_training_transport(linux)
    gray = ((training.astype(np.uint32) * np.asarray([77, 150, 29])).sum(axis=-1) + 128) >> 8
    assert gray.tolist() == [[0, 142, 87, 103, 106, 126, 124, 110, 139]]
    canonical = _canonical_stella_rgb(training)
    assert canonical[0, 2].tolist() == [200, 72, 72]
    assert len(np.unique(canonical.reshape(-1, 3), axis=0)) == 9
    with pytest.raises(ValueError, match="undeclared"):
        _linux_stella_training_transport(np.asarray([[[255, 1, 255]]], dtype=np.uint8))


def test_scaling_chart_inverts_shape_local_ci_and_keeps_slow_points() -> None:
    from turbobench.showcase import scaling_chart

    shape = paired_statistics([{"left_sps": [2, 2, 2], "right_sps": [1, 1, 1]}] * 7)
    result = {
        "comparison": {
            "left": {"provider": "upstream", "version": "1"},
            "right": {"provider": "candidate", "version": "2"},
            "shapes": {"1": {"statistics": shape}, "32": {"statistics": shape}},
        }
    }
    chart = scaling_chart(result, diagnostic=False)
    assert chart.count("0.50x") == 2
    assert "n_envs=32" in chart
    assert "stroke-dasharray" in chart
    assert 'stroke-width="3"' in chart
    assert "#0d1117" in chart


def test_video_keeps_initial_frame_at_large_speedup_and_holds_final_frame(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from PIL import Image

    from turbobench.showcase import PANEL_POSITIONS, PANEL_SIZE, SIZE, _encode

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg unavailable")
    card = tmp_path / "card.png"
    Image.new("RGB", SIZE, "#0d1117").save(card)
    width, height, count = 16, 16, 300
    raw = tmp_path / "frames.rgb"
    raw.write_bytes(
        bytes([255, 0, 0]) * width * height
        + bytes([0, 0, 255]) * width * height * (count - 2)
        + bytes([0, 255, 0]) * width * height
    )
    record = {"frame_width": width, "frame_height": height, "frame_count": count}
    output = tmp_path / "comparison.mp4"
    total = _encode(
        card,
        dict.fromkeys(("left", "right"), record),
        dict.fromkeys(("left", "right"), raw),
        300,
        output,
    )
    for frame, dominant in ((0, 0), (total - 120, 1), (total - 1, 1)):
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-i",
                str(output),
                "-vf",
                f"select=eq(n\\,{frame})",
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        image = Image.open(io.BytesIO(decoded.stdout)).convert("RGB")
        for x, y in PANEL_POSITIONS:
            pixel = image.getpixel((x + PANEL_SIZE[0] // 2, y + PANEL_SIZE[1] // 2))
            assert pixel[dominant] > 240 and sum(pixel) - pixel[dominant] < 15


def test_full_measurement_sampling_cannot_be_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from turbobench import workflow

    monkeypatch.setattr(
        "turbobench.engine.wait_for_load",
        lambda **kwargs: {"passed": True, "forced": False, "threshold": 4, "samples": []},
    )

    req = request()
    req.update(
        smoke=False,
        left="stable-retro@1.0+speed.1",
        right="env-breakoutatari2600-turbo-native@1.0+speed.2",
    )
    req["request_id"] = canonical_json_hash({**req, "request_id": ""})
    monkeypatch.setattr(
        workflow, "machine_identity", lambda: {"machine_sha256": "measurement-host", "hardware": {}}
    )
    root, result = run_comparison_resolved(
        get_profile(req["profile"]),
        prepare_runtime(fake_resolved("stable-retro", speed=1)),
        prepare_runtime(fake_resolved("env-breakoutatari2600-turbo-native", speed=2)),
        tmp_path / "full",
        ComparisonOptions(measurement_only=True, workflow_request=req),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )
    assert set(result["comparison"]["shapes"]) == {"1", "16", "32"}
    assert result["sampling"] == {"mode": "full", "pairs": 7, "repetitions": 3, "warmup_pairs": 1}
    assert verify_bundle(root)["passed"]
    proof = require_proof(root)
    invocation = root / "raw/shape-1/pair-01-left.json"
    data = read_json(invocation)
    data["sps"] = [v * 10 for v in data["sps"]]
    write_json(invocation, data)
    finalize_proof(root, proof["schema"], proof["bindings"])
    assert not verify_bundle(root)["passed"]


def test_policy_package_rejects_checkpoint_and_training_setting_substitution(
    tmp_path: Path,
) -> None:
    from turbobench.proofs import pack_policy
    from turbobench.util import sha256_file

    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model.zip").write_bytes(b"checkpoint-fixture")
    profile = get_profile("breakout/firstwall-policy-v1")
    preprocessing = {
        "frame_skip": 2,
        "frame_stack": 4,
        "obs_resize": [84, 84],
        "obs_grayscale": True,
        "obs_resize_algorithm": "area",
        "max_pool_frames": False,
        "obs_crop": [17, 0, 0, 0],
        "obs_crop_mode": "mask",
        "sticky_action_prob": 0.0,
    }
    table = [["BUTTON"], ["RIGHT"], ["LEFT"]]
    recipe = {
        "document_type": "gradlab.recipe",
        "format_version": 4,
        "recipe": {
            "policy_environment_hash": "sha256:fixture",
            "environment": {
                "preprocessing": preprocessing,
                "provider_args": {"use_restricted_actions": table, "use_fire_reset": False},
                "state": "Start",
            },
        },
    }
    write_json(model_root / "recipe.json", recipe)
    capture = {
        "schema": "gradlab.readme-playback/v1",
        "checkpoint_sha256": sha256_file(model_root / "model.zip"),
        "contract": {
            "matches_training": True,
            "frame_skip": {"training": 2, "playback": 2},
            "policy_environment_hash": "sha256:fixture",
            "training_policy_environment_hash": "sha256:fixture",
        },
        "frame_skip": 2,
        "initial": {"reset_info": {"noop_reset_count": 1}},
        "action_contract": {"requested": {"table": table}},
        "selection": {"mode": "stochastic", "seed": 123},
        "transitions": [{"native_action": i} for i in [0, 1, 2, 0]],
    }
    model = {
        "document_type": "gradlab.model",
        "format_version": 3,
        "checkpoint": {
            "sha256": capture["checkpoint_sha256"],
            "size_bytes": (model_root / "model.zip").stat().st_size,
            "step": 10,
        },
        "recipe": {"sha256": sha256_file(model_root / "recipe.json")},
        "provenance": {
            "run_name": "test-run",
            "training_metadata": {"action_contract": capture["action_contract"]},
        },
    }
    write_json(model_root / "model.json", model)
    actions = {
        "schema": "turbobench.imported-policy-actions/v1",
        "policy_frame_skip": 2,
        "reset_noop_prefix": 1,
        "actions": [[]] + [table[i] for i in [0, 0, 1, 1, 2, 2, 0, 0]],
    }
    write_json(tmp_path / "capture.json", capture)
    write_json(tmp_path / "actions.json", actions)
    output = pack_policy(
        model_root,
        tmp_path / "capture.json",
        tmp_path / "actions.json",
        tmp_path / "policy",
        profile.id,
        "https://tracking.example/test",
        "fixture",
    )
    proof = require_proof(output)
    (output / "model.zip").write_bytes(b"replacement")
    finalize_proof(output, proof["schema"], proof["bindings"])
    assert not verify_bundle(output)["passed"]
    recipe["recipe"]["environment"]["preprocessing"]["frame_stack"] = 2
    write_json(model_root / "recipe.json", recipe)
    model["recipe"]["sha256"] = sha256_file(model_root / "recipe.json")
    write_json(model_root / "model.json", model)
    with pytest.raises(ValueError, match="frame_stack"):
        pack_policy(
            model_root,
            tmp_path / "capture.json",
            tmp_path / "actions.json",
            tmp_path / "wrong-stack",
            profile.id,
            "https://tracking.example/test",
            "fixture",
        )
