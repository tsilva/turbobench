"""Normalize saved policy contracts once, independent of benchmark profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from turbobench.policy import require_policy_frame_skip
from turbobench.util import canonical_json_hash, read_json, sha256_file


def read_policy(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = read_json(root / "model.json")
    recipe = read_json(root / "recipe.json")
    capture = read_json(root / "capture.json")
    actions = read_json(root / "actions.json")
    if (
        model.get("document_type") != "gradlab.model"
        or model.get("format_version") != 3
        or recipe.get("document_type") != "gradlab.recipe"
        or recipe.get("format_version") != 4
    ):
        raise ValueError("unsupported GradLab model/recipe format")
    checkpoint = model["checkpoint"]
    if (
        checkpoint["sha256"] != sha256_file(root / "model.zip")
        or checkpoint["size_bytes"] != (root / "model.zip").stat().st_size
    ):
        raise ValueError("checkpoint bytes do not match saved model metadata")
    if model["recipe"]["sha256"] != sha256_file(root / "recipe.json"):
        raise ValueError("saved recipe digest mismatch")
    if (
        capture.get("schema") != "gradlab.readme-playback/v1"
        or actions.get("schema") != "turbobench.imported-policy-actions/v1"
    ):
        raise ValueError("unsupported capture/action format")
    if capture["checkpoint_sha256"] != checkpoint["sha256"]:
        raise ValueError("capture belongs to a different checkpoint")
    environment = recipe["recipe"]["environment"]
    preprocessing = environment["preprocessing"]
    require_policy_frame_skip(
        capture,
        preprocessing["frame_skip"],
        benchmark_frame_skip=preprocessing["frame_skip"],
        action_stream=actions,
    )
    saved_action = model["provenance"]["training_metadata"]["action_contract"]
    if saved_action != capture["action_contract"]:
        raise ValueError("capture action/wrapper contract differs from training")
    environment_hash = recipe["recipe"]["policy_environment_hash"]
    if (
        capture["contract"].get("policy_environment_hash") != environment_hash
        or capture["contract"].get("training_policy_environment_hash") != environment_hash
    ):
        raise ValueError("capture environment/input/wrapper contract differs from saved recipe")
    # Capture is a recorded inference output. Import never executes untrusted weights.
    summary = {
        "checkpoint_sha256": checkpoint["sha256"],
        "recipe_sha256": sha256_file(root / "recipe.json"),
        "training_environment_sha256": environment_hash,
        "frame_skip": preprocessing["frame_skip"],
        "action_sha256": canonical_json_hash(actions["actions"]),
        "selection": capture["selection"],
        "decisions": (len(actions["actions"]) - actions["reset_noop_prefix"])
        // preprocessing["frame_skip"],
        "total_captured_decisions": len(capture["transitions"]),
        "training_run": model["provenance"]["run_name"],
        "checkpoint_step": checkpoint["step"],
        "mlflow_url": read_json(root / "provenance.json")["mlflow_url"],
        "limitations": read_json(root / "provenance.json")["limitations"],
        "benchmark_differences": [
            "seeded canonical benchmark controls; policy controls used for the showcase excerpt",
            "policy inference, context/task/reward wrappers and training buffer/thread options excluded from matched environment timing",
            "neutral reset-prefix recording control added after trained action IDs; declared Linux RGB565 palette conversion reproduces the saved training transport",
        ],
    }

    return summary, environment, model["provenance"]["training_metadata"]


def policy_contract(root: Path, profile: Any) -> dict[str, Any]:
    summary, environment, _metadata = read_policy(root)
    capture = read_json(root / "capture.json")
    preprocessing = environment["preprocessing"]
    if preprocessing["frame_skip"] != profile.frame_skip:
        raise ValueError("policy/profile cadence mismatch")
    expected = {
        "frame_stack": profile.frame_stack,
        "obs_resize": list(profile.resize),
        "obs_grayscale": profile.grayscale,
        "obs_resize_algorithm": profile.resize_algorithm,
        "max_pool_frames": profile.maxpool_last_two,
        "obs_crop": [profile.crop_top, profile.crop_bottom, 0, 0],
        "obs_crop_mode": profile.crop_mode,
        "sticky_action_prob": 0.0,
    }
    for key, value in expected.items():
        if preprocessing.get(key) != value:
            raise ValueError(f"policy/profile mismatch: {key}")
    provider_args = environment["provider_args"]
    table = [list(profile.action_table[name]) for name in profile.semantic_actions]
    if (
        provider_args.get("use_restricted_actions") != table
        or capture["action_contract"]["requested"]["table"] != table
        or environment["state"] != profile.states[0]
        or provider_args.get("use_fire_reset") is not False
    ):
        raise ValueError("policy/profile reset or action-table mismatch")
    return summary


def normalize_policy(root: Path) -> dict[str, Any]:
    summary, environment, metadata = read_policy(root)
    pre = environment["preprocessing"]
    for name in ("frame_skip", "frame_stack"):
        if type(pre.get(name)) is not int or pre[name] < 1:
            raise ValueError(f"invalid training preprocessing: {name}")
    resize = pre.get("obs_resize")
    if (
        not isinstance(resize, list)
        or len(resize) != 2
        or any(type(v) is not int or v <= 0 for v in resize)
    ):
        raise ValueError("missing or invalid training resize")
    for name in ("obs_grayscale", "max_pool_frames"):
        if type(pre.get(name)) is not bool:
            raise ValueError(f"missing training preprocessing: {name}")
    crop = pre.get("obs_crop")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(type(v) is not int or v < 0 for v in crop)
    ):
        raise ValueError("missing or invalid training crop")
    if pre.get("obs_crop_mode") not in ("remove", "mask") or pre.get(
        "obs_resize_algorithm"
    ) not in ("area", "nearest", "linear"):
        raise ValueError("missing or unsupported training image preprocessing")
    if (
        pre.get("sticky_action_prob") != 0
        or pre.get("obs_crop_fill") != 0
        or environment["provider_args"].get("use_fire_reset") is not False
    ):
        raise ValueError(
            "sticky actions, crop fill, or fire reset need an explicit supported execution adapter"
        )
    execution = metadata.get("policy_execution_contract", {})
    base = execution.get("model_inputs", {}).get("base_observation_space", {})
    channels = pre["frame_stack"] * (1 if pre["obs_grayscale"] else 3)
    if (
        base.get("kind") != "box"
        or base.get("dtype") not in ("|u1", "uint8")
        or base.get("shape") != [channels, resize[1], resize[0]]
    ):
        raise ValueError("saved policy image contract must explicitly establish CHW layout")
    table = environment["provider_args"].get("use_restricted_actions")
    capture = read_json(root / "capture.json")
    if (
        not isinstance(table, list)
        or not table
        or table != capture["action_contract"]["requested"]["table"]
    ):
        raise ValueError("saved policy action ordering differs from captured actions")
    if any(
        not isinstance(row, list) or any(not isinstance(v, str) or not v for v in row)
        for row in table
    ) or len({tuple(row) for row in table}) != len(table):
        raise ValueError("invalid saved policy action table")
    result = {
        **summary,
        "schema": "turbobench.policy-contract/v1",
        "environment": environment,
        "action_contract": metadata["action_contract"],
        "execution_contract": execution,
        "training_versions": metadata.get("versions", {}),
        "layout": "chw",
        "reset_noop_prefix": read_json(root / "actions.json")["reset_noop_prefix"],
    }
    result["benchmark_differences"] = [
        "seeded canonical benchmark controls; policy controls used for the showcase excerpt",
        "policy inference and saved context/task/reward wrappers excluded from environment timing",
        "benchmark obs_copy=copy and num_threads=n_envs; saved training buffer/thread settings retained",
        "neutral reset-prefix recording controls do not change trained action IDs or decision cadence",
    ]
    return result
