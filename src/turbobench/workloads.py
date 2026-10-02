"""Resolve immutable comparison workloads from authoritative source documents."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path
from typing import Any

from turbobench.model import Profile
from turbobench.util import canonical_json_hash, read_json

WORKLOAD_SCHEMA = "turbobench.resolved-workload/v1"


def _resource(directory: str, identity: str) -> dict[str, Any]:
    if "/" not in identity or any(part in ("", ".", "..") for part in identity.split("/")):
        raise ValueError("invalid comparison definition/protocol identity")
    path = files("turbobench").joinpath(directory, identity.replace("/", "--") + ".json")
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"unknown comparison definition/protocol: {identity}") from exc


def comparison_definition(identity: str) -> dict[str, Any]:
    return _resource("comparison_definitions", identity)


def definition_for_policy(contract: dict[str, Any]) -> str:
    game = contract["environment"]["env_id"].partition(":")[2]
    matches = []
    for resource in files("turbobench").joinpath("comparison_definitions").iterdir():
        if resource.name.endswith(".json"):
            definition = json.loads(resource.read_text())
            if definition["game"] == game:
                matches.append(definition["id"])
    if len(matches) != 1:
        raise ValueError("policy game has no unique compatible TurboBench comparison definition")
    return matches[0]


def _configuration(definition: dict[str, Any], protocol: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    env = policy["environment"]
    if env["env_id"].partition(":")[2] != definition["game"]:
        raise ValueError("policy game differs from comparison definition")
    pre = env["preprocessing"]
    if pre["obs_crop"][2:] != [0, 0]:
        raise ValueError("horizontal cropping needs an explicit supported execution adapter")
    # The current scalar adapter implements area/grayscale/CHW without pooling.
    # These are implementation capabilities, never defaults for missing metadata.
    if pre["obs_resize_algorithm"] != "area" or pre["max_pool_frames"] or not pre["obs_grayscale"] or policy["layout"] != "chw":
        raise ValueError("current upstream adapter does not support requested image preprocessing")
    table = env["provider_args"]["use_restricted_actions"]
    names = [f"policy_{i}" for i in range(len(table))]
    action_table = dict(zip(names, deepcopy(table), strict=True))
    if [] not in table:
        action_table["reset_noop"] = []
    return {
        "schema": "turbobench.workload-profile/v2",
        "logical_environment": definition["logical_environment"],
        "game": definition["game"],
        "providers": definition["providers"],
        "states": [env["state"]],
        "semantic_actions": names,
        "action_table": action_table,
        "info_integer": definition["info_integer"],
        "info_float": definition["info_float"],
        "observation": {
            "frame_skip": pre["frame_skip"], "frame_stack": pre["frame_stack"],
            "crop_top": pre["obs_crop"][0], "crop_bottom": pre["obs_crop"][1],
            "crop_mode": pre["obs_crop_mode"], "resize": pre["obs_resize"],
            "grayscale": pre["obs_grayscale"], "layout": policy["layout"],
            "resize_algorithm": pre["obs_resize_algorithm"], "maxpool_last_two": pre["max_pool_frames"],
        },
        "run": protocol["run"],
        "parity": {**protocol["parity"], "authority": definition["authority"],
            "authority_version": definition["authority_version"], "candidates": definition["candidates"], "checks": definition["checks"]},
        "promo": {"kind": "policy-excerpt/v1", "steps": policy["decisions"] * pre["frame_skip"] + policy["reset_noop_prefix"],
            "completion_json": json.dumps({"kind": "trajectory-end", "step": policy["decisions"] * pre["frame_skip"] + policy["reset_noop_prefix"]})},
        "exact": {"native_transition_exact": True, "allowed_representation_conversion": definition["allowed_representation_conversion"]},
    }


def check_declaration(declaration: dict[str, Any], configuration: dict[str, Any]) -> None:
    from turbobench.lifecycle import require_attestation
    from turbobench.proofs import validate_document

    validate_document(declaration)
    response = declaration["preflight"]
    require_attestation(response["execution_spec"], response["contract_attestation"])
    if declaration["provider"] != response["execution_spec"]["provider"]:
        raise ValueError("environment declaration artifact/preflight binding mismatch")
    identity = declaration["provider"]["provider"]
    if identity not in configuration["providers"]:
        raise ValueError("declaration is not a compatible provider")
    requested = response["execution_spec"]["constructor"]
    observation = configuration["observation"]
    fields = {"frame_skip": "frame_skip", "frame_stack": "frame_stack", "grayscale": "grayscale", "resize": "resize", "resize_algorithm": "resize_algorithm", "crop_mode": "crop_mode", "layout": "layout", "maxpool_last_two": "maxpool_last_two"}
    if any(requested.get(k) != observation[v] for k, v in fields.items()) or requested.get("crop") != [observation["crop_top"], observation["crop_bottom"], 0, 0]:
        raise ValueError("declaration probed different preprocessing")
    if requested.get("action_table") != list(configuration["action_table"].values()) or requested.get("states") != configuration["states"]:
        raise ValueError("declaration probed different actions/states")
    capabilities = declaration["capabilities"]
    for key, value in (("supported_observation_layouts", observation["layout"]), ("supported_observation_color_modes", "grayscale" if observation["grayscale"] else "rgb"), ("supported_resize_algorithms", observation["resize_algorithm"]), ("supported_crop_modes", observation["crop_mode"])):
        if value not in capabilities.get(key, []):
            raise ValueError(f"{identity}: requested operation not supported: {key}")
    if observation["maxpool_last_two"] and capabilities.get("supports_maxpool_last_two") is not True:
        raise ValueError("provider cannot maxpool")
    if not set(configuration["states"]).issubset(declaration["states"]) or any(row not in declaration["actions"] for row in configuration["action_table"].values()):
        raise ValueError("provider does not declare selected states/actions")
    if response.get("workload_executed") is not False or response["lifecycle"].get("environment_closed") is not True:
        raise ValueError("declaration requires a closed isolated preflight")


def resolve_workload(definition: dict[str, Any], protocol: dict[str, Any], policy: dict[str, Any], declarations: dict[str, Any]) -> dict[str, Any]:
    if definition != comparison_definition(definition["id"]) or protocol != _resource("comparison_protocols", definition["protocol"]):
        raise ValueError("comparison rules differ from their trusted version")
    if policy.get("schema") != "turbobench.policy-contract/v1":
        raise ValueError("unsupported normalized policy contract")
    configuration = _configuration(definition, protocol, policy)
    for declaration in declarations.values():
        check_declaration(declaration, configuration)
    sources = {"policy": canonical_json_hash(policy), "definition": canonical_json_hash(definition), "protocol": canonical_json_hash(protocol), "declarations": {side: canonical_json_hash(value) for side, value in declarations.items()}}
    workload = {"schema": WORKLOAD_SCHEMA, "definition": definition, "protocol": protocol, "policy_contract": policy, "declarations": declarations, "configuration": configuration, "source_sha256": sources,
        "field_sources": {"observation": "policy.environment.preprocessing", "actions": "policy.environment.provider_args.use_restricted_actions", "state": "policy.environment.state", "run": "protocol.run", "checks": "definition.checks", "capability_validation": "declarations"}}
    workload["id"] = definition["id"] + "/" + canonical_json_hash(workload)
    return workload


def preliminary_workload(policy: dict[str, Any], identity: str | None = None) -> dict[str, Any]:
    definition = comparison_definition(identity or definition_for_policy(policy))
    return resolve_workload(definition, _resource("comparison_protocols", definition["protocol"]), policy, {})


def profile_from_workload(workload: dict[str, Any]) -> Profile:
    from turbobench.profile_config import _parse_document
    from turbobench.proofs import validate_document

    validate_document(workload)
    expected = resolve_workload(workload["definition"], workload["protocol"], workload["policy_contract"], workload["declarations"])
    if expected != workload:
        raise ValueError("resolved workload differs from authoritative inputs")
    raw = deepcopy(workload["configuration"])
    raw["id"] = workload["id"]
    parsed = _parse_document(raw, raw["id"].replace("/", "--") + ".toml", "").profile
    from dataclasses import replace

    return replace(parsed, resolved_workload=deepcopy(workload))


def request_profile(request: dict[str, Any]) -> Profile:
    from turbobench.profiles import get_profile

    if "resolved_workload" in request:
        profile = profile_from_workload(request["resolved_workload"])
        if request["profile"] != profile.id:
            raise ValueError("request profile differs from resolved workload")
        return profile
    return get_profile(request["profile"])


def bundle_profile(root: Path, result: dict[str, Any]) -> Profile:
    from turbobench.profiles import get_profile

    path = root / "resolved-workload.json"
    if path.exists():
        profile = profile_from_workload(read_json(path))
        if profile.id != result["profile"]["id"]:
            raise ValueError("result profile differs from archived workload")
        return profile
    return get_profile(result["profile"]["id"])
