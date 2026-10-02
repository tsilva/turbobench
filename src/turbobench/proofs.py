"""Versioned policy, measurement and derived-showcase proof packages."""

from __future__ import annotations

import json
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any

from turbobench.profiles import get_profile
from turbobench.util import canonical_json_hash, read_json, sha256_file, write_json

PROOF_SCHEMAS = {f"turbobench.{kind}-proof/v1" for kind in ("policy", "benchmark", "showcase")}


def validate_document(document: dict[str, Any]) -> None:
    """Validate the shipped JSON schema subset without executing plug-in code."""
    name = document.get("schema", "")
    allowed = PROOF_SCHEMAS | {"turbobench.comparison-request/v1"}
    if name not in allowed:
        raise ValueError(f"unsupported proof schema: {name}")
    schema = json.loads(
        files("turbobench")
        .joinpath("schemas", name.split(".", 1)[1].replace("/", "-") + ".json")
        .read_text()
    )

    def check(value: Any, rule: dict[str, Any], location: str) -> None:
        types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
        if "type" in rule and type(value) is not types[rule["type"]]:
            raise ValueError(f"{location}: expected {rule['type']}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"{location}: invalid enum value")
        if "const" in rule and value != rule["const"]:
            raise ValueError(f"{location}: invalid schema constant")
        if isinstance(value, dict):
            if set(rule.get("required", [])) - value.keys():
                raise ValueError(f"{location}: missing required fields")
            properties = rule.get("properties", {})
            if rule.get("additionalProperties") is False and value.keys() - properties.keys():
                raise ValueError(f"{location}: unexpected fields")
            for key, child in value.items():
                if key in properties:
                    check(child, properties[key], location + "." + key)
        if isinstance(value, list) and "items" in rule:
            for index, child in enumerate(value):
                check(child, rule["items"], f"{location}[{index}]")

    check(document, schema, name)


def proof_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p != root / "manifest.json")


def finalize_proof(root: Path, schema: str, bindings: dict[str, Any]) -> dict[str, Any]:
    schema_directory = root / "schemas"
    schema_directory.mkdir(exist_ok=True)
    for resource in files("turbobench").joinpath("schemas").iterdir():
        if resource.name.endswith(".json"):
            (schema_directory / resource.name).write_bytes(resource.read_bytes())
    records = []
    for path in proof_files(root):
        if path.is_symlink():
            raise ValueError("proof packages cannot contain symlinks")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {"schema": schema, "proof_id": "", "bindings": bindings, "artifacts": records}
    manifest["proof_id"] = canonical_json_hash(manifest)
    validate_document(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def require_proof(root: Path) -> dict[str, Any]:
    result = verify_proof(root)
    if not result["passed"]:
        raise ValueError("proof verification failed: " + "; ".join(result["errors"]))
    return read_json(root / "manifest.json")


def verify_proof(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    manifest: dict[str, Any] = {}
    try:
        manifest = read_json(root / "manifest.json")
        validate_document(manifest)
        if manifest["proof_id"] != canonical_json_hash({**manifest, "proof_id": ""}):
            raise ValueError("proof ID does not match canonical content")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("proof package contains a symlink")
        actual = {p.relative_to(root).as_posix() for p in proof_files(root)}
        recorded: set[str] = set()
        for record in manifest["artifacts"]:
            relative = record["path"]
            path = Path(relative)
            if (
                not relative
                or path.is_absolute()
                or ".." in path.parts
                or relative in recorded
                or relative == "manifest.json"
            ):
                raise ValueError("unsafe or duplicate artifact path")
            recorded.add(relative)
            target = root / path
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("artifact escapes proof package")
            if target.stat().st_size != record["size"] or sha256_file(target) != record["sha256"]:
                raise ValueError(f"artifact hash/size mismatch: {relative}")
        if recorded != actual:
            raise ValueError("proof inventory differs from package contents")
        for resource in files("turbobench").joinpath("schemas").iterdir():
            if (
                resource.name.endswith(".json")
                and (root / "schemas" / resource.name).read_bytes() != resource.read_bytes()
            ):
                raise ValueError("embedded schema differs from the declared contract version")
        bindings = manifest["bindings"]
        if manifest["schema"] == "turbobench.policy-proof/v1":
            contract = policy_contract(root, get_profile(bindings["profile"]))
            if contract != bindings["contract"]:
                raise ValueError("policy contract binding mismatch")
        elif manifest["schema"] == "turbobench.benchmark-proof/v1":
            from turbobench.bundle import _verify_consistency
            from turbobench.workflow import (
                validate_request,
                verify_measurement_replay,
                verify_sampling,
            )

            request = read_json(root / "request.json")
            validate_request(request)
            result = read_json(root / "result.json")
            if bindings != {
                "request_id": request["request_id"],
                "policy_id": request["policy_id"],
                "benchmark_machine": read_json(root / "benchmark-machine.json"),
                "mode": "smoke" if request["smoke"] else "full",
            }:
                raise ValueError("benchmark request/host binding mismatch")
            if (
                request["harness_sha256"] != result["tool"]["source_sha256"]
                or result["profile"]["id"] != request["profile"]
            ):
                raise ValueError("measurement used a different harness/profile")
            _verify_consistency(root, {"bundle_id": manifest["proof_id"]}, errors, [])
            if request["smoke"] != (result["schema"] == "turbobench.result/v3"):
                raise ValueError("measurement sampling mode differs from request")
            verify_sampling(root, request, result)
            verify_measurement_replay(root, request)
            if (
                (root / "chart.svg").exists()
                or (root / "report.md").exists()
                or list((root / "media").glob("*"))
            ):
                raise ValueError("measurement host emitted showcase assets")
        else:
            from turbobench.workflow import verify_showcase

            verify_showcase(root, bindings)
    except (ValueError, KeyError, TypeError, IndexError, OSError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return {
        "passed": not errors,
        "proof_id": manifest.get("proof_id"),
        "errors": errors,
        "warnings": ["integrity verification does not authenticate the result author"],
    }


def policy_contract(root: Path, profile: Any) -> dict[str, Any]:
    from turbobench.policy_contracts import policy_contract as legacy_contract

    return legacy_contract(root, profile)


def pack_policy(
    model: Path,
    capture: Path,
    actions: Path,
    output: Path,
    profile_id: str,
    mlflow_url: str,
    limitations: str,
) -> Path:
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    for name in ("model.json", "model.zip", "recipe.json"):
        shutil.copyfile(model / name, staging / name)
    shutil.copyfile(capture, staging / "capture.json")
    shutil.copyfile(actions, staging / "actions.json")
    write_json(staging / "provenance.json", {"mlflow_url": mlflow_url, "limitations": limitations})
    contract = policy_contract(staging, get_profile(profile_id))
    finalize_proof(
        staging, "turbobench.policy-proof/v1", {"profile": profile_id, "contract": contract}
    )
    require_proof(staging)
    staging.rename(output)
    return output
