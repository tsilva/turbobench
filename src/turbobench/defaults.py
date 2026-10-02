"""Local convenience defaults; resolved inputs still enter immutable proofs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from turbobench.profiles import get_profile
from turbobench.proofs import require_proof
from turbobench.providers import load_providers
from turbobench.util import read_json, write_json


def config_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "turbobench" / "config.json"


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {"schema": "turbobench.local-defaults/v1"}
    config = read_json(path)
    if not isinstance(config, dict) or config.get("schema") != "turbobench.local-defaults/v1":
        raise ValueError(f"unsupported local defaults in {path}")
    if set(config) - {"schema", "benchmark_host", "policy", "policy_id"}:
        raise ValueError(f"unknown local defaults in {path}")
    if any(not isinstance(value, str) or not value for value in config.values()):
        raise ValueError(f"invalid local defaults in {path}")
    return config


def _host(host: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host):
        raise ValueError("benchmark host must be a configured SSH hostname or alias")
    return host


def _policy(path: Path) -> dict[str, Any]:
    proof = require_proof(path)
    if proof["schema"] != "turbobench.policy-proof/v1":
        raise ValueError(f"not a policy package: {path}")
    return proof


def configure(host: str | None, policy: Path | None) -> dict[str, Any]:
    config = load_config()
    if host is not None:
        config["benchmark_host"] = _host(host)
    if policy is not None:
        proof = _policy(policy)
        config.update(policy=str(policy.resolve()), policy_id=proof["proof_id"])
    if host is not None or policy is not None:
        write_json(config_path(), config)
    return {"config": str(config_path()), **config}


def _discover_policy(profile: str | None) -> tuple[Path, dict[str, Any]]:
    matches = {}
    for manifest in sorted(Path("turbobench-results/policies").glob("*/manifest.json")):
        # Incomplete/obsolete packages are not candidates for automatic selection.
        try:
            proof = _policy(manifest.parent)
        except (ValueError, RuntimeError, FileNotFoundError):
            continue
        if profile is None or proof["bindings"]["profile"] == profile:
            matches.setdefault(proof["proof_id"], (manifest.parent, proof))
    if len(matches) == 1:
        return next(iter(matches.values()))
    if not matches:
        raise ValueError(
            "no verified matching policy package; import with policy-pack, then configure --policy PATH or use --policy PATH"
        )
    raise ValueError(
        "multiple verified policy packages; choose one with configure --policy PATH or --policy PATH"
    )


def _training_provider(path: Path, profile: Any) -> str:
    environment = read_json(path / "recipe.json")["recipe"]["environment"]
    provider = environment.get("env_id", "").partition(":")[0]
    if provider not in profile.candidates:
        raise ValueError(
            "policy training provider is not a profile candidate; select --right PROVIDER_REF"
        )
    definition = load_providers()[provider]
    versions = read_json(path / "model.json")["provenance"]["training_metadata"].get("versions", {})
    key = re.sub(r"[-.]", "_", definition.distribution)
    version = versions.get(key)
    if not isinstance(version, str) or not version:
        raise ValueError(
            "policy has no saved training-provider version; select --right PROVIDER_REF"
        )
    return f"{provider}@{version}"


def apply_comparison_defaults(args: Any) -> None:
    if args.showcase:
        config = load_config()
        proof = None
        if args.policy is None and "policy" in config:
            selected = Path(config["policy"])
            selected_proof = _policy(selected)
            if selected_proof["proof_id"] != config.get("policy_id"):
                raise ValueError(
                    "configured policy changed; configure --policy PATH to select it again"
                )
            if args.profile is None or selected_proof["bindings"]["profile"] == args.profile:
                args.policy, proof = selected, selected_proof
        if args.policy is None:
            args.policy, proof = _discover_policy(args.profile)
        proof = proof or _policy(args.policy)
        args.profile = args.profile or proof["bindings"]["profile"]
        if args.profile != proof["bindings"]["profile"]:
            raise ValueError("locked policy package does not match the selected profile")
        args.benchmark_host = (
            args.benchmark_host
            or os.environ.get("TURBOBENCH_BENCHMARK_HOST")
            or config.get("benchmark_host")
        )
        if not args.benchmark_host:
            raise ValueError(
                "no benchmark host; configure --benchmark-host SSH_HOST once or use --benchmark-host SSH_HOST"
            )
        _host(args.benchmark_host)
    if not args.profile:
        raise ValueError("provide a profile ID, or use --showcase with a verified policy package")
    profile = get_profile(args.profile)
    args.left = args.left or f"{profile.authority}@{profile.authority_version}"
    if not args.right:
        if args.showcase:
            args.right = _training_provider(args.policy, profile)
        elif len(profile.candidates) == 1:
            args.right = f"{profile.candidates[0]}@latest"
        else:
            raise ValueError("profile has multiple candidates; select --right PROVIDER_REF")
