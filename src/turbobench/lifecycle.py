"""Phase-isolated provider execution protocol and attestation bindings.

This module owns the portable identity shared by orchestration and isolated
provider runners.  Contract probes consume their process and environment;
workload processes receive only the resulting immutable attestation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from turbobench.util import canonical_json_hash, find_portability_violations

EXECUTION_PROTOCOL = "turbobench.phase-isolated-execution/v1"
EXECUTION_SPEC_SCHEMA = "turbobench.execution-spec/v1"
ATTESTATION_SCHEMA = "turbobench.contract-attestation/v1"


class AttestationError(ValueError):
    """A workload lacks a successful attestation for its exact configuration."""


def execution_spec(
    *,
    provider: Mapping[str, Any],
    harness: Mapping[str, Any],
    python_minor: str,
    python_identity: Mapping[str, Any] | None = None,
    platform: Mapping[str, Any],
    profile: Mapping[str, Any],
    constructor: Mapping[str, Any],
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the portable identity of one provider execution configuration."""

    spec = {
        "schema": EXECUTION_SPEC_SCHEMA,
        "protocol": EXECUTION_PROTOCOL,
        "provider": dict(provider),
        "harness": dict(harness),
        "python_minor": str(python_minor),
        "python": dict(python_identity or {"minor": str(python_minor)}),
        "platform": dict(platform),
        "profile": dict(profile),
        "constructor": dict(constructor),
        "assets": dict(assets),
    }
    violations = find_portability_violations(spec)
    if violations:
        raise ValueError("execution spec must be portable: " + "; ".join(violations))
    spec["execution_spec_sha256"] = canonical_json_hash(spec)
    return spec


def attest(spec: Mapping[str, Any], contract_report: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a completed contract probe to an exact execution spec."""

    payload = {
        "schema": ATTESTATION_SCHEMA,
        "protocol": EXECUTION_PROTOCOL,
        "execution_spec_sha256": spec.get("execution_spec_sha256"),
        "passed": bool(contract_report.get("passed")),
        "promotable": bool(contract_report.get("promotable")),
        "contract_report": dict(contract_report),
    }
    payload["attestation_sha256"] = canonical_json_hash(payload)
    return payload


def require_attestation(
    spec: Mapping[str, Any], attestation: Mapping[str, Any] | None
) -> str:
    """Fail closed unless an attestation matches this exact execution spec."""

    if not isinstance(attestation, Mapping):
        raise AttestationError("a successful contract attestation is required")
    if attestation.get("schema") != ATTESTATION_SCHEMA:
        raise AttestationError("unsupported contract attestation schema")
    if attestation.get("protocol") != EXECUTION_PROTOCOL:
        raise AttestationError("contract attestation uses a different execution protocol")
    expected_spec_hash = canonical_json_hash(
        {key: value for key, value in spec.items() if key != "execution_spec_sha256"}
    )
    if spec.get("execution_spec_sha256") != expected_spec_hash:
        raise AttestationError("execution spec hash is invalid")
    if attestation.get("execution_spec_sha256") != expected_spec_hash:
        raise AttestationError("contract attestation does not match the execution spec")
    expected_attestation_hash = canonical_json_hash(
        {key: value for key, value in attestation.items() if key != "attestation_sha256"}
    )
    if attestation.get("attestation_sha256") != expected_attestation_hash:
        raise AttestationError("contract attestation hash is invalid")
    if not attestation.get("passed"):
        raise AttestationError("contract attestation records a failed probe")
    return expected_attestation_hash


def require_request_matches_spec(
    request: Mapping[str, Any], spec: Mapping[str, Any]
) -> None:
    """Reject request/spec substitution before any provider environment is built."""

    expected_spec_hash = canonical_json_hash(
        {key: value for key, value in spec.items() if key != "execution_spec_sha256"}
    )
    if (
        spec.get("schema") != EXECUTION_SPEC_SCHEMA
        or spec.get("protocol") != EXECUTION_PROTOCOL
        or spec.get("execution_spec_sha256") != expected_spec_hash
    ):
        raise AttestationError("execution spec is malformed or has an invalid hash")
    provider = spec.get("provider", {})
    profile = spec.get("profile", {})
    constructor = spec.get("constructor", {})
    comparisons = {
        "provider": (request.get("provider"), provider.get("provider")),
        "adapter": (request.get("adapter"), provider.get("adapter")),
        "profile": (request.get("profile"), profile.get("id")),
        "shape": (int(request.get("shape", 0)), int(constructor.get("shape", 0))),
        "frame_skip": (
            int(request.get("frame_skip", constructor.get("frame_skip", 0))),
            int(constructor.get("frame_skip", 0)),
        ),
        "noop_reset_max": (
            int(request.get("noop_reset_max", 0)),
            int(constructor.get("noop_reset_max", 0)),
        ),
    }
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            raise AttestationError(
                f"runner request {name} does not match its execution spec: "
                f"{actual!r} != {expected!r}"
            )


def evidence_binding(attestation_sha256: str) -> dict[str, str]:
    return {
        "execution_protocol": EXECUTION_PROTOCOL,
        "contract_attestation_sha256": attestation_sha256,
    }
