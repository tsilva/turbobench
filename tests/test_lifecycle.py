from __future__ import annotations

import pytest

from turbobench.bundle import verify_bundle
from turbobench.engine import ComparisonOptions, run_comparison_resolved
from turbobench.lifecycle import (
    EXECUTION_PROTOCOL,
    AttestationError,
    attest,
    execution_spec,
    require_attestation,
    require_request_matches_spec,
)
from turbobench.profiles import get_profile
from turbobench.resolution import fake_resolved
from turbobench.runtime import prepare_runtime
from turbobench.util import read_json


def _spec(*, shape: int = 1, frame_skip: int = 4) -> dict:
    return execution_spec(
        provider={
            "provider": "fake",
            "version": "1.0",
            "adapter": "fake",
            "artifact_sha256": "a" * 64,
            "runtime_id": "runtime-1",
        },
        harness={"version": "2.1.0", "source_sha256": "b" * 64},
        python_minor="3.11",
        platform={"os": "Linux", "os_release": "test", "architecture": "x86_64"},
        profile={"id": "supermario/world1-v1", "sha256": "c" * 64},
        constructor={"shape": shape, "frame_skip": frame_skip},
        assets={"required": False, "available": True, "assets": []},
    )


def test_attestation_is_bound_to_the_exact_portable_execution_spec() -> None:
    spec = _spec(shape=16)
    contract = attest(spec, {"passed": True, "promotable": True})

    assert contract["protocol"] == EXECUTION_PROTOCOL
    assert require_attestation(spec, contract) == contract["attestation_sha256"]

    with pytest.raises(AttestationError, match="execution spec"):
        require_attestation(_spec(shape=32), contract)


def test_missing_or_failed_attestations_are_rejected() -> None:
    spec = _spec()

    with pytest.raises(AttestationError, match="required"):
        require_attestation(spec, None)
    with pytest.raises(AttestationError, match="failed"):
        require_attestation(spec, attest(spec, {"passed": False, "promotable": False}))


def test_execution_spec_rejects_local_asset_paths() -> None:
    with pytest.raises(ValueError, match="portable"):
        execution_spec(
            provider={"provider": "fake", "artifact_sha256": "a" * 64},
            harness={"version": "2.1.0", "source_sha256": "b" * 64},
            python_minor="3.11",
            platform={"os": "Linux", "architecture": "x86_64"},
            profile={"id": "profile", "sha256": "c" * 64},
            constructor={"shape": 1},
            assets={"rom_path": "/private/game.rom"},
        )


def test_runner_request_must_match_the_attested_shape_and_constructor_configuration() -> None:
    spec = _spec(shape=16, frame_skip=4)
    request = {
        "provider": "fake",
        "adapter": "fake",
        "profile": "supermario/world1-v1",
        "shape": 16,
        "frame_skip": 4,
        "noop_reset_max": 0,
    }
    require_request_matches_spec(request, spec)

    with pytest.raises(AttestationError, match="shape"):
        require_request_matches_spec({**request, "shape": 32}, spec)


def test_poisoning_contracts_are_process_and_instance_isolated_from_every_workload(
    tmp_path,
) -> None:
    profile = get_profile("supermario/world1-v1")
    left = prepare_runtime(fake_resolved("fake-poison-left", speed=1.0))
    right = prepare_runtime(fake_resolved("fake-poison-right", speed=2.0))
    bundle, _result = run_comparison_resolved(
        profile,
        left,
        right,
        tmp_path / "poison-bundle",
        ComparisonOptions(quick=True, shapes=(1, 4)),
        private_assets={},
        portable_assets={"required": False, "available": True, "assets": []},
    )

    assert verify_bundle(bundle)["passed"]
    contract_processes: set[int] = set()
    contract_instances: set[str] = set()
    for path in (bundle / "verification" / "attestations").glob("*.json"):
        response = read_json(path)
        contract_processes.add(response["runner"]["pid"])
        contract_instances.add(response["lifecycle"]["environment_instance_id"])
        assert response["lifecycle"]["environment_closed"]
        assert response["lifecycle"]["process_global_poisoned"]
        assert response["lifecycle"]["instance_poisoned"]

    seen_shapes: set[int] = set()
    for shape in (1, 4):
        for path in (bundle / "raw" / f"shape-{shape}").glob("*.json"):
            if path.name == "pairs.json":
                continue
            evidence = read_json(path)
            lifecycle = evidence["lifecycle"]
            seen_shapes.add(shape)
            assert evidence["runner"]["pid"] not in contract_processes
            assert lifecycle["environment_instance_id"] not in contract_instances
            assert lifecycle["environment_closed"]
            assert not lifecycle["process_poisoned_at_construction"]
            assert lifecycle["dynamic_contract_validation_calls"] == 0
            if path.name.startswith("trace-"):
                assert lifecycle["render_calls"] > 0
                assert lifecycle["instance_poisoned"]
            else:
                assert lifecycle["render_calls"] == 0
                assert not lifecycle["instance_poisoned"]
                expected = 10_000.0 * (2.0 if "right" in path.stem else 1.0) * shape**0.2
                assert evidence["sps"] == [expected * 0.999, expected, expected * 1.001]
    assert seen_shapes == {1, 4}


def test_contract_failure_prevents_all_dependent_workloads(tmp_path) -> None:
    profile = get_profile("supermario/world1-v1")
    left = prepare_runtime(fake_resolved("fake-contract-failure"))
    right = prepare_runtime(fake_resolved("fake-ok"))

    with pytest.raises(RuntimeError, match="no dependent workload was executed"):
        run_comparison_resolved(
            profile,
            left,
            right,
            tmp_path / "failed-bundle",
            ComparisonOptions(quick=True),
            private_assets={},
            portable_assets={"required": False, "available": True, "assets": []},
        )

    partial = tmp_path / "failed-bundle.partial"
    assert not list((partial / "raw").glob("shape-*"))
