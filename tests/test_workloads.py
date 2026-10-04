from copy import deepcopy

import pytest

from turbobench.policy_contracts import normalize_policy
from turbobench.profiles import profile_hash, profile_payload
from turbobench.proofs import finalize_proof, pack_policy_resolved, require_proof
from turbobench.util import canonical_json_hash, read_json, sha256_file, write_json
from turbobench.workloads import (
    check_declaration,
    comparison_definition,
    preliminary_workload,
    profile_from_workload,
    resolve_workload,
)


@pytest.fixture
def policy_files(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "model.zip").write_bytes(b"checkpoint-fixture")
    table = [["BUTTON"], ["RIGHT"], ["LEFT"]]
    env = {
        "env_id": "env:Breakout-Atari2600-v0",
        "state": "Start",
        "provider_args": {"use_restricted_actions": table, "use_fire_reset": False},
        "preprocessing": {
            "frame_skip": 2,
            "frame_stack": 4,
            "obs_resize": [84, 84],
            "obs_grayscale": True,
            "obs_resize_algorithm": "area",
            "max_pool_frames": False,
            "obs_crop": [17, 0, 0, 0],
            "obs_crop_mode": "mask",
            "obs_crop_fill": 0,
            "sticky_action_prob": 0,
        },
    }
    recipe = {
        "document_type": "gradlab.recipe",
        "format_version": 4,
        "recipe": {"policy_environment_hash": "sha256:fixture", "environment": env},
    }
    write_json(root / "recipe.json", recipe)
    action_contract = {"requested": {"table": table}}
    capture = {
        "schema": "gradlab.readme-playback/v1",
        "checkpoint_sha256": sha256_file(root / "model.zip"),
        "contract": {
            "matches_training": True,
            "frame_skip": {"training": 2, "playback": 2},
            "policy_environment_hash": "sha256:fixture",
            "training_policy_environment_hash": "sha256:fixture",
        },
        "frame_skip": 2,
        "initial": {"reset_info": {"noop_reset_count": 1}},
        "action_contract": action_contract,
        "selection": {"mode": "stochastic", "seed": 123},
        "transitions": [{"native_action": i} for i in [0, 1, 2, 0]],
    }
    model = {
        "document_type": "gradlab.model",
        "format_version": 3,
        "checkpoint": {"sha256": capture["checkpoint_sha256"], "size_bytes": 18, "step": 10},
        "recipe": {"sha256": sha256_file(root / "recipe.json")},
        "provenance": {
            "run_name": "test",
            "training_metadata": {
                "action_contract": action_contract,
                "policy_execution_contract": {
                    "model_inputs": {
                        "base_observation_space": {
                            "kind": "box",
                            "dtype": "|u1",
                            "shape": [4, 84, 84],
                        }
                    },
                    "task_wrapper": {"identity": "saved-task-fixture"},
                },
            },
        },
    }
    write_json(root / "model.json", model)
    write_json(root / "capture.json", capture)
    write_json(
        root / "actions.json",
        {
            "schema": "turbobench.imported-policy-actions/v1",
            "policy_frame_skip": 2,
            "reset_noop_prefix": 1,
            "actions": [[]] + [table[i] for i in [0, 0, 1, 1, 2, 2, 0, 0]],
        },
    )
    write_json(
        root / "provenance.json",
        {"mlflow_url": "https://tracking.example/test", "limitations": "fixture"},
    )
    return root


def test_firstwall_derived_settings_and_order(policy_files):
    contract = normalize_policy(policy_files)
    workload = preliminary_workload(contract)
    profile = profile_from_workload(workload)
    assert (profile.frame_skip, profile.frame_stack, profile.resize, profile.crop_top) == (
        2,
        4,
        (84, 84),
        17,
    )
    assert list(profile.action_table.values()) == [("BUTTON",), ("RIGHT",), ("LEFT",), ()]
    assert profile.semantic_actions == ("policy_0", "policy_1", "policy_2")
    assert profile.promo_steps == 9
    assert contract["execution_contract"]["task_wrapper"]["identity"] == "saved-task-fixture"
    assert profile_payload(profile) == workload
    assert profile_hash(profile) == canonical_json_hash(workload)


def test_changed_recipe_needs_no_new_profile(policy_files):
    original = preliminary_workload(normalize_policy(policy_files))
    recipe = read_json(policy_files / "recipe.json")
    recipe["recipe"]["environment"]["preprocessing"].update(
        frame_stack=2, obs_resize=[64, 64], obs_crop=[0, 0, 0, 0]
    )
    write_json(policy_files / "recipe.json", recipe)
    model = read_json(policy_files / "model.json")
    model["recipe"]["sha256"] = sha256_file(policy_files / "recipe.json")
    model["provenance"]["training_metadata"]["policy_execution_contract"]["model_inputs"][
        "base_observation_space"
    ]["shape"] = [2, 64, 64]
    write_json(policy_files / "model.json", model)
    changed = preliminary_workload(normalize_policy(policy_files))
    profile = profile_from_workload(changed)
    assert changed["id"] != original["id"]
    assert changed["definition"] == original["definition"]
    assert (profile.frame_stack, profile.resize, profile.crop_top) == (2, (64, 64), 0)


@pytest.mark.parametrize(
    "field,value", [("max_pool_frames", True), ("obs_resize_algorithm", "nearest")]
)
def test_unsupported_preprocessing_fails_before_timing(policy_files, field, value):
    policy = normalize_policy(policy_files)
    policy["environment"]["preprocessing"][field] = value
    with pytest.raises(ValueError, match="does not support"):
        preliminary_workload(policy)


def test_missing_training_layout_is_not_guessed(policy_files):
    model = read_json(policy_files / "model.json")
    del model["provenance"]["training_metadata"]["policy_execution_contract"]
    write_json(policy_files / "model.json", model)
    with pytest.raises(ValueError, match="explicitly establish CHW"):
        normalize_policy(policy_files)


def test_non_atari_definition_has_no_breakout_controls_or_crop_defaults(policy_files, monkeypatch):
    from turbobench import workloads

    contract = normalize_policy(policy_files)
    contract["environment"]["env_id"] = "env:Fixture-Game-v0"
    contract["environment"]["preprocessing"]["obs_crop"] = [0, 0, 0, 0]
    contract["environment"]["provider_args"]["use_restricted_actions"] = [[], ["MOVE"]]
    definition = deepcopy(comparison_definition("breakout/policy-v1"))
    definition.update(
        id="fixture/policy-v1",
        game="Fixture-Game-v0",
        logical_environment="fixture",
        providers=["fixture-upstream", "fixture-turbo"],
        authority="fixture-upstream",
        candidates=["fixture-turbo"],
        info_integer=[],
        allowed_representation_conversion="",
        assets={"required": False},
    )
    original_resource = workloads._resource
    monkeypatch.setattr(
        workloads,
        "_resource",
        lambda directory, identity: (
            definition
            if directory == "comparison_definitions"
            else original_resource(directory, identity)
        ),
    )
    protocol = original_resource("comparison_protocols", definition["protocol"])
    resolved = resolve_workload(definition, protocol, contract, {})
    profile = profile_from_workload(resolved)
    assert profile.game == "Fixture-Game-v0"
    assert profile.crop_top == 0
    assert profile.action_table == {"policy_0": (), "policy_1": ("MOVE",)}
    assert profile.info_integer == ()


def test_embedded_rules_cannot_weaken_required_checks(policy_files):
    workload = preliminary_workload(normalize_policy(policy_files))
    workload["definition"]["checks"] = []
    with pytest.raises(ValueError, match="trusted version"):
        profile_from_workload(workload)


def test_resolved_snapshot_cannot_override_source(policy_files):
    workload = preliminary_workload(normalize_policy(policy_files))
    workload["configuration"]["observation"]["frame_skip"] = 4
    with pytest.raises(ValueError, match="authoritative inputs"):
        profile_from_workload(workload)


def test_v2_policy_verification_rejects_binding_tamper_after_reinventory(policy_files, tmp_path):
    output = pack_policy_resolved(
        policy_files,
        policy_files / "capture.json",
        policy_files / "actions.json",
        tmp_path / "proof",
        None,
        "https://tracking.example/test",
        "fixture",
    )
    proof = require_proof(output)
    assert proof["schema"] == "turbobench.policy-proof/v2"
    assert proof["bindings"]["definition"] == "breakout/policy-v1"
    bindings = deepcopy(proof["bindings"])
    bindings["contract"]["environment"]["preprocessing"]["frame_skip"] = 4
    finalize_proof(output, proof["schema"], bindings)
    with pytest.raises(ValueError, match="policy"):
        require_proof(output)


def test_declaration_requires_valid_exact_artifact_attestation(policy_files):
    configuration = preliminary_workload(normalize_policy(policy_files))["configuration"]
    with pytest.raises(ValueError):
        check_declaration({"schema": "turbobench.environment-declaration/v1"}, configuration)


@pytest.fixture
def declaration(policy_files):
    from turbobench.lifecycle import EXECUTION_PROTOCOL, attest, execution_spec

    workload = preliminary_workload(normalize_policy(policy_files))
    config = workload["configuration"]
    observation = config["observation"]
    provider = {
        "provider": "stable-retro",
        "version": "1.0.1",
        "artifact_sha256": "artifact-fixture",
    }
    spec = execution_spec(
        provider=provider,
        harness={},
        python_minor="3.14",
        platform={},
        profile={"id": workload["id"], "sha256": canonical_json_hash(workload)},
        constructor={
            **observation,
            "crop": [observation["crop_top"], observation["crop_bottom"], 0, 0],
            "action_table": list(config["action_table"].values()),
            "states": config["states"],
        },
        assets=canonical_asset_record(),
    )
    attestation = attest(spec, {"passed": True, "promotable": True})
    return {
        "schema": "turbobench.environment-declaration/v1",
        "provider": provider,
        "capabilities": {
            "supported_observation_layouts": ["chw"],
            "supported_observation_color_modes": ["grayscale"],
            "supported_resize_algorithms": ["area"],
            "supported_crop_modes": ["mask"],
        },
        "states": ["Start"],
        "actions": list(config["action_table"].values()),
        "observation_shape": [4, 84, 84],
        "preflight": {
            "execution_spec": spec,
            "contract_attestation": attestation,
            "workload_executed": False,
            "lifecycle": {"environment_closed": True},
        },
        "lifecycle": {
            "execution_protocol": EXECUTION_PROTOCOL,
            "dynamic_contract_validation_calls": 0,
            "contract_attestation_sha256": attestation["attestation_sha256"],
            "environment_closed": True,
        },
    }


@pytest.mark.parametrize(
    "mutation", ["artifact", "states", "actions", "capabilities", "shape", "lifecycle"]
)
def test_mutated_provider_declarations_fail_before_timing(policy_files, declaration, mutation):
    config = preliminary_workload(normalize_policy(policy_files))["configuration"]
    check_declaration(declaration, config)
    if mutation == "artifact":
        declaration["provider"] = {**declaration["provider"], "artifact_sha256": "substituted"}
    elif mutation == "states":
        declaration["states"] = ["Other"]
    elif mutation == "actions":
        declaration["actions"] = [["WRONG"]]
    elif mutation == "capabilities":
        declaration["capabilities"]["supported_resize_algorithms"] = ["nearest"]
    elif mutation == "shape":
        declaration["observation_shape"] = [4, 42, 42]
    else:
        declaration["lifecycle"]["environment_closed"] = False
    with pytest.raises(ValueError):
        check_declaration(declaration, config)


def test_frozen_workload_is_independent_of_profile_registry(policy_files, declaration, monkeypatch):
    from turbobench import profiles

    preliminary = preliminary_workload(normalize_policy(policy_files))
    workload = resolve_workload(
        preliminary["definition"],
        preliminary["protocol"],
        preliminary["policy_contract"],
        {"left": declaration},
    )
    monkeypatch.setattr(
        profiles, "get_profile", lambda *args: pytest.fail("must not consult registry")
    )
    assert profile_from_workload(workload).frame_skip == 2


def test_actual_isolated_declaration_export_and_cache(policy_files, fake_assets):
    from turbobench.resolution import fake_resolved
    from turbobench.runtime import prepare_runtime
    from turbobench.workloads import probe_declarations

    profile = profile_from_workload(preliminary_workload(normalize_policy(policy_files)))
    providers = {"left": prepare_runtime(fake_resolved("stable-retro", speed=1))}
    first = probe_declarations(profile, providers, lambda message: None)
    second = probe_declarations(profile, providers, lambda message: None)
    assert first == second
    assert first["left"]["actions"] == [["BUTTON"], ["RIGHT"], ["LEFT"], []]
    assert first["left"]["lifecycle"]["environment_closed"] is True
    assert first["left"]["lifecycle"]["dynamic_contract_validation_calls"] == 0


def test_complete_v2_measurement_proof_dispatch_and_workload_binding(
    policy_files, tmp_path, monkeypatch, fake_assets
):
    from turbobench import workflow
    from turbobench.bundle import verify_bundle
    from turbobench.engine import ComparisonOptions, run_comparison_resolved
    from turbobench.resolution import fake_resolved
    from turbobench.runtime import harness_source_hash, prepare_runtime
    from turbobench.workloads import probe_declarations

    policy = normalize_policy(policy_files)
    preliminary = preliminary_workload(policy)
    providers = {
        "left": prepare_runtime(fake_resolved("stable-retro", speed=1)),
        "right": prepare_runtime(fake_resolved("env-breakoutatari2600-turbo-native", speed=2)),
    }
    declarations = probe_declarations(
        profile_from_workload(preliminary), providers, lambda message: None
    )
    workload = resolve_workload(
        preliminary["definition"], preliminary["protocol"], policy, declarations
    )
    request = {
        "schema": "turbobench.comparison-request/v2",
        "request_id": "",
        "profile": workload["id"],
        "left": f"stable-retro@{providers['left'].version}",
        "right": f"env-breakoutatari2600-turbo-native@{providers['right'].version}",
        "python_minor": "3.14",
        "smoke": True,
        "policy_id": "policy-fixture",
        "actions": read_json(policy_files / "actions.json")["actions"],
        "policy_contract": policy,
        "render_machine": {"machine_sha256": "render-host", "hardware": {}},
        "harness_sha256": harness_source_hash(),
        "resolved_workload": workload,
    }
    request["request_id"] = canonical_json_hash(request)
    monkeypatch.setattr(
        workflow, "machine_identity", lambda: {"machine_sha256": "benchmark-host", "hardware": {}}
    )
    root, _ = run_comparison_resolved(
        profile_from_workload(workload),
        providers["left"],
        providers["right"],
        tmp_path / "benchmark",
        ComparisonOptions(smoke=True, measurement_only=True, workflow_request=request),
        private_assets={},
        portable_assets=fake_assets,
    )
    assert verify_bundle(root)["passed"]
    proof = require_proof(root)
    assert proof["schema"] == "turbobench.benchmark-proof/v2"
    assert read_json(root / "resolved-workload.json") == workload
    bindings = {**proof["bindings"], "workload_sha256": "substituted"}
    finalize_proof(root, proof["schema"], bindings)
    assert not verify_bundle(root)["passed"]


def test_changed_saved_cadence_and_recorded_expansion_resolve_together(policy_files):
    original = preliminary_workload(normalize_policy(policy_files))
    recipe = read_json(policy_files / "recipe.json")
    recipe["recipe"]["environment"]["preprocessing"]["frame_skip"] = 4
    write_json(policy_files / "recipe.json", recipe)
    model = read_json(policy_files / "model.json")
    model["recipe"]["sha256"] = sha256_file(policy_files / "recipe.json")
    write_json(policy_files / "model.json", model)
    capture = read_json(policy_files / "capture.json")
    capture["frame_skip"] = 4
    capture["contract"]["frame_skip"] = {"training": 4, "playback": 4}
    write_json(policy_files / "capture.json", capture)
    table = recipe["recipe"]["environment"]["provider_args"]["use_restricted_actions"]
    actions = read_json(policy_files / "actions.json")
    actions.update(
        policy_frame_skip=4,
        actions=[[]]
        + [table[t["native_action"]] for t in capture["transitions"] for _ in range(4)],
    )
    write_json(policy_files / "actions.json", actions)
    changed = preliminary_workload(normalize_policy(policy_files))
    assert profile_from_workload(changed).frame_skip == 4
    assert profile_from_workload(changed).promo_steps == 17
    assert changed["id"] != original["id"]
    assert changed["definition"] == original["definition"]


def canonical_asset_record():
    assets = comparison_definition("breakout/policy-v1")["assets"]
    return {
        "required": True,
        "available": True,
        "assets": [
            {"role": "game-payload", "sha256": assets["game_payload_sha256"]},
            {"role": "state", "id": "Start", "sha256": assets["states"]["Start"]},
        ],
    }


@pytest.fixture
def fake_assets(monkeypatch):
    from turbobench import assets

    portable = canonical_asset_record()
    monkeypatch.setattr(assets, "discover_assets", lambda profile: ({}, portable))
    return portable


def test_derived_workload_keeps_canonical_state_commitment(policy_files, tmp_path, monkeypatch):
    from turbobench import assets

    profile = profile_from_workload(preliminary_workload(normalize_policy(policy_files)))
    ambient = tmp_path / "ambient"
    canonical = tmp_path / "canonical"
    for root in (ambient, canonical):
        root.mkdir()
        (root / "rom.a26").write_bytes(b"fixture-rom")
        (root / "Start.state").write_bytes(b"fixture-state")
    monkeypatch.setattr(assets, "_find_game_dirs", lambda profile: [ambient, canonical])
    expected = comparison_definition("breakout/policy-v1")["assets"]
    monkeypatch.setattr(
        assets,
        "sha256_file",
        lambda path: (
            expected["game_payload_sha256"]
            if path.name == "rom.a26"
            else expected["states"]["Start"]
            if path.parent == canonical
            else "wrong-state"
        ),
    )
    private, portable = assets.discover_assets(profile)
    assert private["state_paths"]["Start"] == str(canonical / "Start.state")
    assert portable["available"] is True


def test_declaration_cannot_use_noncanonical_start_state(policy_files, declaration):
    config = preliminary_workload(normalize_policy(policy_files))["configuration"]
    from turbobench.lifecycle import attest

    spec = declaration["preflight"]["execution_spec"]
    spec["assets"]["assets"][1]["sha256"] = "wrong-state"
    spec["execution_spec_sha256"] = canonical_json_hash(
        {k: v for k, v in spec.items() if k != "execution_spec_sha256"}
    )
    attestation = attest(spec, {"passed": True, "promotable": True})
    declaration["preflight"]["contract_attestation"] = attestation
    declaration["lifecycle"]["contract_attestation_sha256"] = attestation["attestation_sha256"]
    with pytest.raises(ValueError, match="canonical comparison assets"):
        check_declaration(
            declaration, config, comparison_definition("breakout/policy-v1")["assets"]
        )


def test_json_round_trip_preserves_large_policy_action_order(policy_files, tmp_path):
    policy = normalize_policy(policy_files)
    table = [[f"CONTROL_{i}"] for i in range(12)]
    policy["environment"]["provider_args"]["use_restricted_actions"] = table
    policy["action_contract"]["requested"]["table"] = table
    workload = preliminary_workload(policy)
    write_json(tmp_path / "workload.json", workload)
    restored = profile_from_workload(read_json(tmp_path / "workload.json"))
    assert list(restored.action_table.values()) == [tuple(row) for row in table] + [()]
    assert restored.semantic_actions == tuple(f"policy_{i}" for i in range(12))


def test_policy_timing_uses_locked_decisions_and_rejects_substitution(policy_files):
    from turbobench.profiles import canonical_actions
    from turbobench.workloads import policy_benchmark_settings

    contract = normalize_policy(policy_files)
    contract["environment"]["provider_args"]["noop_reset_max"] = 1
    actions = read_json(policy_files / "actions.json")["actions"]
    workload = preliminary_workload(contract, policy_actions=actions)
    profile = profile_from_workload(workload)
    assert workload["schema"] == "turbobench.resolved-workload/v2"
    assert profile.measurement_steps == 4
    assert canonical_actions(profile, 2).tolist() == [[0, 0], [1, 1], [2, 2], [0, 0]]
    assert policy_benchmark_settings(profile) == {
        "seed": 123,
        "noop_reset_max": 1,
        "replicate_initial_seed": True,
    }
    with pytest.raises(ValueError, match="entire locked"):
        canonical_actions(profile, 2, 3)
    changed = deepcopy(workload)
    changed["policy_actions"][3] = ["LEFT"]
    with pytest.raises(ValueError, match="digest"):
        profile_from_workload(changed)
    changed = deepcopy(workload)
    changed["configuration"]["run"]["seed"] += 1
    with pytest.raises(ValueError, match="authoritative"):
        profile_from_workload(changed)


def test_policy_timing_rejects_seed_or_effective_cadence_mismatch(policy_files):
    from turbobench.workloads import policy_decisions

    contract = normalize_policy(policy_files)
    actions = read_json(policy_files / "actions.json")["actions"]
    with pytest.raises(ValueError, match="reset seed"):
        preliminary_workload(contract, policy_actions=actions)
    contract["environment"]["provider_args"]["noop_reset_max"] = 1
    actions[2] = ["RIGHT"]
    contract["action_sha256"] = canonical_json_hash(actions)
    with pytest.raises(ValueError, match="cadence"):
        policy_decisions(contract, actions)


def test_future_workflows_scale_but_frozen_v1_workloads_keep_their_shapes(policy_files):
    from turbobench.workloads import _resource

    contract = normalize_policy(policy_files)
    for policy_actions in (None, read_json(policy_files / "actions.json")["actions"]):
        if policy_actions is not None:
            contract["environment"]["provider_args"]["noop_reset_max"] = 1
        future = preliminary_workload(contract, policy_actions=policy_actions)
        assert future["protocol"]["id"].endswith("/v2")
        assert profile_from_workload(future).measurement_shapes == tuple(2**i for i in range(11))
        legacy_id = "paired-policy/v1" if policy_actions is not None else "paired-environment/v1"
        legacy = resolve_workload(
            future["definition"],
            _resource("comparison_protocols", legacy_id),
            contract,
            {},
            policy_actions,
        )
        assert profile_from_workload(legacy).measurement_shapes == (1, 16, 32)
        assert "scaling" not in legacy["protocol"]
        changed = deepcopy(future)
        changed["protocol"]["scaling"]["minimum_gain"] = 0.5
        with pytest.raises(ValueError, match="trusted version"):
            profile_from_workload(changed)


def test_adaptive_engine_stops_before_probing_unused_counts_and_rejects_forged_stop(
    policy_files, tmp_path, monkeypatch, fake_assets
):
    from turbobench import engine, workflow
    from turbobench.bundle import verify_bundle
    from turbobench.engine import ComparisonOptions, run_comparison_resolved
    from turbobench.reporting import render_report
    from turbobench.resolution import fake_resolved
    from turbobench.runtime import harness_source_hash, prepare_runtime
    from turbobench.workflow import verify_sampling
    from turbobench.workloads import probe_declarations

    contract = normalize_policy(policy_files)
    contract["environment"]["provider_args"]["noop_reset_max"] = 1
    actions = read_json(policy_files / "actions.json")["actions"]
    preliminary = preliminary_workload(contract, policy_actions=actions)
    providers = {
        "left": prepare_runtime(fake_resolved("stable-retro", speed=1)),
        "right": prepare_runtime(fake_resolved("env-breakoutatari2600-turbo-native", speed=2)),
    }
    declarations = probe_declarations(profile_from_workload(preliminary), providers, lambda _: None)
    workload = resolve_workload(
        preliminary["definition"], preliminary["protocol"], contract, declarations, actions
    )
    profile = profile_from_workload(workload)
    request = {
        "schema": "turbobench.comparison-request/v3",
        "request_id": "",
        "profile": profile.id,
        "resolved_workload": workload,
        "smoke": False,
        "left": f"stable-retro@{providers['left'].version}",
        "right": f"env-breakoutatari2600-turbo-native@{providers['right'].version}",
        "python_minor": "3.14",
        "policy_id": "policy-fixture",
        "actions": actions,
        "policy_contract": contract,
        "render_machine": {"machine_sha256": "render-host", "hardware": {}},
        "harness_sha256": harness_source_hash(),
    }
    request["request_id"] = canonical_json_hash(request)
    monkeypatch.setattr(
        workflow, "machine_identity", lambda: {"machine_sha256": "benchmark-host", "hardware": {}}
    )
    invocation = engine._benchmark_invocation
    measured = []
    interrupted = False

    def capped_invocation(*args, **kwargs):
        nonlocal interrupted
        provider, count = args[1], args[3]
        if count == 4 and not interrupted:
            interrupted = True
            raise RuntimeError("fixture interrupted before shape 4 timing")
        response = invocation(*args, **kwargs)
        # Upstream saturates at 2, candidate at 8; continue until both are flat twice.
        limit = 2 if provider.provider == "stable-retro" else 8
        response["sps"] = [
            float(min(count, limit) * float(provider.source_identity.rsplit(":", 1)[1]) * 100)
        ] * 3
        measured.append(count)
        return response

    monkeypatch.setattr(engine, "_benchmark_invocation", capped_invocation)
    monkeypatch.setattr(
        engine, "wait_for_load", lambda **kwargs: {"passed": True, "forced": False, "threshold": 1}
    )
    options = ComparisonOptions(measurement_only=True, workflow_request=request)
    with pytest.raises(RuntimeError, match="fixture interrupted"):
        run_comparison_resolved(
            profile,
            providers["left"],
            providers["right"],
            tmp_path / "adaptive",
            options,
            private_assets={},
            portable_assets=fake_assets,
        )
    completed_before_resume = list(measured)
    assert set(completed_before_resume) == {1, 2}
    root, result = run_comparison_resolved(
        profile,
        providers["left"],
        providers["right"],
        tmp_path / "adaptive",
        options,
        private_assets={},
        portable_assets=fake_assets,
    )
    assert set(result["comparison"]["shapes"]) == {"1", "2", "4", "8", "16", "32"}
    assert all(count >= 4 for count in measured[len(completed_before_resume) :])
    assert require_proof(root)["schema"] == "turbobench.benchmark-proof/v3"
    assert max(measured) == 32
    assert set(result["contract_attestations"]["left"]) == set(result["comparison"]["shapes"])
    assert result["scaling"]["complete"]
    assert "both_providers_saturated" in render_report(result)
    assert verify_bundle(root)["passed"]
    verify_sampling(root, request, result)
    # Re-signing a changed stopping record still fails semantic verification.
    result["scaling"]["stop_reason"] = "safety_cap"
    write_json(root / "result.json", result)
    write_json(root / "verification" / "scaling.json", result["scaling"])
    proof = read_json(root / "manifest.json")
    finalize_proof(root, proof["schema"], proof["bindings"])
    assert not verify_bundle(root)["passed"]
    with pytest.raises(ValueError, match="inconsistent"):
        verify_sampling(root, request, result)
