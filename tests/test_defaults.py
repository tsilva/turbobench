from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from turbobench import defaults
from turbobench.cli import build_parser, main
from turbobench.util import read_json, write_json


@pytest.fixture
def local_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("TURBOBENCH_BENCHMARK_HOST", raising=False)
    monkeypatch.chdir(tmp_path)
    package = tmp_path / "policy"
    write_json(
        package / "recipe.json",
        {
            "recipe": {
                "environment": {
                    "env_id": "env-breakoutatari2600-turbo-native:Breakout-Atari2600-v0"
                }
            }
        },
    )
    write_json(
        package / "model.json",
        {
            "provenance": {
                "training_metadata": {"versions": {"env_breakoutatari2600_turbo_native": "0.5.13"}}
            }
        },
    )
    monkeypatch.setattr(
        defaults,
        "_policy",
        lambda path: {
            "schema": "turbobench.policy-proof/v1",
            "proof_id": "policy-id",
            "bindings": {"profile": "breakout/firstwall-policy-v1"},
        },
    )
    return package


def args(*values: str) -> argparse.Namespace:
    return build_parser().parse_args(["compare", *values])


def test_one_time_configuration_resolves_short_command(local_policy: Path) -> None:
    defaults.configure("benchmark.example.com", local_policy)
    invocation = args("--showcase", "--smoke")
    defaults.apply_comparison_defaults(invocation)
    assert invocation.profile == "breakout/firstwall-policy-v1"
    assert invocation.left == "stable-retro@1.0.1"
    assert invocation.right == "env-breakoutatari2600-turbo-native@0.5.13"
    assert invocation.policy == local_policy.resolve()
    assert invocation.benchmark_host == "benchmark.example.com"
    assert invocation.render_host == "local"
    assert invocation.smoke
    assert read_json(defaults.config_path())["policy_id"] == "policy-id"


def test_flags_then_environment_override_config(
    local_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults.configure("configured.example.com", local_policy)
    monkeypatch.setenv("TURBOBENCH_BENCHMARK_HOST", "environment.example.com")
    invocation = args("--showcase")
    defaults.apply_comparison_defaults(invocation)
    assert invocation.benchmark_host == "environment.example.com"
    explicit = args(
        "--showcase",
        "--benchmark-host",
        "flag.example.com",
        "--right",
        "env-breakoutatari2600-turbo-native@latest",
        "--output",
        "custom",
    )
    defaults.apply_comparison_defaults(explicit)
    assert explicit.benchmark_host == "flag.example.com"
    assert explicit.right.endswith("@latest")
    assert explicit.output == Path("custom")


def test_changed_configured_policy_is_rejected(
    local_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults.configure("benchmark.example.com", local_policy)
    monkeypatch.setattr(
        defaults,
        "_policy",
        lambda path: {
            "schema": "turbobench.policy-proof/v1",
            "proof_id": "replacement-id",
            "bindings": {"profile": "breakout/firstwall-policy-v1"},
        },
    )
    with pytest.raises(ValueError, match="configured policy changed"):
        defaults.apply_comparison_defaults(args("--showcase"))


def test_discovery_does_not_choose_between_different_proofs(
    local_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("a", "b"):
        write_json(Path("turbobench-results/policies") / name / "manifest.json", {})
    monkeypatch.setattr(
        defaults,
        "_policy",
        lambda path: {
            "proof_id": path.name,
            "bindings": {"profile": "breakout/firstwall-policy-v1"},
        },
    )
    with pytest.raises(ValueError, match="multiple verified"):
        defaults._discover_policy("breakout/firstwall-policy-v1")
    monkeypatch.setattr(
        defaults,
        "_policy",
        lambda path: {
            "proof_id": "identical",
            "bindings": {"profile": "breakout/firstwall-policy-v1"},
        },
    )
    path, proof = defaults._discover_policy(None)
    assert path.name == "a"
    assert proof["proof_id"] == "identical"


def test_discovery_skips_invalid_and_unrelated_packages(
    local_policy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("invalid", "unrelated", "matching"):
        write_json(Path("turbobench-results/policies") / name / "manifest.json", {})

    def verify(path: Path) -> dict:
        if path.name == "invalid":
            raise ValueError("integrity failed")
        return {
            "proof_id": path.name,
            "bindings": {
                "profile": "breakout/firstwall-policy-v1" if path.name == "matching" else "other"
            },
        }

    monkeypatch.setattr(defaults, "_policy", verify)
    assert defaults._discover_policy("breakout/firstwall-policy-v1")[0].name == "matching"


def test_missing_training_version_requires_explicit_candidate(local_policy: Path) -> None:
    defaults.configure("benchmark.example.com", local_policy)
    write_json(local_policy / "model.json", {"provenance": {"training_metadata": {"versions": {}}}})
    with pytest.raises(ValueError, match="no saved training-provider version"):
        defaults.apply_comparison_defaults(args("--showcase"))
    explicit = args("--showcase", "--right", "env-breakoutatari2600-turbo-native@latest")
    defaults.apply_comparison_defaults(explicit)
    assert explicit.right.endswith("@latest")


def test_short_command_passes_resolved_defaults_to_workflow(
    local_policy: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    defaults.configure("benchmark.example.com", local_policy)
    seen = []

    def run(invocation, output, progress):
        assert invocation.profile == "breakout/firstwall-policy-v1"
        assert invocation.right.endswith("@0.5.13")
        seen.append(output)
        return output

    monkeypatch.setattr("turbobench.workflow.run_workflow", run)
    monkeypatch.setattr(
        "turbobench.publication.export_publication",
        lambda path: path.with_name(path.name + "-publication"),
    )
    assert main(["compare", "--showcase"]) == 0
    assert main(["compare", "--showcase"]) == 0
    assert seen[0] != seen[1]
    assert all(path.parent == Path("turbobench-results") for path in seen)
    assert '"pipeline_passed": true' in capsys.readouterr().out


def test_local_comparison_uses_profile_defaults_without_local_config(local_policy: Path) -> None:
    write_json(defaults.config_path(), {"schema": "unsupported"})
    invocation = args("vizdoom/basic-v1")
    defaults.apply_comparison_defaults(invocation)
    assert invocation.left == "vizdoom@1.3.0"
    assert invocation.right == "env-vizdoom-turbo@latest"
    with pytest.raises(ValueError, match="multiple candidates"):
        defaults.apply_comparison_defaults(args("breakout/start-v1"))


def test_invalid_host_does_not_replace_existing_defaults(local_policy: Path) -> None:
    defaults.configure("benchmark.example.com", local_policy)
    before = defaults.config_path().read_bytes()
    with pytest.raises(ValueError, match="hostname or alias"):
        defaults.configure("-oProxyCommand=bad", None)
    assert defaults.config_path().read_bytes() == before


def test_explicit_profile_must_match_policy(local_policy: Path) -> None:
    with pytest.raises(ValueError, match="does not match"):
        defaults.apply_comparison_defaults(
            args(
                "breakout/start-v1",
                "--showcase",
                "--policy",
                str(local_policy),
                "--benchmark-host",
                "benchmark.example.com",
            )
        )


def test_policy_benchmark_uses_pinned_selection_without_media(local_policy: Path, monkeypatch):
    defaults.configure("benchmark.example.com", local_policy)
    invocation = args("--policy-benchmark")
    defaults.apply_comparison_defaults(invocation)
    assert invocation.policy == local_policy.resolve()
    assert invocation.benchmark_host == "benchmark.example.com"
    assert invocation.right.endswith("@0.5.13")
    seen = []

    def run(invocation, output, progress):
        assert invocation.policy_benchmark
        assert not invocation.showcase
        seen.append(invocation.policy)
        return output

    monkeypatch.setattr("turbobench.workflow.run_workflow", run)
    monkeypatch.setattr(
        "turbobench.publication.export_publication",
        lambda path: path.with_name(path.name + "-publication"),
    )
    assert main(["compare", "--policy", str(local_policy)]) == 0
    assert seen == [local_policy]
    monkeypatch.setattr(
        defaults,
        "_policy",
        lambda path: {
            "schema": "turbobench.policy-proof/v1",
            "proof_id": "changed",
            "bindings": {"profile": "breakout/firstwall-policy-v1"},
        },
    )
    with pytest.raises(ValueError, match="configured policy changed"):
        defaults.apply_comparison_defaults(args("--policy-benchmark"))


def test_policy_timing_can_generate_a_showcase(local_policy, monkeypatch):
    defaults.configure("benchmark.example.com", local_policy)
    seen = []

    def run(invocation, output, progress):
        assert invocation.showcase and invocation.policy_benchmark
        seen.append(invocation.policy)
        return output

    monkeypatch.setattr("turbobench.workflow.run_workflow", run)
    monkeypatch.setattr(
        "turbobench.publication.export_publication",
        lambda path: path.with_name(path.name + "-publication"),
    )
    assert main(["compare", "--policy-benchmark", "--showcase"]) == 0
    assert seen == [local_policy]
