from __future__ import annotations

import subprocess
from pathlib import Path

from turbobench.providers import BUILTIN_PROVIDERS
from turbobench.resolution import fake_resolved, resolve_checkout
from turbobench.runner_client import _offline_environment
from turbobench.runtime import (
    _artifact_score,
    _portable_artifact,
    _snapshot_checkout,
    prepare_runtime,
    runtime_id,
    runtimes_are_isolated,
)


def test_runtime_identity_is_content_addressed() -> None:
    first = fake_resolved("first", speed=1.0)
    same = fake_resolved("first", speed=1.0)
    other = fake_resolved("first", speed=2.0)
    assert runtime_id(first) == runtime_id(same)
    assert runtime_id(first) != runtime_id(other)
    prepared = prepare_runtime(first)
    assert prepared.runtime_id == runtime_id(first)
    assert prepared.runtime_python


def test_runtime_isolation_requires_distinct_python_paths() -> None:
    first = prepare_runtime(fake_resolved("first"))
    second = prepare_runtime(fake_resolved("second"))
    assert not runtimes_are_isolated(first, second)


def test_offline_runner_does_not_inherit_host_retro_data_path(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RETRO_DATA_PATH", str(tmp_path / "host-integrations"))
    environment = _offline_environment(tmp_path)
    assert "RETRO_DATA_PATH" not in environment
    assert environment["TURBOBENCH_NETWORK_DISABLED"] == "1"


def test_artifact_selection_prefers_matching_cp314_arm64_or_universal(monkeypatch) -> None:
    provider = fake_resolved("provider")
    monkeypatch.setattr("turbobench.runtime.sys.platform", "darwin")
    monkeypatch.setattr("turbobench.runtime.platform.machine", lambda: "arm64")
    universal = {"packagetype": "bdist_wheel", "filename": "p-1-py3-none-any.whl"}
    native = {"packagetype": "bdist_wheel", "filename": "p-1-cp314-cp314-macosx_14_0_arm64.whl"}
    wrong = {"packagetype": "bdist_wheel", "filename": "p-1-cp311-cp311-manylinux_x86_64.whl"}
    sdist = {"packagetype": "sdist", "filename": "p-1.tar.gz"}
    assert _artifact_score(provider, native)[0] == 0
    assert _artifact_score(provider, universal)[0] == 0
    assert _artifact_score(provider, sdist)[0] < _artifact_score(provider, wrong)[0]


def test_selected_artifact_metadata_is_portable_and_exact() -> None:
    item = {
        "filename": "provider-1-py3-none-any.whl",
        "packagetype": "bdist_wheel",
        "python_version": "py3",
        "sha256": "a" * 64,
        "size": 123,
        "url": "https://files.pythonhosted.org/secret/cache/path.whl",
    }
    assert _portable_artifact(item) == {
        "filename": item["filename"],
        "packagetype": "bdist_wheel",
        "python_version": "py3",
        "sha256": "a" * 64,
        "size": 123,
    }


def test_clean_checkout_snapshot_preserves_provider_scripts_and_excludes_live_git(tmp_path: Path) -> None:
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    (dependency / "header.hpp").write_text("// pinned submodule\n")
    subprocess.run(["git", "init", "-q"], cwd=dependency, check=True)
    subprocess.run(["git", "add", "."], cwd=dependency, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
        cwd=dependency,
        check=True,
    )
    root = tmp_path / "provider"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="breakout-turbo-env"\nversion="1.0.0"\n')
    (root / "benchmark.py").write_text("print('native')\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            "-q",
            str(dependency),
            "vendor/dependency",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
        cwd=root,
        check=True,
    )
    resolved = resolve_checkout(
        BUILTIN_PROVIDERS["breakout-turbo-env"], root, python_minor="3.14"
    )
    destination = tmp_path / "snapshot"
    _snapshot_checkout(resolved, root, destination)
    assert (destination / "benchmark.py").read_text() == "print('native')\n"
    assert (destination / "vendor/dependency/header.hpp").read_text() == "// pinned submodule\n"
    assert not (destination / ".git").exists()
    assert not (destination / "vendor/dependency/.git").exists()
