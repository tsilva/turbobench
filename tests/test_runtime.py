from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from turbobench.providers import BUILTIN_PROVIDERS
from turbobench.resolution import fake_resolved, resolve_checkout
from turbobench.runner_client import _offline_environment
from turbobench.runtime import (
    _artifact_score,
    _portable_artifact,
    _probe_install,
    _snapshot_checkout,
    prepare_runtime,
    runtime_id,
    runtimes_are_isolated,
)


def test_install_probe_ignores_caller_import_shadow(tmp_path: Path, monkeypatch) -> None:
    caller = tmp_path / "caller"
    shadow = caller / "turbobench"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text(
        "raise RuntimeError('imported caller shadow')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(caller)
    provider = SimpleNamespace(
        distribution="turbobench-cli",
        import_name="turbobench",
    )

    probe = _probe_install(Path(sys.executable), provider)

    assert probe["import_file"] == "__init__.py"


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
    (dependency / ".gitignore").write_text("build/\n")
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
    (root / "pyproject.toml").write_text(
        '[project]\nname="env-breakoutatari2600-turbo-native"\nversion="1.0.0"\n'
    )
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
        BUILTIN_PROVIDERS["env-breakoutatari2600-turbo-native"],
        root,
        python_minor="3.14",
    )
    destination = tmp_path / "snapshot"
    _snapshot_checkout(resolved, root, destination)
    assert (destination / "benchmark.py").read_text() == "print('native')\n"
    assert (destination / "vendor/dependency/header.hpp").read_text() == "// pinned submodule\n"
    assert not (destination / ".git").exists()
    assert not (destination / "vendor/dependency/.git").exists()

    live_submodule = root / "vendor/dependency"
    (live_submodule / "header.hpp").write_text("// current work\n")
    (live_submodule / "new.hpp").write_text("// untracked current work\n")
    (live_submodule / "build").mkdir()
    (live_submodule / "build/cache.bin").write_bytes(b"ignored host output")
    dirty = resolve_checkout(
        BUILTIN_PROVIDERS["env-breakoutatari2600-turbo-native"],
        root,
        python_minor="3.14",
        allow_dirty=True,
    )
    dirty_destination = tmp_path / "dirty-snapshot"
    _snapshot_checkout(dirty, root, dirty_destination)
    assert dirty.artifact_sha256 != resolved.artifact_sha256
    assert (dirty_destination / "vendor/dependency/header.hpp").read_text() == "// current work\n"
    assert (dirty_destination / "vendor/dependency/new.hpp").read_text() == (
        "// untracked current work\n"
    )
    assert not (dirty_destination / "vendor/dependency/build").exists()
    assert not (dirty_destination / "vendor/dependency/.git").exists()


def test_dirty_checkout_snapshot_overlays_git_visible_files_only(tmp_path: Path) -> None:
    root = tmp_path / "provider"
    root.mkdir()
    (root / ".gitignore").write_text("CMakeCache.txt\nbuild/\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname="env-breakoutatari2600-turbo-native"\nversion="1.0.0"\n'
    )
    (root / "tracked.py").write_text("before\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "initial",
        ],
        cwd=root,
        check=True,
    )
    (root / "tracked.py").write_text("after\n")
    (root / "new.py").write_text("new\n")
    (root / "CMakeCache.txt").write_text("host-specific\n")
    (root / "build").mkdir()
    (root / "build" / "extension.so").write_bytes(b"host binary")
    resolved = resolve_checkout(
        BUILTIN_PROVIDERS["env-breakoutatari2600-turbo-native"],
        root,
        python_minor="3.14",
        allow_dirty=True,
    )
    destination = tmp_path / "snapshot"
    _snapshot_checkout(resolved, root, destination)
    assert (destination / "tracked.py").read_text() == "after\n"
    assert (destination / "new.py").read_text() == "new\n"
    assert not (destination / "CMakeCache.txt").exists()
    assert not (destination / "build").exists()
