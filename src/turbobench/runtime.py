"""Content-addressed isolated provider runtime construction."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from turbobench.model import ResolvedProvider
from turbobench.resolution import with_runtime
from turbobench.util import canonical_json_hash, read_json, sha256_file, write_json

HARNESS_REQUIREMENTS = ("gymnasium==1.2.2", "numpy==2.4.2")


def cache_root() -> Path:
    configured = os.environ.get("TURBOBENCH_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "turbobench" / "v1").resolve()


def harness_source_hash() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and item.suffix in {".py", ".toml"}
    ):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def runtime_id(
    provider: ResolvedProvider,
    *,
    cache_context: str | None = None,
    installed_lock: tuple[str, ...] = (),
) -> str:
    provider_identity = provider.portable()
    provider_identity.pop("runtime_id", None)
    provider_identity.pop("installed_lock", None)
    provider_identity.pop("selected_artifact", None)
    return canonical_json_hash(
        {
            "provider": provider_identity,
            "python_minor": provider.python_minor,
            "host": {"system": platform.system(), "machine": platform.machine()},
            "harness_source_sha256": harness_source_hash(),
            "harness_requirements": HARNESS_REQUIREMENTS,
            "cache_context": cache_context,
            "installed_lock": list(installed_lock),
        }
    )


def prepare_runtime(
    provider: ResolvedProvider,
    *,
    checkout_path: Path | None = None,
    artifact_path: Path | None = None,
    cache_context: str | None = None,
    uv: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> ResolvedProvider:
    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    if provider.source_kind == "fake":
        return with_runtime(
            provider,
            runtime_id=runtime_id(provider, cache_context=cache_context),
            runtime_python=sys.executable,
            installed_lock=(f"turbobench==fake:{provider.source_identity}",),
        )
    uv_path = uv or shutil.which("uv")
    if not uv_path:
        raise RuntimeError("uv is required to resolve provider runtimes")
    identifier = runtime_id(provider, cache_context=cache_context)
    root = cache_root() / "runtimes" / identifier
    metadata_path = root / "runtime.json"
    python = _venv_python(root)
    if metadata_path.is_file() and python.is_file():
        metadata = read_json(metadata_path)
        if metadata.get("cache_key") == identifier and metadata.get("complete"):
            report(f"Using cached runtime for {provider.provider}")
            selected_artifact = metadata.get("selected_artifact")
            if provider.source_kind == "pypi" and not selected_artifact:
                selected_artifact = _installed_artifact(python, provider)
                metadata["selected_artifact"] = selected_artifact
                write_json(metadata_path, metadata)
            return with_runtime(
                provider,
                runtime_id=str(metadata["runtime_id"]),
                runtime_python=str(python),
                installed_lock=tuple(metadata["installed_lock"]),
                selected_artifact=selected_artifact,
            )
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier[:12]}.", dir=root.parent))
    try:
        report(f"Creating Python {provider.python_minor} runtime for {provider.provider}")
        _run([uv_path, "venv", "--python", provider.python_minor, str(temporary)])
        temporary_python = _venv_python(temporary)
        report(f"Preparing source artifact for {provider.provider}")
        install_subject, selected_artifact = _install_subject(
            provider,
            temporary,
            checkout_path=checkout_path,
            artifact_path=artifact_path,
        )
        report(f"Installing {provider.provider} and benchmark dependencies")
        _run(
            [
                uv_path,
                "pip",
                "install",
                "--python",
                str(temporary_python),
                "--compile-bytecode",
                "--exclude-newer",
                "7 days",
                *HARNESS_REQUIREMENTS,
                install_subject,
            ]
        )
        report(f"Validating installed {provider.provider} runtime")
        probe = _probe_install(temporary_python, provider)
        if probe["version"] != provider.version:
            raise RuntimeError(
                f"installed {provider.distribution} version {probe['version']} does not match {provider.version}"
            )
        freeze = _run(
            [uv_path, "pip", "freeze", "--python", str(temporary_python)], capture=True
        ).stdout.splitlines()
        installed_lock = tuple(sorted(line for line in freeze if line.strip()))
        exact_runtime_id = runtime_id(
            provider,
            cache_context=cache_context,
            installed_lock=installed_lock,
        )
        metadata = {
            "cache_key": identifier,
            "runtime_id": exact_runtime_id,
            "complete": True,
            "provider": provider.portable(),
            "harness_requirements": list(HARNESS_REQUIREMENTS),
            "harness_source_sha256": harness_source_hash(),
            "installed_lock": list(installed_lock),
            "selected_artifact": selected_artifact,
            "probe": probe,
        }
        write_json(temporary / "runtime.json", metadata)
        try:
            os.replace(temporary, root)
        except OSError:
            if not metadata_path.is_file():
                raise
            shutil.rmtree(temporary)
        report(f"Cached isolated runtime for {provider.provider}")
        return with_runtime(
            provider,
            runtime_id=exact_runtime_id,
            runtime_python=str(_venv_python(root)),
            installed_lock=tuple(metadata["installed_lock"]),
            selected_artifact=selected_artifact,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _install_subject(
    provider: ResolvedProvider,
    runtime_root: Path,
    *,
    checkout_path: Path | None,
    artifact_path: Path | None,
) -> tuple[str, dict[str, Any] | None]:
    if provider.source_kind == "pypi":
        artifact = _download_release_artifact(provider)
        selected = next(
            (
                item
                for item in provider.release_files
                if item.get("filename") == artifact.name
                and item.get("sha256") == artifact.parent.name
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(f"downloaded artifact is absent from the resolved release: {artifact.name}")
        return str(artifact), _portable_artifact(selected)
    if provider.source_kind == "artifact":
        if artifact_path is None:
            raise ValueError("artifact_path is required for artifact provider resolution")
        source = artifact_path.expanduser().resolve()
        if not source.is_file() or sha256_file(source) != provider.artifact_sha256:
            raise ValueError("local provider artifact is missing or changed after resolution")
        return str(source), provider.selected_artifact
    if checkout_path is None:
        raise ValueError("checkout_path is required for checkout provider resolution")
    snapshot = runtime_root / "source"
    _snapshot_checkout(provider, checkout_path, snapshot)
    build_root = snapshot / provider.build_root if provider.build_root else snapshot
    return str(build_root), None


def _portable_artifact(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": str(item.get("filename", "")),
        "packagetype": str(item.get("packagetype", "")),
        "python_version": str(item.get("python_version", "")),
        "sha256": str(item.get("sha256", "")),
        "size": int(item.get("size", 0)),
    }


def _installed_artifact(python: Path, provider: ResolvedProvider) -> dict[str, Any]:
    code = """
import importlib.metadata, json, pathlib, urllib.parse
dist = importlib.metadata.distribution(__import__('sys').argv[1])
path = pathlib.Path(dist._path) / 'direct_url.json'
payload = json.loads(path.read_text())
print(urllib.parse.unquote(urllib.parse.urlparse(payload['url']).path))
"""
    artifact = Path(
        _run(
            [str(python), "-c", code, provider.distribution],
            capture=True,
            cwd=python.parent.parent,
        ).stdout.strip()
    )
    actual = sha256_file(artifact)
    selected = next(
        (
            item
            for item in provider.release_files
            if item.get("filename") == artifact.name and item.get("sha256") == actual
        ),
        None,
    )
    if selected is None:
        raise RuntimeError(
            f"installed artifact is absent from the resolved release: {artifact.name} ({actual})"
        )
    return _portable_artifact(selected)


def _download_release_artifact(provider: ResolvedProvider) -> Path:
    # Prefer a wheel whose tags uv can validate; fall back to sdist. The exact
    # chosen artifact is verified against PyPI's recorded digest before install.
    files = sorted(provider.release_files, key=lambda item: _artifact_score(provider, item))
    artifacts = cache_root() / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for item in files:
        url = str(item.get("url", ""))
        expected = str(item.get("sha256", ""))
        filename = Path(str(item.get("filename", "artifact"))).name
        if not url.startswith("https://files.pythonhosted.org/") or not expected:
            continue
        target = artifacts / expected / filename
        if target.is_file() and sha256_file(target) == expected:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        try:
            request = Request(url, headers={"User-Agent": "turbobench/2.0.0"})
            with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            actual = sha256_file(temporary)
            if actual != expected:
                raise RuntimeError(f"artifact hash mismatch for {filename}: {actual}")
            os.replace(temporary, target)
            return target
        except BaseException as exc:
            temporary.unlink(missing_ok=True)
            errors.append(f"{filename}: {exc}")
    raise RuntimeError("no provider artifact could be downloaded: " + "; ".join(errors))


def _artifact_score(provider: ResolvedProvider, item: dict[str, Any]) -> tuple[int, str]:
    filename = str(item.get("filename", ""))
    if item.get("packagetype") != "bdist_wheel":
        return (50, filename)
    lowered = filename.casefold()
    python_tag = "cp" + provider.python_minor.replace(".", "")
    if "py3-none-any" in lowered or python_tag in lowered:
        python_score = 0
    elif "abi3" in lowered:
        match = re.search(r"cp(\d{2,3})-abi3", lowered)
        target = int(provider.python_minor.replace(".", ""))
        python_score = 1 if match and int(match.group(1)) <= target else 40
    else:
        python_score = 40
    machine = platform.machine().casefold()
    if "none-any" in lowered:
        platform_score = 0
    elif sys.platform == "darwin":
        platform_score = 0 if ("macosx" in lowered and (machine in lowered or "universal2" in lowered)) else 20
    elif sys.platform.startswith("linux"):
        aliases = {"x86_64", "amd64"} if machine in {"x86_64", "amd64"} else {machine, "aarch64"}
        platform_score = 0 if ("linux" in lowered and any(alias in lowered for alias in aliases)) else 20
    else:
        platform_score = 0 if "none-any" in lowered else 20
    if python_score >= 40 or platform_score:
        return (100, filename)
    return (python_score, filename)


def _snapshot_checkout(provider: ResolvedProvider, checkout: Path, destination: Path) -> None:
    root = checkout.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dirty = "dirty checkout override" in provider.diagnostic_reasons
    submodules = _checkout_submodules(root, allow_moved=dirty)
    _archive_git_tree(root, provider.commit or "HEAD", destination, "source")
    if dirty:
        _overlay_dirty_checkout(
            root,
            destination,
            exclude_roots=tuple(relative for relative, _submodule_root, _commit in submodules),
        )
    for relative, submodule_root, commit in submodules:
        _archive_git_tree(
            submodule_root,
            commit,
            destination / relative,
            "submodule-" + relative.as_posix().replace("/", "-"),
        )
        if dirty:
            nested = tuple(
                child.relative_to(relative)
                for child, _child_root, _child_commit in submodules
                if child != relative and child.is_relative_to(relative)
            )
            _overlay_dirty_checkout(
                submodule_root,
                destination / relative,
                exclude_roots=nested,
            )


def _checkout_submodules(
    root: Path, *, allow_moved: bool
) -> list[tuple[Path, Path, str]]:
    process = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or "could not inspect checkout submodules")
    records: list[tuple[Path, Path, str]] = []
    for raw in process.stdout.splitlines():
        if not raw:
            continue
        status = raw[0]
        fields = raw[1:].split(maxsplit=2)
        if len(fields) < 2:
            raise RuntimeError(f"could not parse submodule status: {raw!r}")
        commit, relative_text = fields[:2]
        relative = Path(relative_text)
        if status == "-":
            raise RuntimeError(f"checkout submodule is not initialized: {relative_text}")
        if status == "U" or (status == "+" and not allow_moved):
            raise RuntimeError(f"checkout submodule is not at its pinned commit: {relative_text}")
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe checkout submodule path: {relative_text}")
        submodule_root = (root / relative).resolve()
        if not submodule_root.is_dir():
            raise RuntimeError(f"checkout submodule is unavailable: {relative_text}")
        records.append((relative, submodule_root, commit))
    return records


def _overlay_dirty_checkout(
    root: Path, destination: Path, *, exclude_roots: tuple[Path, ...] = ()
) -> None:
    """Overlay only Git-visible dirty content on a clean commit archive.

    Copying a dirty worktree wholesale also copies ignored CMake caches,
    editable environments, and stale compiled extensions. Those artifacts are
    neither part of the checkout identity nor portable to the isolated runtime
    and can make a valid diagnostic checkout impossible to build.
    """

    commands = (
        ["git", "diff", "--name-only", "-z", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    )
    relative_paths: set[Path] = set()
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                completed.stderr.decode(errors="replace").strip()
                or "could not inspect dirty checkout"
            )
        relative_paths.update(
            Path(os.fsdecode(raw))
            for raw in completed.stdout.split(b"\0")
            if raw
        )
    for relative in sorted(relative_paths):
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe dirty checkout path: {relative}")
        if any(relative == excluded or relative.is_relative_to(excluded) for excluded in exclude_roots):
            continue
        source = root / relative
        target = destination / relative
        if not source.exists() and not source.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.unlink(missing_ok=True)
            target.symlink_to(os.readlink(source))
        elif source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


def _archive_git_tree(root: Path, commit: str, destination: Path, label: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination.parent / f".{label}.tar"
    with archive_path.open("wb") as handle:
        process = subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=root,
            stdout=handle,
            stderr=subprocess.PIPE,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors="replace"))
    with tarfile.open(archive_path) as archive:
        archive.extractall(destination, filter="data")
    archive_path.unlink()


def _probe_install(python: Path, provider: ResolvedProvider) -> dict[str, Any]:
    code = """
import hashlib, importlib, importlib.metadata, json, pathlib
dist = importlib.metadata.distribution(__import__('sys').argv[1])
module = importlib.import_module(__import__('sys').argv[2])
h = hashlib.sha256()
for file in sorted(dist.files or [], key=str):
    path = pathlib.Path(dist.locate_file(file))
    if path.is_file() and not path.name.endswith(('.pyc', '.pyo')):
        h.update(str(file).encode()); h.update(b'\\0'); h.update(hashlib.sha256(path.read_bytes()).digest())
print(json.dumps({'version': dist.version, 'distribution_tree_sha256': h.hexdigest(), 'import_file': pathlib.Path(module.__file__).name}))
"""
    try:
        process = _run(
            [str(python), "-c", code, provider.distribution, provider.import_name],
            capture=True,
            cwd=python.parent.parent,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(
            f"installed {provider.distribution}=={provider.version} failed import "
            f"{provider.import_name!r}: {detail}"
        ) from exc
    return json.loads(process.stdout)


def _run(
    command: list[str],
    *,
    capture: bool = False,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=True,
    )


def runtimes_are_isolated(left: ResolvedProvider, right: ResolvedProvider) -> bool:
    if not left.runtime_python or not right.runtime_python:
        return False
    # Do not resolve the venv interpreter symlink: different isolated venvs may
    # intentionally point at the same base CPython executable.
    return Path(left.runtime_python).absolute() != Path(right.runtime_python).absolute()
