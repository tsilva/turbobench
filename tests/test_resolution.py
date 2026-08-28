from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from turbobench.profiles import get_profile
from turbobench.providers import BUILTIN_PROVIDERS, parse_provider_ref
from turbobench.resolution import (
    _checkout_version,
    _enforce_lineage,
    pypi_candidates,
    resolve_checkout,
    resolve_pair,
)

NOW = datetime(2026, 8, 7, tzinfo=UTC)


def test_standardized_breakout_provider_imports() -> None:
    assert (
        BUILTIN_PROVIDERS["env-breakoutatari2600-turbo-native"].import_name
        == "env_breakoutatari2600_turbo_native"
    )
    assert (
        BUILTIN_PROVIDERS["env-stableretro-turbo"].import_name
        == "env_stableretro_turbo"
    )


def test_stable_retro_turbo_checkout_uses_standardized_version_path(
    tmp_path: Path,
) -> None:
    package = tmp_path / "env_stableretro_turbo"
    package.mkdir()
    (package / "VERSION.txt").write_text("1.0.1.post44\n", encoding="utf-8")

    assert (
        _checkout_version(
            tmp_path,
            tmp_path,
            BUILTIN_PROVIDERS["env-stableretro-turbo"],
        )
        == "1.0.1.post44"
    )


def _file(uploaded: datetime, *, yanked: bool = False, requires: str = ">=3.11") -> dict:
    return {
        "filename": "provider-1-py3-none-any.whl",
        "packagetype": "bdist_wheel",
        "python_version": "py3",
        "requires_python": requires,
        "yanked": yanked,
        "digests": {"sha256": "a" * 64},
        "size": 123,
        "upload_time_iso_8601": uploaded.isoformat(),
        "url": "https://files.pythonhosted.org/provider.whl",
    }


def _metadata(versions: dict[str, list[dict]]) -> dict:
    return {"info": {"requires_python": ">=3.11"}, "releases": versions}


def test_provider_ref_forms_and_validation(tmp_path: Path) -> None:
    assert parse_provider_ref("vizdoom").selector == "latest"
    assert parse_provider_ref("vizdoom@latest").selector == "latest"
    assert parse_provider_ref("vizdoom@1.3.0").value == "1.3.0"
    checkout = parse_provider_ref(f"vizdoom@checkout:{tmp_path.resolve()}")
    assert checkout.selector == "checkout"
    with pytest.raises(ValueError, match="absolute"):
        parse_provider_ref("vizdoom@checkout:relative")
    with pytest.raises(ValueError, match="unknown"):
        parse_provider_ref("unknown")


@pytest.mark.parametrize(
    "legacy_provider",
    (
        "supermariobrosnes-turbo",
        "breakout-turbo-env",
        "stable-retro-turbo",
        "vizdoom-turbo",
    ),
)
def test_legacy_provider_ids_are_not_aliases(legacy_provider: str) -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        parse_provider_ref(legacy_provider)


def test_release_filtering_surfaces_quarantine_yanked_prerelease_and_python() -> None:
    metadata = _metadata(
        {
            "1.0.0": [_file(NOW - timedelta(days=30))],
            "1.1.0": [_file(NOW - timedelta(days=2))],
            "1.2.0": [_file(NOW - timedelta(days=30), yanked=True)],
            "1.3.0rc1": [_file(NOW - timedelta(days=30))],
            "1.4.0": [_file(NOW - timedelta(days=30), requires="<3.14")],
        }
    )
    candidates, excluded = pypi_candidates(
        BUILTIN_PROVIDERS["vizdoom"], python_minor="3.14", now=NOW, metadata=metadata
    )
    assert [item.version for item in candidates] == ["1.1.0", "1.0.0"]
    reasons = {(item["version"], item["reason"]) for item in excluded}
    assert ("1.1.0", "seven-day-quarantine") in reasons
    assert ("1.2.0", "yanked") in reasons
    assert ("1.3.0rc1", "prerelease") in reasons
    assert ("1.4.0", "python-3.14-incompatible") in reasons


def test_latest_resolves_newest_eligible_tuple_and_exact_newer_is_diagnostic() -> None:
    profile = get_profile("vizdoom/basic-v1")
    common = _metadata(
        {
            "1.3.0": [_file(NOW - timedelta(days=30))],
            "1.3.1": [_file(NOW - timedelta(days=2))],
        }
    )
    turbo = _metadata(
        {
            "1.3.0.post2": [_file(NOW - timedelta(days=20))],
            "1.3.1.post1": [_file(NOW - timedelta(days=1))],
        }
    )
    latest = resolve_pair(
        profile,
        parse_provider_ref("env-vizdoom-turbo"),
        parse_provider_ref("vizdoom"),
        BUILTIN_PROVIDERS,
        now=NOW,
        metadata={"env-vizdoom-turbo": turbo, "vizdoom": common},
    )
    assert latest.left.version == "1.3.0.post2"
    assert latest.right.version == "1.3.0"
    exact = resolve_pair(
        profile,
        parse_provider_ref("env-vizdoom-turbo@1.3.1.post1"),
        parse_provider_ref("vizdoom@1.3.1"),
        BUILTIN_PROVIDERS,
        now=NOW,
        metadata={"env-vizdoom-turbo": turbo, "vizdoom": common},
    )
    assert exact.left.diagnostic_reasons == (
        "exact version is inside the seven-day quarantine",
    )
    assert exact.right.diagnostic_reasons == (
        "exact version is inside the seven-day quarantine",
    )


def test_exact_supermario_release_is_quarantine_exempt() -> None:
    profile = get_profile("supermario/canonical-v1")
    candidate = _metadata(
        {"0.6.4": [_file(NOW - timedelta(days=1), requires=">=3.9")]}
    )
    baseline = _metadata(
        {"1.0.1": [_file(NOW - timedelta(days=30), requires=">=3.9")]}
    )

    result = resolve_pair(
        profile,
        parse_provider_ref("env-supermariobrosnes-turbo-emu@0.6.4"),
        parse_provider_ref("stable-retro@1.0.1"),
        BUILTIN_PROVIDERS,
        now=NOW,
        metadata={
            "env-supermariobrosnes-turbo-emu": candidate,
            "stable-retro": baseline,
        },
    )

    assert not result.left.diagnostic_reasons
    assert not result.right.diagnostic_reasons
    assert (
        "0.6.4",
        "seven-day-quarantine",
    ) in {
        (item["version"], item["reason"])
        for item in result.excluded["env-supermariobrosnes-turbo-emu"]
    }


def test_exact_breakout_release_is_quarantine_exempt() -> None:
    profile = get_profile("breakout/start-v2")
    candidate = _metadata({"0.5.6": [_file(NOW - timedelta(days=1))]})
    baseline = _metadata({"1.0.1": [_file(NOW - timedelta(days=30))]})

    result = resolve_pair(
        profile,
        parse_provider_ref("stable-retro@1.0.1"),
        parse_provider_ref("env-breakoutatari2600-turbo-native@0.5.6"),
        BUILTIN_PROVIDERS,
        now=NOW,
        metadata={
            "stable-retro": baseline,
            "env-breakoutatari2600-turbo-native": candidate,
        },
    )

    assert not result.left.diagnostic_reasons
    assert not result.right.diagnostic_reasons
    assert (
        "0.5.6",
        "seven-day-quarantine",
    ) in {
        (item["version"], item["reason"])
        for item in result.excluded["env-breakoutatari2600-turbo-native"]
    }


def test_latest_solves_newest_compatible_lineage_tuple_and_surfaces_newer() -> None:
    profile = get_profile("vizdoom/basic-v1")
    common = _metadata(
        {
            "1.3.0": [_file(NOW - timedelta(days=40))],
            "1.4.0": [_file(NOW - timedelta(days=20))],
        }
    )
    turbo = _metadata(
        {
            "1.3.0.post3": [_file(NOW - timedelta(days=15))],
            "1.5.0.post1": [_file(NOW - timedelta(days=10))],
        }
    )
    result = resolve_pair(
        profile,
        parse_provider_ref("env-vizdoom-turbo"),
        parse_provider_ref("vizdoom"),
        BUILTIN_PROVIDERS,
        now=NOW,
        metadata={"env-vizdoom-turbo": turbo, "vizdoom": common},
    )
    assert result.left.version == "1.3.0.post3"
    assert result.right.version == "1.3.0"
    assert {
        (item["version"], item["reason"])
        for items in result.excluded.values()
        for item in items
    } >= {
        ("1.5.0.post1", "lineage-incompatible"),
        ("1.4.0", "lineage-incompatible"),
    }


def test_latest_stable_retro_pair_requires_matching_base_lineage() -> None:
    profile = get_profile("breakout/start-v1")
    common = _metadata(
        {
            "1.0.1": [_file(NOW - timedelta(days=40))],
            "1.1.0": [_file(NOW - timedelta(days=20))],
        }
    )
    turbo = _metadata(
        {
            "1.0.1.post37": [_file(NOW - timedelta(days=15))],
            "1.2.0.post1": [_file(NOW - timedelta(days=10))],
        }
    )
    result = resolve_pair(
        profile,
        parse_provider_ref("env-stableretro-turbo"),
        parse_provider_ref("stable-retro"),
        BUILTIN_PROVIDERS,
        now=NOW,
        metadata={"env-stableretro-turbo": turbo, "stable-retro": common},
    )
    assert result.left.version == "1.0.1.post37"
    assert result.right.version == "1.0.1"
    assert result.left.compatibility_lineage == "1.0.1"
    assert result.right.compatibility_lineage == "1.0.1"
    assert {
        (item["version"], item["reason"])
        for items in result.excluded.values()
        for item in items
    } >= {
        ("1.2.0.post1", "lineage-incompatible"),
        ("1.1.0", "lineage-incompatible"),
    }


def test_profile_compatibility_and_lineage_are_enforced() -> None:
    profile = get_profile("vizdoom/basic-v1")
    with pytest.raises(ValueError, match="compatible pair"):
        resolve_pair(
            profile,
            parse_provider_ref("stable-retro"),
            parse_provider_ref("vizdoom"),
            BUILTIN_PROVIDERS,
            now=NOW,
            metadata={},
        )
    from turbobench.resolution import fake_resolved

    left = fake_resolved("env-vizdoom-turbo")
    right = fake_resolved("vizdoom")
    left = left.__class__(**{**left.__dict__, "version": "1.3.0.post1"})
    right = right.__class__(**{**right.__dict__, "version": "1.4.0"})
    with pytest.raises(ValueError, match="lineage"):
        _enforce_lineage(left, right)


def test_checkout_resolution_uses_clean_commit_tree_and_vizdoom_subproject(tmp_path: Path) -> None:
    root = tmp_path / "vizdoom"
    (root / "turbo").mkdir(parents=True)
    (root / "turbo" / "pyproject.toml").write_text(
        '[project]\nname="env-vizdoom-turbo"\nversion="1.3.0.post1"\n',
        encoding="utf-8",
    )
    (root / "keep-benchmark.py").write_text("print('preserved')\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "initial"],
        cwd=root,
        check=True,
    )
    resolved = resolve_checkout(
        BUILTIN_PROVIDERS["env-vizdoom-turbo"], root, python_minor="3.14"
    )
    assert resolved.build_root == "turbo"
    assert resolved.commit and resolved.tree and not resolved.diagnostic_reasons
    (root / "keep-benchmark.py").write_text("print('dirty')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        resolve_checkout(
            BUILTIN_PROVIDERS["env-vizdoom-turbo"], root, python_minor="3.14"
        )
    dirty = resolve_checkout(
        BUILTIN_PROVIDERS["env-vizdoom-turbo"],
        root,
        python_minor="3.14",
        allow_dirty=True,
    )
    assert dirty.diagnostic_reasons == ("dirty checkout override",)
