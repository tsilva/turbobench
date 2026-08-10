"""Built-in provider identities and the custom adapter entry-point protocol."""

from __future__ import annotations

import re
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version

from turbobench.model import ProviderDefinition, ProviderRef

BUILTIN_PROVIDERS: dict[str, ProviderDefinition] = {
    "supermariobrosnes-turbo": ProviderDefinition(
        id="supermariobrosnes-turbo",
        distribution="supermariobrosnes-turbo",
        import_name="supermariobrosnes_turbo",
        adapter="turbo-vector-v1",
        turbo_api=1,
        lineage="stable-retro/nes/supermariobros",
    ),
    "breakout-turbo-env": ProviderDefinition(
        id="breakout-turbo-env",
        distribution="breakout-turbo-env",
        import_name="breakout_turbo_env",
        adapter="turbo-vector-v1",
        turbo_api=1,
        lineage="stable-retro/atari2600/breakout",
    ),
    "stable-retro-turbo": ProviderDefinition(
        id="stable-retro-turbo",
        distribution="stable-retro-turbo",
        import_name="stable_retro",
        adapter="stable-retro-turbo",
        lineage="stable-retro",
    ),
    "stable-retro": ProviderDefinition(
        id="stable-retro",
        distribution="stable-retro",
        import_name="retro",
        adapter="stable-retro-scalar",
        lineage="stable-retro",
    ),
    "vizdoom-turbo": ProviderDefinition(
        id="vizdoom-turbo",
        distribution="vizdoom-turbo",
        import_name="vizdoom_turbo",
        adapter="vizdoom-turbo",
        build_subdirectory="turbo",
        lineage="vizdoom",
    ),
    "vizdoom": ProviderDefinition(
        id="vizdoom",
        distribution="vizdoom",
        import_name="vizdoom",
        adapter="vizdoom-scalar",
        lineage="vizdoom",
    ),
}

_PROVIDER_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")


def load_providers(*, include_plugins: bool = True) -> dict[str, ProviderDefinition]:
    providers = dict(BUILTIN_PROVIDERS)
    if not include_plugins:
        return providers
    for point in entry_points(group="turbobench.providers"):
        loaded: Any = point.load()
        definition = loaded() if callable(loaded) else loaded
        if not isinstance(definition, ProviderDefinition):
            raise TypeError(f"entry point {point.name!r} did not return ProviderDefinition")
        if definition.id in providers:
            raise ValueError(f"provider entry point collides with {definition.id!r}")
        providers[definition.id] = definition
    return providers


def parse_provider_ref(text: str, providers: dict[str, ProviderDefinition] | None = None) -> ProviderRef:
    known = providers or load_providers()
    provider, separator, selector = text.partition("@")
    if not _PROVIDER_ID.fullmatch(provider) or provider not in known:
        raise ValueError(f"unknown provider {provider!r}")
    if not separator or selector == "latest":
        return ProviderRef(provider, "latest", "latest")
    if selector.startswith("checkout:"):
        raw_path = selector.removeprefix("checkout:")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError("checkout provider refs require an absolute path")
        return ProviderRef(provider, "checkout", str(path))
    try:
        parsed = Version(selector)
    except InvalidVersion as exc:
        raise ValueError(f"invalid provider version {selector!r}") from exc
    if parsed.is_prerelease:
        raise ValueError("prerelease provider versions are not eligible")
    return ProviderRef(provider, "version", str(parsed))


def lineage_version(version: str) -> str:
    parsed = Version(version)
    return parsed.base_version

