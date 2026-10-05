"""Keep machine access details out of benchmark publications and console output."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

# These origins describe public artifacts, documentation, or test placeholders,
# never the user's benchmark infrastructure. Explicit ports and logins are private.
_PUBLIC_ORIGINS = frozenset(
    {
        "github.com",
        "api.github.com",
        "raw.githubusercontent.com",
        "objects.githubusercontent.com",
        "pypi.org",
        "files.pythonhosted.org",
        "docs.github.com",
        "huggingface.co",
        "json-schema.org",
        "turbobench.dev",
        "example.com",
        "example.org",
    }
)
_URL = re.compile(r'https?://[^\s<>"`)]+')
_LOCATION_KEY = re.compile(
    r"^(?:hostname|host_name|benchmark_host|render_host|ssh_host|ssh_alias|ip|ip_address|address|port|tracking_uri|mlflow_url|server_url|endpoint)$",
    re.I,
)
_HOST_CONTEXT = re.compile(
    r"(?i)(\b(?:hostname|ssh host|benchmark host|remote host|server|tracking endpoint)\s*[:=]\s*)([^\s,;]+)"
)


def private_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.rstrip(".,;"))
        return bool(
            parsed.username
            or parsed.password
            or parsed.port
            or parsed.hostname not in _PUBLIC_ORIGINS
        )
    except ValueError:
        return True


def public_text(value: str) -> str:
    """Remove private URLs and explicitly labelled access destinations."""
    value = _URL.sub(
        lambda match: "[private endpoint omitted]" if private_url(match.group()) else match.group(),
        value,
    )
    return _HOST_CONTEXT.sub(lambda match: match.group(1) + "[access details omitted]", value)


def training_reference(value: str) -> str:
    """Retain a non-routable run identifier without exposing its tracking server."""
    match = re.search(r"/runs/([a-zA-Z0-9_-]+)", value)
    if match:
        return "training run `" + match.group(1) + "`"
    if re.fullmatch(r"training-run:[a-zA-Z0-9_-]+", value):
        return "training run `" + value.partition(":")[2] + "`"
    return "locked training run (access details omitted)"


def _has_access_fields(value, *, key="") -> bool:
    if isinstance(value, dict):
        return any(_has_access_fields(child, key=name) for name, child in value.items())
    if isinstance(value, list):
        return any(_has_access_fields(child, key=key) for child in value)
    if _LOCATION_KEY.fullmatch(key) and value not in (
        None,
        "",
        "<redacted>",
        "[access details omitted]",
    ):
        # Non-routable policy references are allowed in legacy field names.
        return not (
            key == "mlflow_url" and isinstance(value, str) and value.startswith("training-run:")
        )
    return isinstance(value, str) and any(
        private_url(match.group()) for match in _URL.finditer(value)
    )


def require_public_proof(root: Path) -> None:
    """Fail before exporting any proof that would expose private infrastructure."""
    for path in root.rglob("*"):
        if (
            path.suffix not in {".json", ".md", ".toml", ".txt", ".log", ".csv"}
            or not path.is_file()
        ):
            continue
        text = path.read_text(encoding="utf-8")
        sensitive = any(private_url(match.group()) for match in _URL.finditer(text))
        if path.suffix == ".json":
            sensitive = sensitive or _has_access_fields(json.loads(text))
        if sensitive:
            raise ValueError(
                "proof contains private machine access details; keep it private and prepare a privacy-safe proof before publication"
            )
