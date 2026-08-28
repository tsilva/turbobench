"""Normative Turbo Vector API v2 schema and runtime validation."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from turbobench.util import canonical_json_hash

REPORT_SCHEMA = "turbobench.turbo-contract-report/v1"
TURBO_API_VERSION = 2

COMMON_CONSTRUCTOR_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("game", inspect.Parameter.empty),
    ("state", None),
    ("scenario", None),
    ("info", None),
    ("use_restricted_actions", "default"),
    ("record", False),
    ("players", 1),
    ("inttype", "stable"),
    ("obs_type", "image"),
    ("render_mode", None),
    ("num_envs", 1),
    ("num_threads", None),
    ("rom_path", None),
    ("transport", "default"),
    ("obs_copy", "safe_view"),
    ("obs_resize", (84, 84)),
    ("obs_crop", None),
    ("obs_crop_mode", "remove"),
    ("obs_crop_fill", 0),
    ("obs_grayscale", True),
    ("obs_resize_algorithm", "area"),
    ("obs_layout", "chw"),
    ("frame_skip", 4),
    ("frame_stack", 4),
    ("maxpool_last_two", False),
    ("noop_reset_max", 0),
    ("use_fire_reset", False),
    ("sticky_action_prob", 0.0),
    ("reward_clip", False),
    ("info_filter", "all"),
    ("info_frame_stack_keys", None),
    ("state_catalog", None),
)

CAPABILITY_KEYS = (
    "supported_action_modes",
    "supported_observation_layouts",
    "supported_observation_color_modes",
    "supported_resize_algorithms",
    "supported_crop_modes",
    "supported_observation_copy_modes",
    "supported_transition_transports",
    "supports_async_step",
    "supports_branching",
    "supports_device_api",
    "supports_emulator_ram",
    "supports_enemy_variants",
    "supports_fire_reset",
    "supports_info_frame_stack",
    "supports_live_snapshots",
    "supports_maxpool_last_two",
    "supports_noop_reset",
    "supports_per_lane_rgb",
    "supports_reward_clipping",
    "supports_snapshot_codec",
    "supports_state_catalog",
    "supports_sticky_action_prob",
    "supports_surface_variants",
)

SEQUENCE_CAPABILITIES = CAPABILITY_KEYS[:7]
SEQUENCE_CAPABILITY_EXTENSIONS = ("supported_filtered_actions",)
_PORTABLE_DTYPES = frozenset(
    {
        "bool",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    }
)
FEATURE_METHODS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "supports_async_step": ("step_async", "step_wait"),
        "supports_branching": ("branch",),
        "supports_device_api": ("step_device", "reset_device"),
        "supports_emulator_ram": ("ram",),
        "supports_live_snapshots": ("capture_snapshots",),
        "supports_per_lane_rgb": ("render_lane", "get_images"),
        "supports_snapshot_codec": ("encode_snapshots", "decode_snapshots"),
        "supports_state_catalog": ("active_state_indices",),
    }
)


class TurboContractError(RuntimeError):
    """A declared v2 provider failed before its workload could start."""

    def __init__(self, report: dict[str, Any]):
        super().__init__("Turbo Vector API v2 validation failed")
        self.report = report


def declared_api_version(environment_type: type[Any]) -> int | None:
    metadata = getattr(environment_type, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("turbo_api_version")
    return int(value) if value is not None else None


def validate_constructor(environment_type: type[Any], provider: str) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    signature = inspect.signature(environment_type)
    parameters = tuple(signature.parameters.values())
    expected_names = tuple(name for name, _default in COMMON_CONSTRUCTOR_DEFAULTS)
    actual_names = tuple(parameter.name for parameter in parameters[: len(expected_names)])
    _check(
        checks,
        errors,
        "constructor common order",
        actual_names == expected_names,
        f"expected {expected_names}, got {actual_names}",
    )
    for index, (name, expected_default) in enumerate(COMMON_CONSTRUCTOR_DEFAULTS):
        if index >= len(parameters) or parameters[index].name != name:
            continue
        parameter = parameters[index]
        _check(
            checks,
            errors,
            f"constructor default {name}",
            _same_default(parameter.default, expected_default),
            f"expected {expected_default!r}, got {parameter.default!r}",
        )
        expected_kind = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD
            if index < 10
            else inspect.Parameter.KEYWORD_ONLY
        )
        _check(
            checks,
            errors,
            f"constructor kind {name}",
            parameter.kind is expected_kind,
            f"expected {expected_kind.name}, got {parameter.kind.name}",
        )
    extensions = parameters[len(expected_names) :]
    _check(
        checks,
        errors,
        "constructor keyword-only extensions",
        all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in extensions),
        "provider extensions must be keyword-only",
    )
    _check(
        checks,
        errors,
        "constructor rejects unknown arguments",
        all(parameter.kind is not inspect.Parameter.VAR_KEYWORD for parameter in parameters),
        "catch-all **kwargs is forbidden",
    )
    return _report(
        provider=provider,
        api_version=declared_api_version(environment_type),
        applicable=True,
        promotable=not errors,
        checks=checks,
        errors=errors,
        phase="constructor",
    )


def legacy_report(provider: str, api_version: int | None) -> dict[str, Any]:
    reason = (
        "historical Turbo Vector API v1 is runnable only as a diagnostic workload"
        if api_version == 1
        else "provider does not declare the Turbo Vector API and is outside this contract"
    )
    return _report(
        provider=provider,
        api_version=api_version,
        applicable=api_version == 1,
        promotable=api_version is None,
        checks=[{"name": "API declaration", "passed": api_version is None, "detail": reason}],
        errors=[] if api_version is None else [reason],
        phase="declaration",
        passed=True,
    )


def validate_environment(environment_type: type[Any], env: Any, provider: str) -> dict[str, Any]:
    constructor = validate_constructor(environment_type, provider)
    checks = list(constructor["checks"])
    errors = list(constructor["errors"])
    raw_metadata = getattr(env, "metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    transport = metadata.get("transition_transport")
    _check(
        checks,
        errors,
        "metadata mapping",
        isinstance(raw_metadata, Mapping),
        f"got {type(raw_metadata).__name__}",
    )
    _check(
        checks,
        errors,
        "metadata API version",
        metadata.get("turbo_api_version") == 2,
        f"got {metadata.get('turbo_api_version')!r}",
    )
    _check(
        checks,
        errors,
        "metadata transition transport",
        transport in {"numpy", "torch"},
        f"got {transport!r}",
    )
    autoreset = metadata.get("autoreset_mode") if isinstance(metadata, Mapping) else None
    _check(
        checks,
        errors,
        "disabled autoreset",
        "disabled" in str(autoreset).casefold(),
        f"got {autoreset!r}",
    )
    _check(
        checks,
        errors,
        "resolved transition transport",
        getattr(env, "transport", None) == transport,
        f"metadata={transport!r}, env.transport={getattr(env, 'transport', None)!r}",
    )

    _validate_spaces(checks, errors, env, transport)

    capabilities = getattr(env, "capabilities", None)
    _check(
        checks,
        errors,
        "immutable capabilities",
        isinstance(capabilities, MappingProxyType),
        f"got {type(capabilities).__name__}",
    )
    if isinstance(capabilities, Mapping):
        common_capability_keys = tuple(
            name
            for name in capabilities
            if name not in SEQUENCE_CAPABILITY_EXTENSIONS
        )
        _check(
            checks,
            errors,
            "exact capability keys",
            common_capability_keys == CAPABILITY_KEYS,
            f"got {tuple(capabilities)}",
        )
        for name in SEQUENCE_CAPABILITIES:
            _check(
                checks,
                errors,
                f"immutable capability {name}",
                isinstance(capabilities.get(name), tuple),
                f"got {capabilities.get(name)!r}",
            )
        for name in SEQUENCE_CAPABILITY_EXTENSIONS:
            if name not in capabilities:
                continue
            _check(
                checks,
                errors,
                f"immutable capability extension {name}",
                isinstance(capabilities.get(name), tuple),
                f"got {capabilities.get(name)!r}",
            )
        for name in CAPABILITY_KEYS[len(SEQUENCE_CAPABILITIES) :]:
            _check(
                checks,
                errors,
                f"boolean capability {name}",
                isinstance(capabilities.get(name), bool),
                f"got {capabilities.get(name)!r}",
            )
        for feature, methods in FEATURE_METHODS.items():
            advertised = capabilities.get(feature) is True
            coherent = not advertised or all(
                callable(getattr(env, method, None)) for method in methods
            )
            _check(
                checks,
                errors,
                f"capability/method coherence {feature}",
                coherent,
                f"requires {methods}",
            )
        _check(
            checks,
            errors,
            "transport capability coherence",
            transport in capabilities.get("supported_transition_transports", ()),
            f"{transport!r} not advertised",
        )

    schema = getattr(env, "signal_schema", None)
    _check(
        checks,
        errors,
        "immutable signal schema",
        isinstance(schema, MappingProxyType),
        f"got {type(schema).__name__}",
    )
    if isinstance(schema, Mapping):
        for name, entry in schema.items():
            valid = (
                isinstance(name, str)
                and isinstance(entry, MappingProxyType)
                and set(entry) == {"dtype", "shape", "available_on_reset", "available_on_step"}
                and isinstance(entry.get("dtype"), str)
                and entry.get("dtype") in _PORTABLE_DTYPES
                and isinstance(entry.get("shape"), tuple)
                and all(isinstance(size, int) and size >= 0 for size in entry.get("shape", ()))
                and isinstance(entry.get("available_on_reset"), bool)
                and isinstance(entry.get("available_on_step"), bool)
            )
            detail = repr(dict(entry)) if isinstance(entry, Mapping) else repr(entry)
            _check(checks, errors, f"portable signal schema {name}", valid, detail)

    ownership = getattr(env, "observation_ownership", None)
    buffer_depth = getattr(env, "observation_buffer_depth", None)
    valid_ownership = ownership in {"owned", "safe_view", "unsafe_view"}
    _check(
        checks,
        errors,
        "observation ownership declaration",
        valid_ownership,
        f"ownership={ownership!r}",
    )
    expected_depth = {"owned": None, "safe_view": 2, "unsafe_view": 1}.get(ownership)
    _check(
        checks,
        errors,
        "observation buffer depth",
        valid_ownership and buffer_depth == expected_depth,
        f"ownership={ownership!r}, depth={buffer_depth!r}",
    )

    catalog = getattr(env, "state_catalog", None)
    if isinstance(capabilities, Mapping) and capabilities.get("supports_state_catalog"):
        _check(
            checks,
            errors,
            "immutable non-empty state catalog",
            isinstance(catalog, tuple) and bool(catalog),
            repr(catalog),
        )
        try:
            unique = len(set(catalog)) == len(catalog)
        except TypeError:
            unique = False
        _check(checks, errors, "unique state catalog", unique, repr(catalog))

    runtime = _validate_transitions(env, transport, capabilities, schema)
    checks.extend(runtime[0])
    errors.extend(runtime[1])
    return _report(
        provider=provider,
        api_version=2,
        applicable=True,
        promotable=not errors,
        checks=checks,
        errors=errors,
        phase="runtime",
    )


def _validate_transitions(
    env: Any,
    transport: str | None,
    capabilities: Mapping[str, Any] | None,
    schema: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    if transport not in {"numpy", "torch"}:
        return checks, errors
    num_envs = int(getattr(env, "num_envs", 0))
    module = _transport_module(transport)
    device = getattr(env, "device", None)
    mask = _array(module, [True] * num_envs, "bool", device)
    options: dict[str, Any] = {"reset_mask": mask}
    catalog = getattr(env, "state_catalog", ())
    if catalog:
        options["state_indices"] = _array(module, [0] * num_envs, "int32", device)
    try:
        reset = env.reset(seed=123, options=options)
        valid_reset = isinstance(reset, tuple) and len(reset) == 2
        _check(checks, errors, "reset return arity", valid_reset, repr(type(reset)))
        if not valid_reset:
            return checks, errors
        observations, infos = reset
        _check(
            checks,
            errors,
            "reset observation transport",
            _is_transport(observations, transport),
            type(observations).__name__,
        )
        _check(
            checks,
            errors,
            "reset observation shape",
            tuple(observations.shape) == (num_envs, 4, 84, 84),
            repr(tuple(observations.shape)),
        )
        _check(
            checks,
            errors,
            "reset infos mapping",
            _mapping_transport(infos, transport, num_envs),
            type(infos).__name__,
        )
        _validate_reset_infos(checks, errors, infos, transport, num_envs)
        _validate_info_schema(checks, errors, infos, schema, num_envs, reset=True)
        _check(
            checks,
            errors,
            "reset has no object arrays",
            not _contains_object_array(reset),
            "object/string transition arrays are forbidden",
        )

        reset_observations = _clone(observations, transport)
        actions = _zero_actions(env, transport, module, device)
        step = env.step(actions)
        valid_step = isinstance(step, tuple) and len(step) == 5
        _check(checks, errors, "step return arity", valid_step, repr(type(step)))
        if valid_step:
            step_obs, rewards, terminated, truncated, step_infos = step
            for name, value in (
                ("observations", step_obs),
                ("rewards", rewards),
                ("terminations", terminated),
                ("truncations", truncated),
            ):
                _check(
                    checks,
                    errors,
                    f"step {name} transport",
                    _is_transport(value, transport),
                    type(value).__name__,
                )
            _check(
                checks,
                errors,
                "step observation shape",
                tuple(step_obs.shape) == tuple(observations.shape),
                repr(tuple(step_obs.shape)),
            )
            _check(
                checks,
                errors,
                "step reward shape and dtype",
                tuple(rewards.shape) == (num_envs,)
                and _dtype_name(rewards) in {"float32", "float64"},
                f"shape={tuple(rewards.shape)}, dtype={_dtype_name(rewards)}",
            )
            _check(
                checks,
                errors,
                "step termination shape and dtype",
                tuple(terminated.shape) == (num_envs,) and _dtype_name(terminated) == "bool",
                f"shape={tuple(terminated.shape)}, dtype={_dtype_name(terminated)}",
            )
            _check(
                checks,
                errors,
                "step truncation shape and dtype",
                tuple(truncated.shape) == (num_envs,) and _dtype_name(truncated) == "bool",
                f"shape={tuple(truncated.shape)}, dtype={_dtype_name(truncated)}",
            )
            _check(
                checks,
                errors,
                "step infos transport",
                _mapping_transport(step_infos, transport, num_envs),
                "lane-aligned values and masks",
            )
            _validate_info_schema(checks, errors, step_infos, schema, num_envs, reset=False)
            _check(
                checks,
                errors,
                "step has no object arrays",
                not _contains_object_array(step),
                "object/string transition arrays are forbidden",
            )
            ownership = getattr(env, "observation_ownership", None)
            if ownership in {"owned", "safe_view"}:
                _check(
                    checks,
                    errors,
                    f"{ownership} observation lifetime",
                    _equal(observations, reset_observations, transport),
                    "reset observation changed after one subsequent observation call",
                )

            if num_envs > 1:
                selective = _array(module, [True] + [False] * (num_envs - 1), "bool", device)
                selected_options: dict[str, Any] = {"reset_mask": selective}
                if catalog:
                    selected_options["state_indices"] = _array(
                        module, [0] * num_envs, "int32", device
                    )
                unchanged_before_reset = _clone(step_obs[1:], transport)
                selected_obs, _selected_infos = env.reset(options=selected_options)
                unchanged = _equal(selected_obs[1:], unchanged_before_reset, transport)
                _check(
                    checks,
                    errors,
                    "selective reset lane isolation",
                    unchanged,
                    "unselected observations changed",
                )

        if capabilities and capabilities.get("supports_async_step"):
            env.step_async(_zero_actions(env, transport, module, device))
            async_step = env.step_wait()
            _check(
                checks,
                errors,
                "async step return",
                isinstance(async_step, tuple) and len(async_step) == 5,
                repr(type(async_step)),
            )
        if capabilities and capabilities.get("supports_per_lane_rgb"):
            frame = env.render()
            lane = env.render_lane(0)
            images = env.get_images()
            valid_render = (
                isinstance(frame, np.ndarray)
                and isinstance(lane, np.ndarray)
                and isinstance(images, Sequence)
                and len(images) == num_envs
                and all(
                    isinstance(value, np.ndarray)
                    and value.dtype == np.uint8
                    and value.ndim == 3
                    and value.shape[-1] == 3
                    for value in images
                )
                and np.array_equal(frame, lane)
                and np.array_equal(lane, images[0])
            )
            _check(
                checks, errors, "per-lane RGB rendering", valid_render, "expected NumPy RGB frames"
            )
    except Exception as exc:  # validation must report provider failures portably
        _check(
            checks, errors, "runtime validation completed", False, f"{type(exc).__name__}: {exc}"
        )
    return checks, errors


def _validate_reset_infos(
    checks: list[dict[str, Any]],
    errors: list[str],
    infos: Mapping[str, Any],
    transport: str,
    num_envs: int,
) -> None:
    expected = {
        "state_index": "int32",
        "start_source": "int8",
        "noop_reset_count": "int64",
        "_state_index": "bool",
        "_start_source": "bool",
        "_noop_reset_count": "bool",
    }
    for name, dtype in expected.items():
        value = infos.get(name)
        valid = (
            value is not None
            and _is_transport(value, transport)
            and tuple(value.shape) == (num_envs,)
            and _dtype_name(value) == dtype
        )
        _check(
            checks,
            errors,
            f"reset info {name}",
            valid,
            f"expected {transport} {dtype}[{num_envs}], got {type(value).__name__} {_dtype_name(value)}",
        )
    start_source = infos.get("start_source")
    if start_source is not None and _is_transport(start_source, transport):
        valid_sources = _all_values_in(start_source, (0, 1), transport)
        _check(checks, errors, "reset start_source values", valid_sources, "expected only 0 or 1")


def _validate_spaces(
    checks: list[dict[str, Any]], errors: list[str], env: Any, transport: str | None
) -> None:
    num_envs = int(getattr(env, "num_envs", 0))
    single = getattr(env, "single_observation_space", None)
    batch = getattr(env, "observation_space", None)
    action = getattr(env, "action_space", None)
    _check(
        checks,
        errors,
        "single observation space",
        tuple(getattr(single, "shape", ())) == (4, 84, 84) and _dtype_name(single) == "uint8",
        f"shape={getattr(single, 'shape', None)}, dtype={_dtype_name(single)}",
    )
    _check(
        checks,
        errors,
        "batched observation space",
        tuple(getattr(batch, "shape", ())) == (num_envs, 4, 84, 84)
        and _dtype_name(batch) == "uint8",
        f"shape={getattr(batch, 'shape', None)}, dtype={_dtype_name(batch)}",
    )
    action_shape = tuple(getattr(action, "shape", ()))
    _check(
        checks,
        errors,
        "batched action space",
        bool(action_shape)
        and action_shape[0] == num_envs
        and callable(getattr(action, "sample", None)),
        f"shape={action_shape}, transport={transport}",
    )


def _validate_info_schema(
    checks: list[dict[str, Any]],
    errors: list[str],
    infos: Mapping[str, Any],
    schema: Mapping[str, Any] | None,
    num_envs: int,
    *,
    reset: bool,
) -> None:
    if not isinstance(schema, Mapping) or not isinstance(infos, Mapping):
        return
    availability = "available_on_reset" if reset else "available_on_step"
    for name, entry in schema.items():
        if not entry.get(availability):
            continue
        value = infos.get(name)
        mask = infos.get(f"_{name}")
        expected_shape = (num_envs, *entry.get("shape", ()))
        valid = (
            value is not None
            and _dtype_name(value) == entry.get("dtype")
            and tuple(getattr(value, "shape", ())) == expected_shape
        )
        _check(
            checks,
            errors,
            f"signal dtype {name} on {'reset' if reset else 'step'}",
            valid,
            f"expected {entry.get('dtype')!r}{expected_shape}, "
            f"got {_dtype_name(value)!r}{tuple(getattr(value, 'shape', ()))!r}",
        )
        valid_mask = (
            mask is not None
            and _dtype_name(mask) == "bool"
            and tuple(getattr(mask, "shape", ())) == (num_envs,)
        )
        _check(
            checks,
            errors,
            f"signal mask {name} on {'reset' if reset else 'step'}",
            valid_mask,
            f"expected bool[{num_envs}], got {_dtype_name(mask)!r}"
            f"{tuple(getattr(mask, 'shape', ()))!r}",
        )


def _transport_module(transport: str) -> Any:
    if transport == "numpy":
        return np
    import torch  # lazy: scalar and NumPy providers never import Torch

    return torch


def _array(module: Any, values: Sequence[Any], dtype: str, device: Any) -> Any:
    if module is np:
        return np.asarray(values, dtype=np.dtype(dtype))
    return module.tensor(values, dtype=getattr(module, dtype), device=device)


def _zero_actions(env: Any, transport: str, module: Any, device: Any) -> Any:
    sample = np.asarray(env.action_space.sample())
    values = np.zeros(sample.shape, dtype=sample.dtype)
    if transport == "numpy":
        return values
    return module.as_tensor(values, device=device)


def _is_transport(value: Any, transport: str) -> bool:
    if transport == "numpy":
        return isinstance(value, np.ndarray)
    module = _transport_module("torch")
    return isinstance(value, module.Tensor)


def _mapping_transport(values: Any, transport: str, num_envs: int) -> bool:
    return isinstance(values, Mapping) and all(
        _is_transport(value, transport)
        and bool(tuple(value.shape))
        and int(value.shape[0]) == num_envs
        for value in values.values()
    )


def _all_values_in(value: Any, allowed: tuple[int, ...], transport: str) -> bool:
    if transport == "numpy":
        return bool(np.all(np.isin(value, allowed)))
    module = _transport_module("torch")
    return bool(module.all(module.isin(value, module.tensor(allowed, device=value.device))))


def _dtype_name(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def _contains_object_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.dtype.kind in {"O", "U", "S"}
    if isinstance(value, Mapping):
        return any(_contains_object_array(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_object_array(child) for child in value)
    return False


def _equal(left: Any, right: Any, transport: str) -> bool:
    if transport == "numpy":
        return bool(np.array_equal(left, right))
    module = _transport_module("torch")
    return bool(module.equal(left, right))


def _clone(value: Any, transport: str) -> Any:
    if transport == "numpy":
        return value.copy()
    return value.clone()


def _same_default(actual: Any, expected: Any) -> bool:
    if expected is inspect.Parameter.empty:
        return actual is inspect.Parameter.empty
    return type(actual) is type(expected) and actual == expected


def _check(
    checks: list[dict[str, Any]], errors: list[str], name: str, passed: bool, detail: str
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _report(
    *,
    provider: str,
    api_version: int | None,
    applicable: bool,
    promotable: bool,
    checks: list[dict[str, Any]],
    errors: list[str],
    phase: str,
    passed: bool | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": REPORT_SCHEMA,
        "provider": provider,
        "api_version": api_version,
        "applicable": applicable,
        "phase": phase,
        "passed": not errors if passed is None else passed,
        "promotable": bool(promotable and not errors),
        "checks": checks,
        "errors": errors,
    }
    payload["report_sha256"] = canonical_json_hash(payload)
    return payload


__all__ = [
    "CAPABILITY_KEYS",
    "COMMON_CONSTRUCTOR_DEFAULTS",
    "REPORT_SCHEMA",
    "SEQUENCE_CAPABILITY_EXTENSIONS",
    "TURBO_API_VERSION",
    "TurboContractError",
    "declared_api_version",
    "legacy_report",
    "validate_constructor",
    "validate_environment",
]
