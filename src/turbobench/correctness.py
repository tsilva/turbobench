"""Direct, shape-local correctness and replay parity gates."""

from __future__ import annotations

import math
from typing import Any

from turbobench.model import Gate, Profile

FLOAT_TOLERANCE = 1e-6


def compare_traces(left: dict[str, Any], right: dict[str, Any], profile: Profile) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    _same(mismatches, "schema", left.get("schema"), right.get("schema"))
    _same(mismatches, "profile", left.get("profile"), right.get("profile"))
    _same(mismatches, "shape", left.get("shape"), right.get("shape"))
    _same(
        mismatches,
        "action_stream_sha256",
        left.get("action_stream_sha256"),
        right.get("action_stream_sha256"),
    )
    for field in ("observation_sha256", "raw_frame_sha256", "raw_frame_shapes"):
        _same(mismatches, f"initial.{field}", left.get("initial", {}).get(field), right.get("initial", {}).get(field))
    _infos(
        mismatches,
        "initial.infos",
        left.get("initial", {}).get("infos", []),
        right.get("initial", {}).get("infos", []),
        profile,
    )
    left_steps = left.get("steps", [])
    right_steps = right.get("steps", [])
    if len(left_steps) != len(right_steps):
        mismatches.append({"field": "step_count", "left": len(left_steps), "right": len(right_steps)})
    for index, (left_step, right_step) in enumerate(zip(left_steps, right_steps, strict=False), start=1):
        prefix = f"steps[{index}]"
        for field in (
            "observation_sha256",
            "raw_frame_sha256",
            "terminations",
            "truncations",
            "reset_lanes",
        ):
            _same(mismatches, f"{prefix}.{field}", left_step.get(field), right_step.get(field))
        _float_list(mismatches, f"{prefix}.rewards", left_step.get("rewards", []), right_step.get("rewards", []))
        _infos(mismatches, f"{prefix}.infos", left_step.get("infos", []), right_step.get("infos", []), profile)
    _same(mismatches, "reset_points", left.get("reset_points"), right.get("reset_points"))
    _same(mismatches, "completion_step", left.get("completion_step"), right.get("completion_step"))
    gates = [
        Gate("exact policy observations", not _has(mismatches, "observation_sha256"), "all lane/step hashes"),
        Gate("exact raw RGB frames", not _has(mismatches, "raw_frame"), "all lane/step hashes and shapes"),
        Gate("matched rewards", not _has(mismatches, "rewards"), f"absolute tolerance {FLOAT_TOLERANCE:g}"),
        Gate("matched terminations and truncations", not _has_any(mismatches, ("terminations", "truncations")), "exact"),
        Gate("matched resets", not _has_any(mismatches, ("reset_lanes", "reset_points")), "exact selective reset points"),
        Gate("matched completion", not _has(mismatches, "completion_step"), "exact completion step"),
        Gate("matched semantic infos", not _has(mismatches, ".infos"), "integer exact; declared float absolute tolerance"),
        Gate("training-safe observations", _safe_ownership(left) and _safe_ownership(right), "owned or safe-view observations"),
    ]
    passed = not mismatches and all(gate.passed for gate in gates)
    return {
        "passed": passed,
        "shape": left.get("shape"),
        "gates": [gate.__dict__ for gate in gates],
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:50],
    }


def compare_replays(left: dict[str, Any], right: dict[str, Any], profile: Profile) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for field in ("action_stream_sha256", "frame_count", "frame_width", "frame_height", "frame_sha256", "completion_step"):
        _same(mismatches, field, left.get(field), right.get(field))
    left_transitions = left.get("transitions", [])
    right_transitions = right.get("transitions", [])
    if len(left_transitions) != len(right_transitions):
        mismatches.append({"field": "transition_count", "left": len(left_transitions), "right": len(right_transitions)})
    for index, (lhs, rhs) in enumerate(zip(left_transitions, right_transitions, strict=False), start=1):
        prefix = f"transitions[{index}]"
        for field in ("observation_sha256", "raw_frame_sha256", "terminated", "truncated"):
            _same(mismatches, f"{prefix}.{field}", lhs.get(field), rhs.get(field))
        _float_list(mismatches, f"{prefix}.reward", [lhs.get("reward")], [rhs.get("reward")])
        _infos(mismatches, f"{prefix}.infos", [lhs.get("infos", {})], [rhs.get("infos", {})], profile)
    expected = profile.completion.get("step")
    if expected is not None and left.get("completion_step") != expected:
        mismatches.append({"field": "expected_completion_step", "left": left.get("completion_step"), "right": expected})
    return {
        "passed": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:50],
        "completion_step": left.get("completion_step"),
        "frame_count": left.get("frame_count"),
    }


def _same(mismatches: list[dict[str, Any]], field: str, left: Any, right: Any) -> None:
    if left != right:
        mismatches.append({"field": field, "left": left, "right": right})


def _float_list(mismatches: list[dict[str, Any]], field: str, left: list[Any], right: list[Any]) -> None:
    if len(left) != len(right):
        mismatches.append({"field": field, "left": left, "right": right})
        return
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        if not math.isclose(float(lhs), float(rhs), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
            mismatches.append({"field": f"{field}[{index}]", "left": lhs, "right": rhs})


def _infos(
    mismatches: list[dict[str, Any]],
    field: str,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    profile: Profile,
) -> None:
    if len(left) != len(right):
        mismatches.append({"field": field, "left": left, "right": right})
        return
    for lane, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        for key in profile.info_integer:
            if key not in lhs or key not in rhs or int(lhs.get(key, -1)) != int(rhs.get(key, -2)):
                mismatches.append({"field": f"{field}[{lane}].{key}", "left": lhs.get(key), "right": rhs.get(key)})
        for key in profile.info_float:
            if key not in lhs or key not in rhs:
                mismatches.append({"field": f"{field}[{lane}].{key}", "left": lhs.get(key), "right": rhs.get(key)})
            else:
                _float_list(mismatches, f"{field}[{lane}].{key}", [lhs[key]], [rhs[key]])


def _safe_ownership(trace: dict[str, Any]) -> bool:
    return trace.get("environment", {}).get("observation", {}).get("ownership") != "unsafe_view"


def _has(mismatches: list[dict[str, Any]], fragment: str) -> bool:
    return any(fragment in str(item.get("field")) for item in mismatches)


def _has_any(mismatches: list[dict[str, Any]], fragments: tuple[str, ...]) -> bool:
    return any(_has(mismatches, fragment) for fragment in fragments)
