"""Direct, shape-local correctness and replay parity gates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from turbobench.model import Gate, Profile
from turbobench.util import canonical_json_hash

FLOAT_TOLERANCE = 1e-6


def compare_traces(
    left: dict[str, Any],
    right: dict[str, Any],
    profile: Profile,
    *,
    require_snapshot: bool = False,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    _required_trace_evidence(mismatches, "left", left, require_snapshot=require_snapshot)
    _required_trace_evidence(mismatches, "right", right, require_snapshot=require_snapshot)
    _same(mismatches, "schema", left.get("schema"), right.get("schema"))
    _same(mismatches, "profile", left.get("profile"), right.get("profile"))
    _same(mismatches, "shape", left.get("shape"), right.get("shape"))
    _same(
        mismatches,
        "action_stream_sha256",
        left.get("action_stream_sha256"),
        right.get("action_stream_sha256"),
    )
    for field in (
        "observation_sha256",
        "raw_frame_sha256",
        "raw_frame_shapes",
        "ram_sha256",
        "ram_shapes",
    ):
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
            "ram_sha256",
            "ram_shapes",
        ):
            _same(mismatches, f"{prefix}.{field}", left_step.get(field), right_step.get(field))
        _float_list(
            mismatches,
            f"{prefix}.rewards",
            left_step.get("rewards", []),
            right_step.get("rewards", []),
            exact=profile.native_transition_exact,
        )
        _infos(mismatches, f"{prefix}.infos", left_step.get("infos", []), right_step.get("infos", []), profile)
    _same(mismatches, "reset_points", left.get("reset_points"), right.get("reset_points"))
    _same(mismatches, "completion_step", left.get("completion_step"), right.get("completion_step"))
    expected_actions = [list(labels) for labels in profile.action_table.values()]
    _same(
        mismatches,
        "environment.action_identity.left",
        left.get("environment", {}).get("action_table"),
        expected_actions,
    )
    _same(
        mismatches,
        "environment.action_identity.right",
        right.get("environment", {}).get("action_table"),
        expected_actions,
    )
    _same(
        mismatches,
        "snapshot_continuation",
        left.get("snapshot_continuation"),
        right.get("snapshot_continuation"),
    )
    for side, trace in (("left", left), ("right", right)):
        snapshot = trace.get("snapshot_continuation")
        if snapshot is not None and not snapshot.get("replay_exact"):
            mismatches.append(
                {
                    "field": f"snapshot_continuation.{side}.replay_exact",
                    "left": snapshot.get("uninterrupted_sha256"),
                    "right": snapshot.get("replayed_sha256"),
                }
            )
    gates = [
        Gate("exact action identity", not _has(mismatches, "action_identity"), "profile order and labels"),
        Gate("exact policy observations", not _has(mismatches, "observation_sha256"), "all lane/step hashes"),
        Gate("exact raw RGB frames", not _has(mismatches, "raw_frame"), "all lane/step hashes and shapes"),
        Gate(
            "exact rewards" if profile.native_transition_exact else "matched rewards",
            not _has(mismatches, "rewards"),
            "bit-exact numeric values"
            if profile.native_transition_exact
            else f"absolute tolerance {FLOAT_TOLERANCE:g}",
        ),
        Gate("matched terminations and truncations", not _has_any(mismatches, ("terminations", "truncations")), "exact"),
        Gate("matched resets", not _has_any(mismatches, ("reset_lanes", "reset_points")), "exact selective reset points"),
        Gate("matched completion", not _has(mismatches, "completion_step"), "exact completion step"),
        Gate("matched semantic infos", not _has(mismatches, ".infos"), "integer exact; declared float absolute tolerance"),
        Gate(
            "exact emulator RAM",
            not _has(mismatches, "ram_"),
            "all declared lane/step hashes and shapes",
        ),
        Gate(
            "exact snapshot continuation",
            not _has(mismatches, "snapshot_continuation"),
            "uninterrupted suffix equals restore-and-replay suffix",
        ),
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
        _float_list(
            mismatches,
            f"{prefix}.reward",
            [lhs.get("reward")],
            [rhs.get("reward")],
            exact=profile.native_transition_exact,
        )
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


def compare_reset_distributions(
    authority: dict[str, Any], candidate: dict[str, Any], *, maximum_cdf_distance: float = 0.15
) -> dict[str, Any]:
    """Compare reset randomness statistically and reset outcomes exactly by noop count."""

    errors: list[dict[str, Any]] = []
    maximum = int(authority.get("maximum", 0))
    if maximum <= 0 or candidate.get("maximum") != maximum:
        errors.append({"field": "maximum", "authority": maximum, "candidate": candidate.get("maximum")})
    authority_samples = list(authority.get("samples", []))
    candidate_samples = list(candidate.get("samples", []))
    if len(authority_samples) < 32 or len(authority_samples) != len(candidate_samples):
        errors.append(
            {
                "field": "sample_count",
                "authority": len(authority_samples),
                "candidate": len(candidate_samples),
            }
        )
    authority_counts = np.asarray([item.get("count", -1) for item in authority_samples], dtype=np.int64)
    candidate_counts = np.asarray([item.get("count", -1) for item in candidate_samples], dtype=np.int64)
    if maximum > 0 and (
        np.any((authority_counts < 1) | (authority_counts > maximum))
        or np.any((candidate_counts < 1) | (candidate_counts > maximum))
    ):
        errors.append({"field": "count_bounds", "authority": "1..maximum", "candidate": "1..maximum"})
    cdf_distance = 1.0
    if maximum > 0 and authority_counts.size and candidate_counts.size:
        authority_histogram = np.bincount(authority_counts, minlength=maximum + 1)[1 : maximum + 1]
        candidate_histogram = np.bincount(candidate_counts, minlength=maximum + 1)[1 : maximum + 1]
        authority_cdf = np.cumsum(authority_histogram) / authority_counts.size
        candidate_cdf = np.cumsum(candidate_histogram) / candidate_counts.size
        cdf_distance = float(np.max(np.abs(authority_cdf - candidate_cdf)))
        if cdf_distance > maximum_cdf_distance:
            errors.append(
                {
                    "field": "cdf_distance",
                    "authority": maximum_cdf_distance,
                    "candidate": cdf_distance,
                }
            )
    def outcomes(samples: list[dict[str, Any]]) -> dict[int, set[str]]:
        grouped: dict[int, set[str]] = {}
        for sample in samples:
            grouped.setdefault(int(sample.get("count", -1)), set()).add(
                canonical_json_hash(
                    {
                        "observation_sha256": sample.get("observation_sha256"),
                        "raw_frame_sha256": sample.get("raw_frame_sha256"),
                        "infos": sample.get("infos"),
                    }
                )
            )
        return grouped

    authority_outcomes = outcomes(authority_samples)
    candidate_outcomes = outcomes(candidate_samples)
    complete_coverage = set(authority_outcomes) == set(range(1, maximum + 1)) and set(
        candidate_outcomes
    ) == set(range(1, maximum + 1))
    if not complete_coverage:
        errors.append(
            {
                "field": "complete_count_coverage",
                "authority_counts": sorted(authority_outcomes),
                "candidate_counts": sorted(candidate_outcomes),
            }
        )
    if authority_outcomes != candidate_outcomes:
        errors.append(
            {
                "field": "outcomes_by_noop_count",
                "authority_counts": sorted(authority_outcomes),
                "candidate_counts": sorted(candidate_outcomes),
            }
        )
    return {
        "passed": not errors,
        "sample_count": min(len(authority_samples), len(candidate_samples)),
        "maximum": maximum,
        "cdf_distance": cdf_distance,
        "maximum_cdf_distance": maximum_cdf_distance,
        "complete_count_coverage": complete_coverage,
        "first_mismatches": errors[:20],
    }


def _required_trace_evidence(
    mismatches: list[dict[str, Any]],
    side: str,
    trace: dict[str, Any],
    *,
    require_snapshot: bool,
) -> None:
    initial = trace.get("initial")
    if not isinstance(initial, dict):
        mismatches.append({"field": f"evidence.{side}.initial", "left": "required", "right": None})
        return
    for field in ("observation_sha256", "raw_frame_sha256", "raw_frame_shapes", "infos"):
        if field not in initial or not isinstance(initial[field], list) or not initial[field]:
            mismatches.append(
                {"field": f"evidence.{side}.initial.{field}", "left": "required", "right": initial.get(field)}
            )
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        mismatches.append({"field": f"evidence.{side}.steps", "left": "required", "right": steps})
    else:
        required = {
            "observation_sha256",
            "raw_frame_sha256",
            "rewards",
            "terminations",
            "truncations",
            "infos",
            "reset_lanes",
        }
        for index, step in enumerate(steps):
            missing = sorted(required - set(step)) if isinstance(step, dict) else sorted(required)
            if missing:
                mismatches.append(
                    {
                        "field": f"evidence.{side}.steps[{index}]",
                        "left": "required",
                        "right": missing,
                    }
                )
                break
    if require_snapshot and not isinstance(trace.get("snapshot_continuation"), dict):
        mismatches.append(
            {
                "field": f"evidence.{side}.snapshot_continuation",
                "left": "required",
                "right": trace.get("snapshot_continuation"),
            }
        )


def _same(mismatches: list[dict[str, Any]], field: str, left: Any, right: Any) -> None:
    if left != right:
        mismatches.append({"field": field, "left": left, "right": right})


def _float_list(
    mismatches: list[dict[str, Any]],
    field: str,
    left: list[Any],
    right: list[Any],
    *,
    exact: bool = False,
) -> None:
    if len(left) != len(right):
        mismatches.append({"field": field, "left": left, "right": right})
        return
    for index, (lhs, rhs) in enumerate(zip(left, right, strict=True)):
        if (
            float(lhs) != float(rhs)
            if exact
            else not math.isclose(
                float(lhs), float(rhs), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
            )
        ):
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
                _float_list(
                    mismatches,
                    f"{field}[{lane}].{key}",
                    [lhs[key]],
                    [rhs[key]],
                    exact=profile.native_transition_exact,
                )


def _safe_ownership(trace: dict[str, Any]) -> bool:
    return trace.get("environment", {}).get("observation", {}).get("ownership") != "unsafe_view"


def _has(mismatches: list[dict[str, Any]], fragment: str) -> bool:
    return any(fragment in str(item.get("field")) for item in mismatches)


def _has_any(mismatches: list[dict[str, Any]], fragments: tuple[str, ...]) -> bool:
    return any(_has(mismatches, fragment) for fragment in fragments)
