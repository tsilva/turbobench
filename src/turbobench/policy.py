"""Validate imported policy cadence before timing or rendering its actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_policy_frame_skip(
    capture: Mapping[str, Any],
    training_frame_skip: int,
    *,
    benchmark_frame_skip: int | None = None,
    action_stream: Mapping[str, Any] | None = None,
) -> int:
    """Reject missing or inconsistent cadence; training metadata is authoritative.

    Raw-frame rendering uses one simulator frame per expanded action. This does
    not change the policy's frame skip: each captured decision must still occupy
    exactly the training number of consecutive raw actions.
    """
    if type(training_frame_skip) is not int or training_frame_skip < 1:
        raise ValueError("Policy training frame skip must be a positive integer")
    contract = capture.get("contract", {})
    if not isinstance(contract, Mapping) or contract.get("matches_training") is not True:
        raise ValueError("Policy playback must match its saved training contract")
    cadence = contract.get("frame_skip", {})
    if not isinstance(cadence, Mapping):
        raise ValueError("Policy playback is missing its frame-skip contract")
    values = {
        "captured training": cadence.get("training"),
        "captured playback": cadence.get("playback"),
        "policy decisions": capture.get("frame_skip"),
    }
    if benchmark_frame_skip is not None:
        values["benchmark"] = benchmark_frame_skip
    if action_stream is not None:
        values["rendered action stream"] = action_stream.get("policy_frame_skip")
    for source, value in values.items():
        if type(value) is not int or value != training_frame_skip:
            raise ValueError(
                f"Policy frame skip mismatch: training={training_frame_skip}, {source}={value!r}"
            )
    if action_stream is not None:
        noops = capture["initial"]["reset_info"]["noop_reset_count"]
        if action_stream.get("reset_noop_prefix") != noops:
            raise ValueError("Rendered action stream does not preserve the captured reset prefix")
        actions = action_stream.get("actions", [])
        decisions, remainder = divmod(len(actions) - noops, training_frame_skip)
        if remainder or decisions < 1 or decisions > len(capture["transitions"]):
            raise ValueError("Rendered action stream must end at a complete policy decision")
        table = capture["action_contract"]["requested"]["table"]
        expected = [()] * noops
        for transition in capture["transitions"][:decisions]:
            expected.extend([tuple(table[transition["native_action"]])] * training_frame_skip)
        if [tuple(action) for action in actions] != expected:
            raise ValueError("Rendered actions do not preserve the policy's trained action cadence")
    return training_frame_skip
