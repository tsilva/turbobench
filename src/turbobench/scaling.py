"""Deterministic, evidence-bound stopping rules for adaptive vector scaling."""

from __future__ import annotations

import math
from typing import Any

from turbobench.model import Profile


def scaling_rule(profile: Profile) -> dict[str, Any] | None:
    workload = profile.resolved_workload
    return workload["protocol"].get("scaling") if workload is not None else None


def scaling_progress(profile: Profile, shapes: dict[str, Any]) -> dict[str, Any]:
    """Recompute every decision from shape-local provider medians, never speedup."""
    rule = scaling_rule(profile)
    if rule is None:
        raise ValueError("profile has no adaptive scaling rule")
    counts = sorted(map(int, shapes))
    if not counts or tuple(counts) != profile.measurement_shapes[: len(counts)]:
        raise ValueError("adaptive counts must be a nonempty prefix of the locked schedule")
    best: dict[str, float] = {}
    streaks = dict.fromkeys(("left", "right"), 0)
    history = []
    stop = None
    for index, count in enumerate(counts):
        if stop is not None:
            raise ValueError("adaptive evidence continues after its required stop")
        providers = {}
        for side in ("left", "right"):
            sps = float(shapes[str(count)]["statistics"][f"median_{side}_sps"])
            if not math.isfinite(sps) or sps <= 0:
                raise ValueError("adaptive throughput must be finite and positive")
            previous_best = best.get(side)
            gain = None if previous_best is None else sps / previous_best - 1
            streaks[side] = (
                streaks[side] + 1 if gain is not None and gain < rule["minimum_gain"] else 0
            )
            status = (
                "downgrade"
                if gain is not None and gain <= -rule["downgrade_fraction"]
                else "plateau"
                if streaks[side] >= rule["plateau_confirmations"]
                else "pending_plateau"
                if gain is not None and gain < rule["minimum_gain"]
                else "improving"
                if gain is not None
                else "baseline"
            )
            best[side] = max(previous_best or sps, sps)
            providers[side] = {
                "median_sps": sps,
                "previous_best_sps": previous_best,
                "gain_over_previous_best": gain,
                "low_gain_streak": streaks[side],
                "status": status,
            }
        history.append({"n_envs": count, "providers": providers})
        if all(p["status"] in {"plateau", "downgrade"} for p in providers.values()):
            stop = "both_providers_saturated"
        elif index + 1 == len(profile.measurement_shapes):
            stop = "safety_cap"
    return {
        "schema": "turbobench.scaling-decision/v1",
        "rule": rule,
        "history": history,
        "stop_reason": stop,
        "complete": stop == "both_providers_saturated",
    }


def verify_scaling(profile: Profile, result: dict[str, Any]) -> tuple[int, ...]:
    record = scaling_progress(profile, result["comparison"]["shapes"])
    if result.get("scaling") != record or record["stop_reason"] is None:
        raise ValueError("adaptive stopping record is missing, incomplete, or inconsistent")
    if result.get("schema") != "turbobench.result/v4":
        raise ValueError("adaptive full results require result/v4")
    if not record["complete"] and result["claim"]["status"] != "diagnostic":
        raise ValueError("safety cap cannot support a completed scaling claim")
    return tuple(row["n_envs"] for row in record["history"])
