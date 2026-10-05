"""Readable publication views derived from an immutable benchmark proof."""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path
from typing import Any

from turbobench.proofs import require_proof
from turbobench.showcase import BACKGROUND, NAMES, YELLOW
from turbobench.util import read_json, sha256_file, write_json


def peak_rows(result: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[int]]:
    """Keep the measured prefix through the first maximum candidate median SPS."""
    rows = sorted(result["comparison"]["shapes"].items(), key=lambda row: int(row[0]))
    peak = max(range(len(rows)), key=lambda i: rows[i][1]["statistics"]["median_right_sps"])
    return rows[: peak + 1], [int(shape) for shape, _ in rows[peak + 1 :]]


def readme_chart(result: dict[str, Any], *, diagnostic: bool) -> str:
    rows, omitted = peak_rows(result)
    peak = max(
        row["statistics"][f"median_{side}_sps"] for _, row in rows for side in ("left", "right")
    )
    magnitude = 10 ** math.floor(math.log10(peak / 4))
    step = next(n * magnitude for n in (1, 2, 5, 10) if n * magnitude >= peak / 4)
    maximum = math.ceil(peak / step) * step
    height = 280 + len(rows) * 140 + 112
    colors = {"left": "#acbde1", "right": YELLOW}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{height}" '
        f'viewBox="0 0 800 {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Environment throughput through the candidate peak</title>',
        '<desc id="description">Paired horizontal bars use one linear scale from zero. '
        "Every measured count through the first candidate throughput maximum is shown. "
        "Later counts remain in the complete benchmark proof. "
        "Speedup and 95% intervals use paired candidate/upstream ratios.</desc>",
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        '<g font-family="Arial, Helvetica, sans-serif" fill="#f0f3f6">',
        '<text x="32" y="55" font-size="38" font-weight="700">Scaling to peak throughput</text>',
        '<text x="32" y="98" font-size="26">Environment steps / second · higher is faster</text>',
    ]
    for side, y in (("left", 141), ("right", 182)):
        provider = result["comparison"][side]
        name = f"{NAMES.get(provider['provider'], provider['provider'])} {provider['version']}"
        svg.append(
            f'<rect x="32" y="{y - 20}" width="22" height="22" '
            f'fill="{colors[side]}"/><text x="66" y="{y}" font-size="28" '
            f'fill="{colors[side]}">{escape(name)}</text>'
        )
    if diagnostic:
        svg.append(
            '<text x="32" y="219" fill="#ff7b72" font-size="24">'
            "DIAGNOSTIC · no validated performance claim</text>"
        )
    else:
        svg.append(
            '<text x="32" y="219" fill="#acbde1" font-size="24">'
            "Both bars use the same linear scale, starting at zero</text>"
        )
    for i in range(round(maximum / step) + 1):
        value = i * step
        x = 36 + value / maximum * 520
        label = f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"
        svg.append(
            f'<text x="{x:.3f}" y="258" text-anchor="middle" '
            f'font-size="23" fill="#acbde1">{label}</text>'
        )
    svg.append(
        '<text x="768" y="258" text-anchor="end" font-size="23" fill="#acbde1">Median SPS</text>'
    )
    for index, (shape, row) in enumerate(rows):
        y = 298 + index * 140
        stats = row["statistics"]
        ratio = 1 / stats["median_paired_ratio_left_over_right"]
        svg.append(
            f'<g data-n-envs="{shape}"><text x="32" y="{y}" font-size="28" '
            f'font-weight="700">n_envs = {shape}</text>'
            f'<text x="768" y="{y}" text-anchor="end" fill="{YELLOW}" '
            f'font-size="28" font-weight="700">{ratio:.2f}\u00d7 speedup</text>'
        )
        for side, offset in (("left", 16), ("right", 46)):
            sps = stats[f"median_{side}_sps"]
            width = sps / maximum * 520
            svg.append(
                f'<rect data-provider="{side}" data-sps="{sps}" '
                f'x="36" y="{y + offset}" width="{width:.6f}" height="18" '
                f'fill="{colors[side]}"/>'
                f'<text x="768" y="{y + offset + 19}" text-anchor="end" '
                f'font-size="28" fill="{colors[side]}">{sps:,.1f}</text>'
            )
        ci = stats["bootstrap"]
        uncertainty = (
            f"95% paired CI: {1 / ci['ci'][1]:.2f}\u2013{1 / ci['ci'][0]:.2f}\u00d7"
            if ci
            else "One sample; no confidence interval"
        )
        svg.append(
            f'<text x="768" y="{y + 99}" text-anchor="end" font-size="24" '
            f'fill="#acbde1">{escape(uncertainty)}</text>'
            f'<path d="M32 {y + 117} H768" stroke="#30363d"/></g>'
        )
    footer = 298 + len(rows) * 140
    svg.append(
        f'<text x="32" y="{footer}" font-size="26">Candidate peak at n_envs = {rows[-1][0]}</text>'
    )
    omitted_text = (
        "Later counts omitted here: " + ", ".join(map(str, omitted))
        if omitted
        else "All measured counts are shown"
    )
    svg.append(
        f'<text x="32" y="{footer + 38}" font-size="24" fill="#acbde1">'
        f"{escape(omitted_text)}</text></g></svg>"
    )
    return "\n".join(svg) + "\n"


def export_readme_chart(benchmark: Path, output: Path) -> dict[str, Any]:
    """Verify before export; keep publication artifacts outside the immutable proof."""
    benchmark = benchmark.resolve()
    output = output.resolve()
    if output.is_relative_to(benchmark):
        raise ValueError("publication exports must be outside the immutable benchmark proof")
    manifest = require_proof(benchmark)
    if manifest["schema"] not in {"turbobench.benchmark-proof/v2", "turbobench.benchmark-proof/v3"}:
        raise ValueError("expected a benchmark proof")
    result = read_json(benchmark / "result.json")
    diagnostic = (
        result["claim"]["status"] != "official" or result["comparison"]["outcome"] == "inconclusive"
    )
    rows, omitted = peak_rows(result)
    output.write_text(readme_chart(result, diagnostic=diagnostic))
    provenance = {
        "format": "turbobench.readme-chart/v1",
        "benchmark_id": manifest["proof_id"],
        "result_sha256": sha256_file(benchmark / "result.json"),
        "renderer_sha256": sha256_file(Path(__file__)),
        "selection": "measured-prefix-through-first-candidate-median-sps-maximum",
        "shown_n_envs": [int(shape) for shape, _ in rows],
        "omitted_n_envs": omitted,
        "diagnostic": diagnostic,
        "asset": {
            "path": output.name,
            "sha256": sha256_file(output),
            "size": output.stat().st_size,
        },
    }
    write_json(output.with_suffix(".json"), provenance)
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_readme_chart(args.benchmark, args.output)
