"""Readable publication views derived from an immutable benchmark proof."""

from __future__ import annotations

import argparse
import math
from html import escape
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from turbobench.proofs import require_proof
from turbobench.showcase import BACKGROUND, NAMES, YELLOW, scaling_chart
from turbobench.util import read_json, sha256_file, write_json


def peak_rows(result: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[int]]:
    """Keep the measured prefix through the first maximum candidate median SPS."""
    rows = sorted(result["comparison"]["shapes"].items(), key=lambda row: int(row[0]))
    peak = max(range(len(rows)), key=lambda i: rows[i][1]["statistics"]["median_right_sps"])
    return rows[: peak + 1], [int(shape) for shape, _ in rows[peak + 1 :]]


def readme_chart(result: dict[str, Any], *, diagnostic: bool) -> str:
    rows, _ = peak_rows(result)
    peak = max(
        row["statistics"][f"median_{side}_sps"] for _, row in rows for side in ("left", "right")
    )
    magnitude = 10 ** math.floor(math.log10(peak / 4))
    step = next(n * magnitude for n in (1, 2, 5, 10) if n * magnitude >= peak / 4)
    maximum = math.ceil(peak / step) * step
    height = 280 + len(rows) * 108 + 16
    colors = {"left": "#acbde1", "right": YELLOW}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{height}" '
        f'viewBox="0 0 800 {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Environment throughput through the candidate peak</title>',
        '<desc id="description">Paired horizontal bars use one linear scale from zero. '
        "Every measured count through the first candidate throughput maximum is shown. "
        "Later counts remain in the complete benchmark proof. "
        "Speedup uses paired candidate/upstream ratios. Exact values and uncertainty remain in the benchmark report.</desc>",
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
        y = 298 + index * 108
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
                f'font-size="28" fill="{colors[side]}">{sps:,.0f}</text>'
            )
        svg.append(f'<path d="M32 {y + 89} H768" stroke="#30363d"/></g>')
    svg.append("</g></svg>")
    return "\n".join(svg) + "\n"


def full_publication_chart(result: dict[str, Any], *, diagnostic: bool) -> str:
    """Simplify labels in the complete view without altering archived proof charts."""
    namespace = "http://www.w3.org/2000/svg"
    ElementTree.register_namespace("", namespace)
    svg = ElementTree.fromstring(scaling_chart(result, diagnostic=diagnostic))
    for group in svg.iter():
        children = list(group)
        for index, element in enumerate(children):
            if "data-provider" in element.attrib:
                children[index + 1].text = f"{float(element.attrib['data-sps']):,.0f}"
                title = element.find(f"{{{namespace}}}title")
                if title is not None:
                    title.text = (
                        f"n_envs={group.attrib['data-n-envs']}, {element.attrib['data-provider']}: "
                        f"{float(element.attrib['data-sps']):,.0f} SPS"
                    )
            elif element.tag == f"{{{namespace}}}text" and (element.text or "").startswith(
                ("95% paired CI:", "One sample;", "Bars:")
            ):
                group.remove(element)
    svg.set("height", "610")
    svg.set("viewBox", f"0 0 {svg.attrib['width']} 610")
    ElementTree.indent(svg)
    return ElementTree.tostring(svg, encoding="unicode") + "\n"


def export_readme_chart(benchmark: Path, output: Path, *, full: bool = False) -> dict[str, Any]:
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
    if full:
        rows = sorted(result["comparison"]["shapes"].items(), key=lambda row: int(row[0]))
        omitted = []
    renderer = full_publication_chart if full else readme_chart
    output.write_text(renderer(result, diagnostic=diagnostic))
    provenance = {
        "format": "turbobench.publication-chart/v1",
        "benchmark_id": manifest["proof_id"],
        "result_sha256": sha256_file(benchmark / "result.json"),
        "renderer_sha256": sha256_file(Path(__file__)),
        "selection": "all-measured-counts"
        if full
        else "measured-prefix-through-first-candidate-median-sps-maximum",
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
    parser.add_argument("--full", action="store_true", help="export every measured count")
    args = parser.parse_args()
    export_readme_chart(args.benchmark, args.output, full=args.full)
