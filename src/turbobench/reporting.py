"""Human-readable Markdown and dependency-free SVG result views."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from turbobench.util import atomic_write


def render_report(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    left = comparison["left"]
    right = comparison["right"]
    lines = [
        f"# Turbobench: {result['profile']['id']}",
        "",
        f"- Validity passed: `{str(result['validity']['passed']).lower()}`",
        f"- Claim status: `{result['claim']['status']}`",
        f"- Shape-1 outcome: `{comparison['outcome']}`",
        f"- Promo eligible: `{str(result['promo']['eligible']).lower()}`",
        f"- Execution protocol: `{result.get('execution_protocol', 'legacy-contaminable')}`",
        f"- Left: `{left['provider']}=={left['version']}`",
        f"- Right: `{right['provider']}=={right['version']}`",
        "",
        "## Shape-local results",
        "",
        "| Envs | Left median SPS | Right median SPS | Left/right ratio | 95% paired CI | Outcome |",
        "| ---: | ---: | ---: | ---: | :---: | :--- |",
    ]
    for shape, payload in sorted(comparison["shapes"].items(), key=lambda item: int(item[0])):
        stats = payload["statistics"]
        lower, upper = stats["bootstrap"]["ci"]
        lines.append(
            f"| {shape} | {stats['median_left_sps']:,.1f} | {stats['median_right_sps']:,.1f} | "
            f"{stats['median_paired_ratio_left_over_right']:.4f}× | [{lower:.4f}, {upper:.4f}] | "  # noqa: RUF001
            f"{stats['outcome']} |"
        )
    lines.extend(("", "No SPS values are aggregated across shapes. Shape 1 is the promo basis.", ""))
    lines.extend(("## Validity gates", ""))
    for gate in result["validity"]["gates"]:
        mark = "PASS" if gate["passed"] else "FAIL"
        lines.append(f"- **{mark}** — {gate['name']}: {gate['detail']}")
    lines.extend(
        (
            "",
            "## Method",
            "",
            "Each official shape uses one unmeasured warmup pair followed by seven alternating AB/BA measured pairs. "
            "Every invocation contains three repetitions; invocation medians form paired ratios and a deterministic "
            "20,000-resample bootstrap 95% confidence interval.",
            "",
            "Contract validation, correctness traces, warmups, timed measurements, and promotional replay use "
            "phase-isolated provider processes and fresh environment instances. Each workload evidence record "
            "references the successful attestation for its exact execution configuration.",
            "",
            "Timed SPS includes preprocessing, IPC, infos, terminal detection, and selective resets. It excludes "
            "construction, initial reset, action generation, warmup, correctness replay, rendering, and encoding.",
            "",
        )
    )
    return "\n".join(lines)


def render_chart(result: dict[str, Any]) -> str:
    shapes = sorted(result["comparison"]["shapes"].items(), key=lambda item: int(item[0]))
    maximum = max(
        max(float(item[1]["statistics"]["median_left_sps"]), float(item[1]["statistics"]["median_right_sps"]))
        for item in shapes
    )
    width, height = 960, 140 + len(shapes) * 115
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        '<text x="40" y="48" fill="#f8fafc" font-family="sans-serif" font-size="28" font-weight="700">Shape-local median SPS</text>',
        '<text x="40" y="78" fill="#94a3b8" font-family="sans-serif" font-size="15">No aggregation across vector shapes</text>',
    ]
    left_label = escape(result["comparison"]["left"]["provider"])
    right_label = escape(result["comparison"]["right"]["provider"])
    for row, (shape, payload) in enumerate(shapes):
        y = 120 + row * 115
        left = float(payload["statistics"]["median_left_sps"])
        right = float(payload["statistics"]["median_right_sps"])
        left_width = 650 * left / maximum
        right_width = 650 * right / maximum
        elements.extend(
            (
                f'<text x="40" y="{y + 18}" fill="#e2e8f0" font-family="sans-serif" font-size="16">{shape} envs</text>',
                f'<rect x="160" y="{y}" width="{left_width:.2f}" height="28" rx="4" fill="#38bdf8"/>',
                f'<text x="{170 + left_width:.2f}" y="{y + 20}" fill="#e2e8f0" font-family="sans-serif" font-size="14">{left_label} {left:,.0f}</text>',
                f'<rect x="160" y="{y + 38}" width="{right_width:.2f}" height="28" rx="4" fill="#fbbf24"/>',
                f'<text x="{170 + right_width:.2f}" y="{y + 58}" fill="#e2e8f0" font-family="sans-serif" font-size="14">{right_label} {right:,.0f}</text>',
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def write_views(bundle: Path, result: dict[str, Any]) -> None:
    atomic_write(bundle / "report.md", render_report(result))
    atomic_write(bundle / "chart.svg", render_chart(result))
