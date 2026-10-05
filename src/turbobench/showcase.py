"""The reusable GitHub-dark arcade showcase renderer; all numbers come from proofs."""

from __future__ import annotations

import math
import shutil
import tempfile
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from turbobench.promo import _probe, _run
from turbobench.util import sha256_file

SIZE = (1672, 940)
BACKGROUND = "#0d1117"
YELLOW = "#ffcc22"
MUTED = "#606975"
PANEL_SIZE = (544, 644)
PANEL_POSITIONS = ((106, 171), (1025, 171))
NAMES = {
    "stable-retro": "Stable Retro",
    "env-breakoutatari2600-turbo-native": "BreakoutAtari2600-turbo",
    "env-stableretro-turbo": "Stable Retro Turbo",
    "env-vizdoom-turbo": "ViZDoom-turbo",
    "vizdoom": "ViZDoom",
}
COMPARISON_STYLE = "comparison-style/v3"
COMPARISON_STYLES = {"comparison-style/v1", "comparison-style/v2", COMPARISON_STYLE}


def _pixel(canvas: Image.Image, text: str, center: tuple[int, int], scale: int, color: str) -> None:
    font = ImageFont.truetype(
        str(files("turbobench").joinpath("fonts", "PressStart2P-Regular.ttf")), 8
    )
    box = font.getbbox(text)
    mask = Image.new("1", (max(1, box[2] - box[0]), max(1, box[3] - box[1])))
    ImageDraw.Draw(mask).text((-box[0], -box[1]), text, font=font, fill=1)
    mask = mask.resize((mask.width * scale, mask.height * scale), Image.Resampling.NEAREST)
    canvas.paste(color, (center[0] - mask.width // 2, center[1]), mask)


def draw_card(
    result: dict[str, Any], profile: Any, *, diagnostic: bool, style: str = COMPARISON_STYLE
) -> Image.Image:
    if style not in COMPARISON_STYLES:
        raise ValueError(f"unsupported comparison style: {style}")
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    title = "Same Actions" if style == COMPARISON_STYLE else "Same Policy, Same Actions"
    _pixel(image, title, (836, 35), 3, "#f0f3f6")
    colors = ("#ff4149", "#ff8822", "#ffdd00", "#00de78", "#009cff", "#006dff")
    accent_origins = (594, 978) if style == COMPARISON_STYLE else (438, 1134)
    for side, origin in enumerate(accent_origins):
        for row in range(3):
            draw.rectangle(
                (origin + row * 8, 31 + row * 17, origin + 96 - row * 8, 37 + row * 17),
                fill=colors[side * 3 + row],
            )
    stats = result["comparison"]["shapes"]["1"]["statistics"]
    ratio = 1.0 / stats["median_paired_ratio_left_over_right"]
    for side, center, color, position in zip(
        ("left", "right"), (378, 1297), ("#acbde1", YELLOW), PANEL_POSITIONS, strict=True
    ):
        provider = result["comparison"][side]
        name = f"{NAMES.get(provider['provider'], provider['provider'])} {provider['version']}"
        scale = max(1, min(3, 548 // (len(name) * 8)))
        _pixel(image, name, (center, 123), scale, color)
        x, y = position
        draw.rectangle(
            (x - 4, y - 4, x + PANEL_SIZE[0] + 3, y + PANEL_SIZE[1] + 3), outline="#7185ac", width=2
        )
        draw.line((x - 7, 838, x + PANEL_SIZE[0] + 7, 838), fill="#7185ac", width=2)
        _pixel(image, f"{stats[f'median_{side}_sps']:,.0f}", (center - 26, 860), 5, "#f0f3f6")
        _pixel(image, "SPS", (center + 155, 880), 2, "#acbde1")
        accent_x = x - 13 if side == "left" else x + PANEL_SIZE[0] - 87
        draw.rectangle(
            (accent_x, 863, accent_x + 99, 869), fill="#ff4149" if side == "left" else "#00de78"
        )
        draw.rectangle(
            (
                accent_x - 7 if side == "left" else accent_x + 14,
                885,
                accent_x + 78 if side == "left" else accent_x + 110,
                891,
            ),
            fill="#ffdd00" if side == "left" else "#009cff",
        )
    draw.line((689, 370, 982, 370), fill=YELLOW, width=3)
    draw.line((689, 607, 982, 607), fill=YELLOW, width=3)
    ratio_scale = max(3, min(7, 295 // (len(f"{ratio:.2f}x") * 8)))
    _pixel(
        image,
        f"{ratio:.2f}x",
        (836, 428),
        ratio_scale,
        YELLOW,
    )
    speedup_y = 428 + 8 * ratio_scale + 12 if style == COMPARISON_STYLE else 548
    _pixel(image, "speedup", (836, speedup_y), 3, YELLOW)
    settings = [
        "n_envs = 1",
        f"frameskip = {profile.frame_skip}",
        f"framestack = {profile.frame_stack}",
        f"resize = {profile.resize[0]}x{profile.resize[1]}",
        f"grayscale = {str(profile.grayscale).lower()}",
        f"resize_algo = {profile.resize_algorithm}",
        f"maxpool = {str(profile.maxpool_last_two).lower()}",
    ]
    if profile.crop_top:
        settings.append(
            f"{'mask_top' if profile.crop_mode == 'mask' else 'crop_top'} = {profile.crop_top}"
        )
    for index, setting in enumerate(settings):
        _pixel(image, setting, (836, 633 + index * 25), 2, MUTED)
    if diagnostic:
        # Visible marking stays outside gameplay and the muted recipe block.
        _pixel(image, "SMOKE / DIAGNOSTIC - NO PERFORMANCE CLAIM", (836, 86), 2, "#c8837c")
    return image


def _encode(
    card: Path, records: dict[str, Any], paths: dict[str, Path], ratio: float, output: Path
) -> int:
    total = (
        math.ceil(
            max(
                records["left"]["frame_count"] / 240,
                records["right"]["frame_count"] / (240 * ratio),
            )
            * 60
        )
        + 120
    )
    argv = ["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "60", "-i", str(card)]
    graph = []
    for index, side in enumerate(("left", "right"), 1):
        record = records[side]
        argv += [
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{record['frame_width']}x{record['frame_height']}",
            "-framerate",
            "60",
            "-i",
            str(paths[side]),
        ]
        speed = 4.0 * (ratio if side == "right" else 1.0)
        graph.append(
            f"[{index}:v]scale={PANEL_SIZE[0]}:{PANEL_SIZE[1]}:force_original_aspect_ratio=decrease:flags=neighbor,pad={PANEL_SIZE[0]}:{PANEL_SIZE[1]}:(ow-iw)/2:(oh-ih)/2:black,settb=AVTB,setpts=(PTS-STARTPTS)/{speed:.12f},tpad=stop_mode=clone:stop_duration={total / 60:.6f},fps=60:round=up:start_time=0[{side}]"
        )
    lx, ly = PANEL_POSITIONS[0]
    rx, ry = PANEL_POSITIONS[1]
    graph += [
        f"[0:v][left]overlay={lx}:{ly}:shortest=1[tmp]",
        f"[tmp][right]overlay={rx}:{ry}:shortest=1[out]",
    ]
    argv += [
        "-filter_complex",
        ";".join(graph),
        "-map",
        "[out]",
        "-an",
        "-r",
        "60",
        "-frames:v",
        str(total),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run(argv)
    return total


def _webp(mp4: Path, output: Path) -> dict[str, Any]:
    if not shutil.which("img2webp"):
        raise RuntimeError("lossless animation requires img2webp (macOS: brew install webp)")
    with tempfile.TemporaryDirectory(prefix="turbobench-webp-") as directory:
        root = Path(directory)
        _run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-vf",
                "fps=20",
                str(root / "%06d.png"),
            ]
        )
        frames = sorted(root.glob("*.png"))
        arguments = root / "frames.txt"
        arguments.write_text(
            "-loop 0 -lossless -q 75 -m 6 -d 50\n"
            + "\n".join(str(p) for p in frames)
            + "\n-o "
            + str(output)
            + "\n"
        )
        _run(["img2webp", str(arguments)])
        with Image.open(output) as animation:
            if animation.size != SIZE or animation.info.get("loop") != 0:
                raise ValueError("WebP resolution/loop validation failed")
            source_index = 0
            durations = []
            for index in range(animation.n_frames):
                animation.seek(index)
                decoded = animation.convert("RGB")
                duration = animation.info.get("duration", 0)
                if duration <= 0 or duration % 50:
                    raise ValueError("WebP frame cadence mismatch")
                durations.append(duration)
                for _ in range(duration // 50):
                    with Image.open(frames[source_index]) as source:
                        if decoded.tobytes() != source.convert("RGB").tobytes():
                            raise ValueError("lossless WebP decoded-frame mismatch")
                    source_index += 1
            if source_index != len(frames):
                raise ValueError("WebP duration differs from source video")
            return {
                "width": SIZE[0],
                "height": SIZE[1],
                "frames": animation.n_frames,
                "duration_ms": sum(durations),
                "loop": 0,
                "lossless_source_frames_verified": len(frames),
            }


def video_probe(path: Path) -> dict[str, Any]:
    probe = _probe(path)
    keys = (
        "width",
        "height",
        "codec_name",
        "codec_type",
        "pix_fmt",
        "r_frame_rate",
        "nb_frames",
        "duration",
    )
    return {
        "streams": [
            {key: stream[key] for key in keys if key in stream} for stream in probe["streams"]
        ],
        "duration": probe["format"]["duration"],
    }


def generate_showcase_assets(
    root: Path, result: Any, profile: Any, records: Any, paths: Any, *, diagnostic: bool
) -> dict[str, Any]:
    media = root / "media"
    media.mkdir(exist_ok=True)
    card = media / "card.png"
    draw_card(result, profile, diagnostic=diagnostic).save(card)
    ratio = (
        1 / result["comparison"]["shapes"]["1"]["statistics"]["median_paired_ratio_left_over_right"]
    )
    mp4, webp, poster = media / "comparison.mp4", media / "comparison.webp", media / "poster.png"
    total = _encode(card, records, paths, ratio, mp4)
    probe = video_probe(mp4)
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-frames:v", "1", str(poster)])
    webp_record = _webp(mp4, webp)
    assets = {
        "diagnostic": diagnostic,
        "ratio_right_over_left": ratio,
        "common_time_compression": 4,
        "end_hold_seconds": 2,
        "video_frames": total,
        "video_probe": probe,
        "webp": webp_record,
        "files": {
            p.relative_to(root).as_posix(): sha256_file(p) for p in (card, mp4, webp, poster)
        },
    }
    verify_assets(root, assets, result, diagnostic)
    return assets


def verify_assets(
    root: Path, assets: Any, result: Any, diagnostic: bool, *, style: str = COMPARISON_STYLE
) -> None:
    ratio = (
        1 / result["comparison"]["shapes"]["1"]["statistics"]["median_paired_ratio_left_over_right"]
    )
    if (
        assets["diagnostic"] is not diagnostic
        or assets["ratio_right_over_left"] != ratio
        or assets["common_time_compression"] != 4
        or assets["end_hold_seconds"] != 2
    ):
        raise ValueError("media ratio/mode binding mismatch")
    expected = {
        f"media/{name}" for name in ("card.png", "comparison.mp4", "comparison.webp", "poster.png")
    }
    if set(assets["files"]) != expected:
        raise ValueError("showcase asset inventory mismatch")
    for name, digest in assets["files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("derived asset digest mismatch")
    if assets["video_probe"] != video_probe(root / "media/comparison.mp4"):
        raise ValueError("video probe differs from encoded bytes")
    stream = assets["video_probe"]["streams"][0]
    if any(
        stream.get(k) != value
        for k, value in {
            "width": SIZE[0],
            "height": SIZE[1],
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "r_frame_rate": "60/1",
            "nb_frames": str(assets["video_frames"]),
        }.items()
    ) or any(s["codec_type"] == "audio" for s in assets["video_probe"]["streams"]):
        raise ValueError("video export contract mismatch")
    with Image.open(root / "media/comparison.webp") as animation:
        duration = 0
        for i in range(animation.n_frames):
            animation.seek(i)
            animation.load()
            duration += animation.info.get("duration", 0)
        if (
            animation.size != SIZE
            or animation.info.get("loop") != 0
            or duration != assets["webp"]["duration_ms"]
            or animation.n_frames != assets["webp"]["frames"]
        ):
            raise ValueError("WebP export contract mismatch")
    from turbobench.workloads import bundle_profile

    profile = bundle_profile(root / "benchmark" if (root / "benchmark").exists() else root, result)
    with Image.open(root / "media/card.png") as card:
        if (
            card.convert("RGB").tobytes()
            != draw_card(result, profile, diagnostic=diagnostic, style=style).tobytes()
        ):
            raise ValueError("card does not match evidence and diagnostic marking")


def scaling_chart(result: Any, *, diagnostic: bool, style: str = COMPARISON_STYLE) -> str:
    if style == "comparison-style/v1":
        return _legacy_scaling_chart(result, diagnostic=diagnostic)
    if style not in {"comparison-style/v2", COMPARISON_STYLE}:
        raise ValueError(f"unsupported comparison style: {style}")
    rows = sorted(result["comparison"]["shapes"].items(), key=lambda pair: int(pair[0]))
    peak = max(
        payload["statistics"][f"median_{side}_sps"]
        for _, payload in rows
        for side in ("left", "right")
    )
    magnitude = 10 ** math.floor(math.log10(peak / 4))
    step = next(value * magnitude for value in (1, 2, 5, 10) if value * magnitude >= peak / 4)
    maximum = math.ceil(peak * 1.12 / step) * step
    # Give adaptive sweeps enough room for every measured label and paired CI.
    width = max(1672, len(rows) * 430 + 220) if "scaling" in result else 1672
    plot_left, plot_right, baseline, plot_height = 140, width - 80, 500, 300
    colors = {"left": "#acbde1", "right": YELLOW}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="700" viewBox="0 0 {width} 700" role="img" aria-labelledby="title description">',
        '<title id="title">Environment throughput by environment count</title>',
        '<desc id="description">Side-by-side upstream and candidate bars show median steps per second on a linear axis starting at zero. Speedups use shape-local paired ratios.</desc>',
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        '<g font-family="monospace" fill="#acbde1">',
        '<text x="80" y="60" font-size="32">Throughput by environment count</text>',
        '<text x="80" y="98" font-size="20">Median steps per second (SPS); higher is faster</text>',
    ]
    if diagnostic:
        elements.append(
            '<text x="80" y="130" fill="#c8837c" font-size="20">DIAGNOSTIC - no validated performance claim</text>'
        )
    for side, x in (("left", 140), ("right", 860)):
        provider = result["comparison"][side]
        label = f"{NAMES.get(provider['provider'], provider['provider'])} {provider['version']}"
        elements.append(
            f'<rect x="{x}" y="155" width="20" height="20" fill="{colors[side]}"/>'
            f'<text x="{x + 32}" y="172" font-size="20">{escape(label)}</text>'
        )
    for index in range(round(maximum / step) + 1):
        value = index * step
        y = baseline - value / maximum * plot_height
        label = f"{value:,.0f}" if step >= 1 else f"{value:.{-math.floor(math.log10(step))}f}"
        elements.append(
            f'<path d="M{plot_left} {y:.3f} H{plot_right}" stroke="{MUTED}" stroke-opacity="0.4"/>'
            f'<text x="{plot_left - 16}" y="{y + 6:.3f}" text-anchor="end" font-size="18">{label}</text>'
        )
    group_width = (plot_right - plot_left) / len(rows)
    bar_width = min(150, group_width * 0.28)
    gap = min(24, group_width * 0.06)
    for index, (shape, payload) in enumerate(rows):
        center = plot_left + group_width * (index + 0.5)
        stats = payload["statistics"]
        elements.append(f'<g data-n-envs="{shape}">')
        for side, x in (("left", center - gap / 2 - bar_width), ("right", center + gap / 2)):
            sps = stats[f"median_{side}_sps"]
            height = sps / maximum * plot_height
            top = baseline - height
            elements.append(
                f'<rect data-provider="{side}" data-sps="{sps}" x="{x:.3f}" y="{top:.6f}" width="{bar_width:.3f}" height="{height:.6f}" fill="{colors[side]}">'
                f"<title>n_envs={shape}, {side}: {sps:,.1f} SPS</title></rect>"
                f'<text x="{x + bar_width / 2:.3f}" y="{top - 12:.3f}" text-anchor="middle" fill="{colors[side]}" font-size="22">{sps:,.1f}</text>'
            )
        ratio = 1 / stats["median_paired_ratio_left_over_right"]
        ci = stats["bootstrap"]
        uncertainty = (
            f"95% paired CI: {1 / ci['ci'][1]:.2f}x - {1 / ci['ci'][0]:.2f}x"
            if ci
            else "One sample; no confidence interval"
        )
        elements.append(
            f'<text x="{center:.3f}" y="540" text-anchor="middle" font-size="24">n_envs={shape}</text>'
            f'<text x="{center:.3f}" y="580" text-anchor="middle" fill="{YELLOW}" font-size="22">{ratio:.2f}x speedup</text>'
            f'<text x="{center:.3f}" y="612" text-anchor="middle" font-size="17">{escape(uncertainty)}</text></g>'
        )
    elements.append(
        f'<text x="80" y="670" font-size="16" fill="{MUTED}">Bars: shape-local median SPS. Speedup: paired candidate/upstream ratio; counts are not aggregated.</text></g></svg>'
    )
    return "\n".join(elements) + "\n"


def _legacy_scaling_chart(result: Any, *, diagnostic: bool) -> str:
    """Retain the exact v1 SVG for verification of already archived proofs."""
    rows = sorted(result["comparison"]["shapes"].items(), key=lambda pair: int(pair[0]))
    ratios = [
        1 / payload["statistics"]["median_paired_ratio_left_over_right"] for _, payload in rows
    ]
    upper = [
        1 / payload["statistics"]["bootstrap"]["ci"][0]
        if payload["statistics"]["bootstrap"]
        else ratios[i]
        for i, (_, payload) in enumerate(rows)
    ]
    maximum = max(1, *upper) * 1.15
    height = 410 + len(rows) * 35
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1672" height="{height}" viewBox="0 0 1672 {height}">',
        f'<rect width="100%" height="100%" fill="{BACKGROUND}"/>',
        '<g font-family="monospace" fill="#acbde1">',
        '<text x="80" y="60" font-size="32">Speedup as n_envs grows</text>',
    ]
    if diagnostic:
        elements.append(
            '<text x="80" y="95" fill="#c8837c" font-size="20">SMOKE / DIAGNOSTIC - one sample; no CI or performance claim</text>'
        )

    def y(value: float) -> float:
        return 345 - value / maximum * 220

    elements.append(
        f'<path d="M100 {y(1):.3f} H1560" stroke="{MUTED}" stroke-dasharray="8 8"/><text x="105" y="{y(1) - 8:.3f}" font-size="16">1x</text>'
    )
    for i, ((shape, payload), ratio) in enumerate(zip(rows, ratios, strict=True)):
        x = 160 + i * 1300 / max(1, len(rows) - 1)
        elements.append(
            f'<circle cx="{x:.3f}" cy="{y(ratio):.3f}" r="7" fill="{YELLOW}"/><text x="{x:.3f}" y="380" text-anchor="middle" font-size="22">n_envs={shape}</text><text x="{x:.3f}" y="{y(ratio) - 16:.3f}" text-anchor="middle" fill="{YELLOW}" font-size="22">{ratio:.2f}x</text>'
        )
        stats = payload["statistics"]
        if stats["bootstrap"]:
            lo, hi = stats["bootstrap"]["ci"]
            elements.append(
                f'<path d="M{x:.3f} {y(1 / hi):.3f} V{y(1 / lo):.3f}" stroke="{YELLOW}" stroke-width="3"/>'
            )
        label = f"n_envs={shape}    upstream={stats['median_left_sps']:,.0f} SPS    candidate={stats['median_right_sps']:,.0f} SPS"
        elements.append(f'<text x="100" y="{420 + i * 35}" font-size="19">{escape(label)}</text>')
    left, right = result["comparison"]["left"], result["comparison"]["right"]
    label = f"{right['provider']} {right['version']} / {left['provider']} {left['version']}"
    elements.append(
        f'<text x="80" y="{height - 12}" font-size="16" fill="{MUTED}">{escape(label)}; shape-local paired ratios</text></g></svg>'
    )
    return "\n".join(elements) + "\n"
