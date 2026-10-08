"""Evidence-bound comparison media generation and FFprobe validation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from turbobench.lifecycle import EXECUTION_PROTOCOL
from turbobench.util import canonical_json_hash, redact, sha256_file, write_json

FPS = 60
VIDEO_SIZE = (1280, 720)


def promo_is_eligible(result: dict[str, Any], replay_gate: dict[str, Any] | None = None) -> bool:
    return bool(
        result["validity"]["passed"]
        and result.get("execution_protocol") == EXECUTION_PROTOCOL
        and result["claim"]["status"] == "official"
        and result["comparison"]["outcome"] != "inconclusive"
        and (replay_gate or {}).get("passed")
    )


def generate_media(
    bundle: Path,
    result: dict[str, Any],
    lock: dict[str, Any],
    left_replay: dict[str, Any],
    right_replay: dict[str, Any],
    left_frames: Path,
    right_frames: Path,
    *,
    diagnostic: bool,
) -> dict[str, Any]:
    outcome = result["comparison"]["outcome"]
    eligible = promo_is_eligible(result, result["promo"].get("replay_gate"))
    if not eligible and not diagnostic:
        raise ValueError("invalid or inconclusive evidence cannot produce an unmarked promo")
    if outcome == "inconclusive" and not diagnostic:
        raise ValueError("inconclusive comparison cannot produce a normal promo")
    shape = result["comparison"]["shapes"]["1"]["statistics"]
    left_over_right = float(shape["median_paired_ratio_left_over_right"])
    if outcome == "left_faster":
        slow_result, fast_result = result["comparison"]["right"], result["comparison"]["left"]
        slow_replay, fast_replay = right_replay, left_replay
        slow_frames, fast_frames = right_frames, left_frames
        slow_sps, fast_sps = shape["median_right_sps"], shape["median_left_sps"]
        ratio = left_over_right
    elif outcome == "right_faster":
        slow_result, fast_result = result["comparison"]["left"], result["comparison"]["right"]
        slow_replay, fast_replay = left_replay, right_replay
        slow_frames, fast_frames = left_frames, right_frames
        slow_sps, fast_sps = shape["median_left_sps"], shape["median_right_sps"]
        ratio = 1.0 / left_over_right
    else:
        # Diagnostic media retains CLI order and uses no unsupported speed claim.
        slow_result, fast_result = result["comparison"]["left"], result["comparison"]["right"]
        slow_replay, fast_replay = left_replay, right_replay
        slow_frames, fast_frames = left_frames, right_frames
        slow_sps, fast_sps = shape["median_left_sps"], shape["median_right_sps"]
        ratio = 1.0
    ratio = max(float(ratio), 1.0)
    if int(slow_replay["frame_count"]) != int(fast_replay["frame_count"]):
        raise ValueError("promo replays must contain the same canonical frame count")
    media_dir = bundle / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="turbobench-media-", dir=bundle.parent) as raw_temp:
        temporary = Path(raw_temp)
        card = temporary / "card.png"
        _draw_card(
            card,
            result,
            slow_result,
            fast_result,
            float(slow_sps),
            float(fast_sps),
            ratio,
            diagnostic=diagnostic,
        )
        mp4 = temporary / "comparison.mp4"
        gif = temporary / "comparison.gif"
        _encode_mp4(
            card,
            slow_frames,
            fast_frames,
            slow_replay,
            fast_replay,
            ratio,
            mp4,
        )
        probe = _validate_mp4(
            mp4,
            expected_frames=int(slow_replay["frame_count"]),
            ratio=ratio,
            fast_frame_count=int(fast_replay["frame_count"]),
        )
        _encode_gif(mp4, gif)
        gif_probe = _probe(gif)
        gif_stream = gif_probe["streams"][0]
        if int(gif_stream.get("width", 0)) != 640:
            raise RuntimeError("promo GIF width validation failed")
        portable_probe = redact(probe)
        portable_gif_probe = redact(gif_probe)
        output_mp4 = media_dir / "comparison.mp4"
        output_gif = media_dir / "comparison.gif"
        os.replace(mp4, output_mp4)
        os.replace(gif, output_gif)
    manifest = {
        "schema": "turbobench.media/v1",
        "bundle_id": "",
        "diagnostic_watermark": bool(diagnostic),
        "profile": result["profile"],
        "providers": {"slower": slow_result, "faster": fast_result},
        "shape_1_statistics": {
            "slower_sps": slow_sps,
            "faster_sps": fast_sps,
            "uncapped_timing_ratio": ratio,
            "paired_ci_left_over_right": shape["bootstrap"]["ci"],
            "disclosure": "SPS excludes recording and encoding and comes from the timed benchmark workload",
        },
        "replay": {
            "action_sha256": slow_replay["action_stream_sha256"],
            "slower_frame_hashes_sha256": canonical_json_hash(slow_replay["frame_sha256"]),
            "faster_frame_hashes_sha256": canonical_json_hash(fast_replay["frame_sha256"]),
            "slower_transition_sha256": canonical_json_hash(slow_replay["transitions"]),
            "faster_transition_sha256": canonical_json_hash(fast_replay["transitions"]),
            "completion_step": slow_replay["completion_step"],
        },
        "lock_sha256": canonical_json_hash(lock),
        "ffmpeg": _ffmpeg_version(),
        "timing_transform": {
            "slower": "1x",
            "faster_setpts_divisor": ratio,
            "faster_final_frame_hold": True,
            "fps": FPS,
        },
        "outputs": {
            "mp4": {
                "path": "media/comparison.mp4",
                "sha256": sha256_file(media_dir / "comparison.mp4"),
                "size": (media_dir / "comparison.mp4").stat().st_size,
                "probe": portable_probe,
            },
            "gif": {
                "path": "media/comparison.gif",
                "sha256": sha256_file(media_dir / "comparison.gif"),
                "size": (media_dir / "comparison.gif").stat().st_size,
                "probe": portable_gif_probe,
            },
        },
    }
    write_json(media_dir / "media-manifest.json", manifest)
    return manifest


def _draw_card(
    output: Path,
    result: dict[str, Any],
    slow: dict[str, Any],
    fast: dict[str, Any],
    slow_sps: float,
    fast_sps: float,
    ratio: float,
    *,
    diagnostic: bool,
) -> None:
    image = Image.new("RGB", VIDEO_SIZE, "#0b1020")
    draw = ImageDraw.Draw(image)
    bold = _font(31, bold=True)
    label = _font(22, bold=True)
    regular = _font(16)
    draw.rectangle((0, 0, 1280, 8), fill="#ef4444" if diagnostic else "#fbbf24")
    headline = (
        "DIAGNOSTIC — NOT PROMOTABLE"
        if diagnostic
        else f"SAME WORKLOAD. SAME ACTIONS. {ratio:.2f}× MEASURED THROUGHPUT."  # noqa: RUF001
    )
    _center(draw, headline, 26, bold, "#fecaca" if diagnostic else "#f8fafc")
    for x, provider, sps, factor, color in (
        (60, slow, slow_sps, "1×", "#94a3b8"),  # noqa: RUF001
        (680, fast, fast_sps, f"{ratio:.2f}×", "#fbbf24"),  # noqa: RUF001
    ):
        _center(draw, provider["provider"], 91, label, color, x, x + 540)
        _center(draw, f"{provider['version']}  •  {sps:,.0f} SPS  •  {factor}", 123, regular, color, x, x + 540)
        draw.rounded_rectangle((x, 164, x + 540, 584), radius=10, fill="#020617", outline=color, width=3)
    stats = result["comparison"]["shapes"]["1"]["statistics"]
    lower, upper = stats["bootstrap"]["ci"]
    _center(draw, f"Profile {result['profile']['id']}  •  paired 95% CI [{lower:.3f}, {upper:.3f}]", 610, regular, "#cbd5e1")
    _center(draw, "SPS excludes recording/encoding and comes from the timed benchmark workload", 640, regular, "#94a3b8")
    _center(draw, "Both panels replay the exact locked providers and one canonical semantic action trajectory", 670, regular, "#64748b")
    if diagnostic:
        watermark = _font(48, bold=True)
        _center(draw, "DIAGNOSTIC EVIDENCE", 322, watermark, "#ef4444")
    image.save(output)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold
        else ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def _center(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.ImageFont,
    color: str,
    x0: int = 0,
    x1: int = 1280,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    x = x0 + ((x1 - x0) - (box[2] - box[0])) / 2
    draw.text((x, y), text, font=font, fill=color)


def _encode_mp4(
    card: Path,
    slow_frames: Path,
    fast_frames: Path,
    slow: dict[str, Any],
    fast: dict[str, Any],
    ratio: float,
    output: Path,
) -> None:
    slow_size = f"{slow['frame_width']}x{slow['frame_height']}"
    fast_size = f"{fast['frame_width']}x{fast['frame_height']}"
    filter_graph = (
        "[1:v]scale=520:400:force_original_aspect_ratio=decrease:flags=neighbor,"
        "pad=520:400:(ow-iw)/2:(oh-ih)/2:black,setpts=PTS-STARTPTS[slow];"
        "[2:v]scale=520:400:force_original_aspect_ratio=decrease:flags=neighbor,"
        f"pad=520:400:(ow-iw)/2:(oh-ih)/2:black,setpts=(PTS-STARTPTS)/{ratio:.12f},"
        "fps=60,tpad=stop_mode=clone:stop_duration=3600[fast];"
        "[0:v][slow]overlay=70:174:shortest=1[tmp];[tmp][fast]overlay=690:174:shortest=1[out]"
    )
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-loop", "1", "-framerate", str(FPS), "-i", str(card),
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", slow_size, "-framerate", str(FPS), "-i", str(slow_frames),
            "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", fast_size, "-framerate", str(FPS), "-i", str(fast_frames),
            "-filter_complex", filter_graph,
            "-map", "[out]", "-an", "-r", str(FPS), "-frames:v", str(slow["frame_count"]),
            "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
    )


def _encode_gif(mp4: Path, output: Path) -> None:
    palette = output.with_name("palette.png")
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp4),
            "-vf",
            "fps=12,scale=640:-2:flags=lanczos,palettegen=max_colors=96:stats_mode=diff",
            "-frames:v",
            "1",
            str(palette),
        ]
    )
    try:
        _run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-i",
                str(palette),
                "-filter_complex",
                "[0:v]fps=12,scale=640:-2:flags=lanczos[video];[video][1:v]paletteuse=dither=none",
                str(output),
            ]
        )
    finally:
        palette.unlink(missing_ok=True)


def _validate_mp4(
    path: Path,
    *,
    expected_frames: int,
    ratio: float,
    fast_frame_count: int,
) -> dict[str, Any]:
    probe = _probe(path)
    video = probe["streams"][0]
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 1280,
        "height": 720,
        "r_frame_rate": "60/1",
        "nb_frames": str(expected_frames),
    }
    for field, value in expected.items():
        if video.get(field) != value:
            raise RuntimeError(f"MP4 {field}={video.get(field)!r}, expected {value!r}")
    if any(stream.get("codec_type") == "audio" for stream in probe.get("streams", [])):
        raise RuntimeError("promo MP4 must be silent")
    expected_duration = expected_frames / FPS
    duration = float(video.get("duration", probe.get("format", {}).get("duration", 0)))
    if not math.isclose(duration, expected_duration, abs_tol=1.0 / FPS):
        raise RuntimeError("MP4 duration validation failed")
    if ratio > 1.0 and fast_frame_count / ratio + 2 < expected_frames:
        late_a = max(expected_duration - 1.0, 0.0)
        late_b = max(expected_duration - 2.0, 0.0)
        if not _held_frames_equal(_panel_frame(path, late_a), _panel_frame(path, late_b)):
            raise RuntimeError("faster provider final-frame hold validation failed")
    return probe


def _panel_frame(path: Path, seconds: float) -> bytes:
    process = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-ss", f"{seconds:.6f}", "-i", str(path),
            "-vf", "crop=520:400:690:174", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    return process.stdout


def _held_frames_equal(left: bytes, right: bytes) -> bool:
    if len(left) != len(right) or not left:
        return False
    lhs = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    rhs = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    difference = np.abs(lhs - rhs)
    # H.264 may reconstruct nominally static macroblocks a few levels apart
    # across seek points. A tight mean/p99 threshold still rejects moving panels.
    return float(np.mean(difference)) <= 1.0 and float(np.quantile(difference, 0.99)) <= 4.0


def _probe(path: Path) -> dict[str, Any]:
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
        ],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    )
    return json.loads(process.stdout)


def _ffmpeg_version() -> str:
    process = subprocess.run(
        ["ffmpeg", "-version"], text=True, stdout=subprocess.PIPE, check=True
    )
    return process.stdout.splitlines()[0]


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)
