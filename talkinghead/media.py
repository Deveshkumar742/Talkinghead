"""ffmpeg and ffprobe wrappers.

These are the CPU stages: assemble audio, fit the base loop to the narration
length, encode the final file. No GPU, no ML dependencies.

**FFmpeg is a host-provided dependency.** This project never installs it. The
intended runtime is a Kaggle or Colab notebook session, both of which ship with
FFmpeg already on PATH -- and on Kaggle, a CPU-only session is unmetered, so
running these stages there costs nothing against the GPU quota.

A local install is optional and exists only to run the test suite offline; see
``runtime.FFMPEG_MISSING_HELP``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from talkinghead.runtime import FFMPEG_MISSING_HELP, detect


class MediaError(RuntimeError):
    """An ffmpeg/ffprobe invocation failed, or its output was unusable."""


@lru_cache(maxsize=8)
def find_binary(name: str) -> str:
    """Resolve an ffmpeg-family binary from the host environment.

    Cached, because this is called on every ffmpeg invocation and the answer
    cannot change within a session.

    Raises:
        MediaError: if the binary is absent, with guidance that leads to a cloud
            session rather than a local install.
    """
    found = shutil.which(name)
    if found:
        return found

    host = detect()
    raise MediaError(
        f"Could not find {name!r} on PATH (host: {host.host}).\n\n"
        f"{FFMPEG_MISSING_HELP}"
    )


def ffmpeg_available() -> bool:
    """True if both binaries are present. Never raises."""
    try:
        find_binary("ffmpeg")
        find_binary("ffprobe")
        return True
    except MediaError:
        return False


def _run(args: list[str], *, what: str) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, raising a useful error on failure.

    ffmpeg writes progress to stderr even on success, so stderr is only surfaced
    when the exit code is non-zero -- and then trimmed, because ffmpeg's banner
    buries the actual error under a wall of build flags.
    """
    proc = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"{what} failed (exit {proc.returncode}):\n{tail}")
    return proc


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaInfo:
    """What we need to know about a media file to make decisions about it."""

    path: Path
    duration_s: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None
    channels: int | None = None


def probe(path: Path | str) -> MediaInfo:
    """Read stream metadata via ffprobe.

    Raises:
        MediaError: if the file is missing or ffprobe cannot parse it.
    """
    path = Path(path)
    if not path.exists():
        raise MediaError(f"File not found: {path}")

    proc = _run(
        [
            find_binary("ffprobe"),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        what=f"ffprobe {path.name}",
    )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned unparseable JSON for {path.name}") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # Prefer container duration; fall back to the stream's own if absent.
    duration = data.get("format", {}).get("duration")
    if duration is None:
        duration = (video or audio or {}).get("duration")
    try:
        duration_s = float(duration)
    except (TypeError, ValueError):
        raise MediaError(f"Could not determine duration of {path.name}") from None

    fps = None
    if video:
        # avg_frame_rate arrives as a rational string like "30000/1001".
        raw = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0"
        try:
            num, _, den = raw.partition("/")
            fps = float(num) / float(den) if float(den) else None
        except (ValueError, ZeroDivisionError):
            fps = None

    return MediaInfo(
        path=path,
        duration_s=duration_s,
        has_video=video is not None,
        has_audio=audio is not None,
        width=int(video["width"]) if video and "width" in video else None,
        height=int(video["height"]) if video and "height" in video else None,
        fps=fps,
        sample_rate=int(audio["sample_rate"]) if audio and "sample_rate" in audio else None,
        channels=int(audio["channels"]) if audio and "channels" in audio else None,
    )


# --------------------------------------------------------------------------
# Audio assembly
# --------------------------------------------------------------------------

def make_silence(duration_s: float, out_path: Path, sample_rate: int = 24_000) -> Path:
    """Generate a mono silent WAV. Used for padding and for test fixtures."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_binary("ffmpeg"), "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=mono:sample_rate={sample_rate}",
            "-t", f"{duration_s:.4f}",
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        what="generate silence",
    )
    return out_path


def make_tone(
    duration_s: float, out_path: Path, freq: int = 440, sample_rate: int = 24_000
) -> Path:
    """Generate a mono sine tone. Test fixture helper, not used in the pipeline."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_binary("ffmpeg"), "-y",
            "-f", "lavfi",
            "-i", f"sine=frequency={freq}:sample_rate={sample_rate}",
            "-t", f"{duration_s:.4f}",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        what="generate tone",
    )
    return out_path


def concat_audio(
    clips: list[tuple[Path, int]],
    out_path: Path,
    sample_rate: int = 24_000,
) -> Path:
    """Concatenate audio clips, appending silence after each.

    Args:
        clips: ``(path, pause_after_ms)`` pairs, in playback order.
        out_path: destination WAV.
        sample_rate: everything is resampled to this before concat, which also
            protects against clips that disagree on rate.

    Done as a single ``filter_complex`` call rather than by generating silence
    files and using the concat demuxer -- one process, no temp files, and no
    chance of a sample-rate mismatch silently resampling mid-stream.

    Raises:
        MediaError: if ``clips`` is empty or any clip is missing.
    """
    if not clips:
        raise MediaError("concat_audio called with no clips.")
    for path, _ in clips:
        if not Path(path).exists():
            raise MediaError(f"Audio clip not found: {path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [find_binary("ffmpeg"), "-y"]
    for path, _ in clips:
        args += ["-i", str(path)]

    filters = []
    labels = []
    for i, (_, pause_ms) in enumerate(clips):
        label = f"a{i}"
        chain = f"[{i}:a]aresample={sample_rate},aformat=channel_layouts=mono"
        if pause_ms > 0:
            chain += f",apad=pad_dur={pause_ms / 1000.0:.4f}"
        filters.append(f"{chain}[{label}]")
        labels.append(f"[{label}]")

    filters.append(f"{''.join(labels)}concat=n={len(clips)}:v=0:a=1[out]")

    args += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-c:a", "pcm_s16le",
        str(out_path),
    ]
    _run(args, what="concat audio")
    return out_path


def normalize_loudness(
    in_path: Path, out_path: Path, target_lufs: float = -16.0
) -> Path:
    """Normalize to a broadcast-ish loudness target.

    Single-pass ``loudnorm``. Two-pass measures more accurately, but for
    speech-only material assembled from one voice the difference is not audible,
    and one pass keeps the pipeline simple. -16 LUFS is the usual target for
    spoken web video.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_binary("ffmpeg"), "-y",
            "-i", str(in_path),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        what="normalize loudness",
    )
    return out_path


# --------------------------------------------------------------------------
# Video preparation
# --------------------------------------------------------------------------

def make_pingpong(in_path: Path, out_path: Path) -> Path:
    """Append a reversed copy of the clip to itself.

    This is the fallback when a base loop does not start and end in the same
    pose: a straight loop would jump-cut, whereas ping-pong is seamless at the
    cost of one visibly reversed stretch of motion. Silent -- audio is added
    later.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_binary("ffmpeg"), "-y",
            "-i", str(in_path),
            "-filter_complex",
            "[0:v]split[fwd][tmp];[tmp]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0[out]",
            "-map", "[out]",
            "-an",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        what="build ping-pong loop",
    )
    return out_path


def fit_video_to_duration(
    in_path: Path,
    target_duration_s: float,
    out_path: Path,
    fps: int = 25,
    pingpong: bool = False,
) -> Path:
    """Produce a silent video of exactly ``target_duration_s``.

    Trims if the source is long enough, otherwise loops it. The result feeds the
    lipsync stage, so it is kept at source resolution and re-encoded losslessly
    enough that the mouth region is not degraded before the model ever sees it.

    Raises:
        MediaError: if the target duration is not positive.
    """
    if target_duration_s <= 0:
        raise MediaError(f"Target duration must be positive, got {target_duration_s}.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(in_path)

    # Ping-pong first, so the looping below operates on the seamless version.
    if pingpong:
        pp = out_path.with_name(f"{out_path.stem}_pingpong.mp4")
        source = make_pingpong(source, pp)

    info = probe(source)
    if not info.has_video:
        raise MediaError(f"{source.name} has no video stream.")

    args = [find_binary("ffmpeg"), "-y"]
    # -stream_loop must precede -i. Looping a source that is already long
    # enough is harmless, but skipping it avoids a needless re-read.
    if info.duration_s < target_duration_s:
        args += ["-stream_loop", "-1"]
    args += [
        "-i", str(source),
        "-t", f"{target_duration_s:.4f}",
        "-r", str(fps),
        "-an",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    _run(args, what="fit base video to narration length")
    return out_path


# --------------------------------------------------------------------------
# Final output
# --------------------------------------------------------------------------

def mux_and_encode(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: int = 25,
    crf: int = 18,
    audio_bitrate: str = "192k",
    metadata: dict[str, str] | None = None,
) -> Path:
    """Combine video and audio into the deliverable MP4.

    Scaling happens here rather than earlier so the lipsync model always sees
    full-resolution frames. For the 720p profile this is the step where the
    downscale pulls the whole frame toward the regenerated mouth's true
    resolution, which is what hides the softness.

    ``-shortest`` guards against a fractional mismatch between the fitted video
    and the narration leaving a frozen tail frame.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        find_binary("ffmpeg"), "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-vf", (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        ),
        "-r", str(fps),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        "-shortest",
    ]
    for key, value in (metadata or {}).items():
        args += ["-metadata", f"{key}={value}"]
    args.append(str(out_path))

    _run(args, what="mux and encode final video")
    return out_path


def make_test_video(
    duration_s: float,
    out_path: Path,
    width: int = 320,
    height: int = 240,
    fps: int = 25,
) -> Path:
    """Generate a synthetic test clip. Fixture helper for the test suite."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            find_binary("ffmpeg"), "-y",
            "-f", "lavfi",
            "-i", f"testsrc=size={width}x{height}:rate={fps}",
            "-t", f"{duration_s:.4f}",
            "-an",
            "-c:v", "libx264", "-crf", "30", "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        what="generate test video",
    )
    return out_path
