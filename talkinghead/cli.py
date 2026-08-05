"""Command-line interface.

Stages are exposed individually as well as end-to-end, because the pipeline is
split across two machines: TTS and assembly run on the laptop, lipsync runs in a
Kaggle session. ``gen`` is the convenience path for when both are available.

``check`` exists so a missing asset or absent ffmpeg surfaces in one second on
the laptop rather than twenty minutes into a metered GPU session.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from talkinghead import __version__, media, runtime
from talkinghead.config import (
    PROFILES,
    REJECTED_MODELS,
    TRUSTED_SOURCES,
    LipsyncModel,
    load_settings,
)
from talkinghead.script_prep import prepare_script

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Generate talking-head video from text using your own face and voice.",
)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


def _read_script(path: Path) -> str:
    if not path.exists():
        typer.secho(f"Script not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    return path.read_text(encoding="utf-8")


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"talkinghead {__version__}")


@app.command()
def check(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Verify the environment and assets before committing to a render.

    Checks ffmpeg, both recorded assets, and the base loop's framing against the
    active profile's face-crop size -- the one thing that quietly degrades output
    quality if it is wrong.
    """
    _setup_logging(verbose)
    settings = load_settings()
    profile = settings.active_profile
    host = runtime.detect()
    problems: list[str] = []
    warnings: list[str] = []

    typer.secho(f"talkinghead {__version__}", bold=True)
    typer.echo(f"  host           {host.describe()}")
    typer.echo(f"  profile        {profile.name} "
               f"({profile.output_width}x{profile.output_height}, "
               f"face crop {profile.face_crop}px)")
    typer.echo(f"  device         {settings.device}")
    typer.echo(f"  lipsync        {settings.lipsync_model}")
    typer.echo(f"  work dir       {settings.work_dir}")
    typer.echo("")

    # ffmpeg comes from the host; we never install it.
    for binary in ("ffmpeg", "ffprobe"):
        try:
            resolved = media.find_binary(binary)
            typer.secho(f"  [ok]   {binary:8s} {resolved}", fg=typer.colors.GREEN)
        except media.MediaError:
            problems.append(
                f"{binary} not available on this host. Compute stages are meant "
                f"to run in a Kaggle or Colab session, where it is preinstalled."
            )
            typer.secho(f"  [FAIL] {binary:8s} not found", fg=typer.colors.RED)

    if not host.is_cloud:
        warnings.append(
            "Running locally. This project is cloud-first: open "
            "notebooks/kaggle_render.ipynb in a Kaggle session to run the "
            "pipeline. Local runs are for script prep and the test suite."
        )

    # Voice reference
    if settings.reference_wav.exists():
        info = media.probe(settings.reference_wav)
        typer.secho(
            f"  [ok]   voice    {info.duration_s:.1f}s @ {info.sample_rate}Hz, "
            f"{info.channels}ch",
            fg=typer.colors.GREEN,
        )
        if not 5 <= info.duration_s <= 40:
            warnings.append(
                f"Voice reference is {info.duration_s:.1f}s. Chatterbox clones "
                f"best from 15-20s; very short or very long clips degrade it."
            )
        if info.sample_rate and info.sample_rate < 24_000:
            warnings.append(
                f"Voice reference is {info.sample_rate}Hz. 24kHz or higher is "
                f"recommended."
            )
    else:
        problems.append(f"Missing voice reference: {settings.reference_wav}")
        typer.secho("  [FAIL] voice    not recorded", fg=typer.colors.RED)

    # Base loop
    if settings.base_loop.exists():
        info = media.probe(settings.base_loop)
        typer.secho(
            f"  [ok]   base     {info.duration_s:.1f}s, "
            f"{info.width}x{info.height} @ {info.fps:.0f}fps",
            fg=typer.colors.GREEN,
        )
        if info.duration_s < 45:
            warnings.append(
                f"Base loop is {info.duration_s:.1f}s. 60-90s is recommended so "
                f"the loop does not visibly repeat over a multi-minute script."
            )
        if info.height and info.height < profile.output_height:
            problems.append(
                f"Base loop is {info.height}p but the {profile.name} profile "
                f"outputs {profile.output_height}p. Upscaling will look soft -- "
                f"re-record at {profile.output_height}p or higher, or switch "
                f"profile with TH_PROFILE."
            )
        if info.has_audio:
            warnings.append(
                "Base loop has an audio track. It is ignored, but its presence "
                "suggests you may have been speaking -- the loop must be silent "
                "and mouth-closed or the lipsync will double-articulate."
            )
    else:
        problems.append(f"Missing base loop: {settings.base_loop}")
        typer.secho("  [FAIL] base     not recorded", fg=typer.colors.RED)

    # When an asset is missing but the mount is present, the cause is almost
    # always a filename or nesting mismatch rather than a failed upload. Showing
    # the actual contents turns "not recorded" into something actionable.
    assets_missing = not (
        settings.reference_wav.exists() and settings.base_loop.exists()
    )
    if assets_missing and settings.assets_dir.exists():
        found = sorted(
            p for p in settings.assets_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".wav", ".mp3", ".m4a"}
        )
        typer.echo(f"\n  Media actually present in {settings.assets_dir}:")
        if found:
            for p in found[:20]:
                size_mb = p.stat().st_size / 1024**2
                typer.echo(f"    {size_mb:8.1f} MB  {p.relative_to(settings.assets_dir)}")
            typer.secho(
                "\n  The filenames must be exactly 'base_loop.mp4' and "
                "'reference.wav'. Rename them in the dataset and re-upload a new "
                "version, or point at them directly:\n"
                "    TH_BASE_LOOP=<path> TH_REFERENCE_WAV=<path> talkinghead check",
                fg=typer.colors.YELLOW,
            )
        else:
            typer.echo("    (no media files found at all)")

    typer.echo("")
    for warning in warnings:
        typer.secho(f"  warning: {warning}", fg=typer.colors.YELLOW)
    for problem in problems:
        typer.secho(f"  problem: {problem}", fg=typer.colors.RED)

    if problems:
        typer.secho(f"\n{len(problems)} problem(s) must be fixed.", fg=typer.colors.RED)
        raise typer.Exit(1)
    typer.secho("\nEnvironment looks good.", fg=typer.colors.GREEN)


@app.command()
def host() -> None:
    """Report the detected runtime and what it provides.

    Useful inside a notebook to confirm the session actually has a GPU attached
    and that the private asset dataset mounted where expected.
    """
    info = runtime.detect()
    settings = load_settings()

    typer.secho(f"host: {info.host}", bold=True)
    typer.echo(f"  cloud session  {info.is_cloud}")
    typer.echo(f"  ffmpeg         {'present' if info.has_ffmpeg else 'ABSENT'}")
    typer.echo(f"  gpu            {'present' if info.has_gpu else 'absent'}")
    typer.echo(f"  input mount    {info.input_root or '(none)'}")
    typer.echo(f"  work root      {info.work_root}")
    typer.echo("")
    typer.echo(f"  assets_dir     {settings.assets_dir}")
    typer.echo(f"  base_loop      {settings.base_loop}")
    typer.echo(f"  reference_wav  {settings.reference_wav}")
    typer.echo(f"  weights_dir    {settings.weights_dir}")

    if info.host is runtime.Host.KAGGLE and not settings.assets_dir.exists():
        typer.secho(
            f"\n  Expected dataset '{settings.kaggle_dataset}' is not mounted.",
            fg=typer.colors.YELLOW,
        )
        mounts = info.available_mounts()
        if mounts:
            # The usual cause is a slug mismatch rather than a missing dataset,
            # so show what is really there and how to point at it.
            typer.echo("  Currently mounted under /kaggle/input:")
            for name in mounts:
                typer.echo(f"    - {name}")
            typer.echo(
                f"\n  If one of those is yours, point at it with:\n"
                f"    TH_KAGGLE_DATASET=<name> talkinghead check\n"
                f"  Kaggle slugifies dataset titles, so the folder name is "
                f"often not what you typed."
            )
        else:
            typer.echo(
                "  Nothing is mounted. Attach your private dataset via "
                "+ Add Input in the notebook sidebar."
            )
    elif not info.is_cloud:
        typer.secho(
            "\n  Running locally. Compute stages belong in a cloud session -- "
            "see notebooks/kaggle_render.ipynb.",
            fg=typer.colors.YELLOW,
        )


@app.command()
def licenses() -> None:
    """Show the model licensing position.

    Worth reading once: the popular defaults in this space are mostly
    non-commercial, and this project is for client-facing work.
    """
    typer.secho("Approved sources (weights verified commercial-safe)", bold=True)
    for key, src in TRUSTED_SOURCES.items():
        typer.echo(f"  {key:12s} {src.license:12s} {src.repo}")
        if src.hf_repo:
            typer.echo(f"  {'':12s} {'':12s} hf.co/{src.hf_repo} @ {src.revision}")

    typer.echo("")
    typer.secho("Rejected — do NOT reintroduce", bold=True, fg=typer.colors.RED)
    for key, reason in REJECTED_MODELS.items():
        typer.echo(f"  {key:12s} {reason}")


@app.command("prep")
def prep(
    script: Annotated[Path, typer.Argument(help="Path to the script text file.")],
    show_text: Annotated[bool, typer.Option("--show-text")] = False,
) -> None:
    """Split a script into segments and report the plan, without synthesizing.

    Use this to sanity-check segmentation and estimated runtime before spending
    CPU time on TTS.
    """
    settings = load_settings()
    prepared = prepare_script(
        _read_script(script),
        max_chars=settings.max_chars_per_segment,
        sentence_pause_ms=settings.sentence_pause_ms,
        paragraph_pause_ms=settings.paragraph_pause_ms,
    )

    typer.secho(
        f"{len(prepared)} segments, {prepared.total_chars} chars, "
        f"~{prepared.estimated_duration_s():.0f}s estimated",
        bold=True,
    )
    typer.echo(f"script hash: {prepared.script_hash}")
    typer.echo("")
    for seg in prepared.segments:
        marker = "¶" if seg.is_paragraph_end else " "
        preview = seg.text if show_text else seg.text[:70] + (
            "..." if len(seg.text) > 70 else ""
        )
        typer.echo(
            f"  {seg.slug} {marker} [{len(seg.text):3d}ch "
            f"+{seg.pause_after_ms:3d}ms] {preview}"
        )


@app.command()
def profiles() -> None:
    """List resolution profiles."""
    for name, profile in PROFILES.items():
        typer.secho(f"  {name}", bold=True)
        typer.echo(f"    output      {profile.output_width}x{profile.output_height}")
        typer.echo(f"    face crop   {profile.face_crop}px")
        typer.echo(f"    checkpoint  hf.co/{profile.hf_repo}")
        typer.echo(f"    num_frames  {profile.num_frames}")
        typer.echo(f"    notes       {profile.notes}")
        typer.echo("")


def _build_tts(settings) -> "object":
    """Construct the TTS provider.

    Imported here rather than at module scope so the CLI still starts on a
    machine with no torch installed -- `check`, `prep` and `host` must keep
    working locally.
    """
    from talkinghead.tts.chatterbox_local import ChatterboxTTS

    return ChatterboxTTS(device=settings.device)


def _build_lipsync(settings, repo_dir: Path, cache_dir: Path | None, steps: int,
                   num_frames: int | None) -> "object":
    """Construct the lipsync provider for the active profile."""
    if settings.lipsync_model is LipsyncModel.MUSETALK:
        raise typer.BadParameter(
            "The MuseTalk provider is not built yet. It is the planned "
            "alternative for the quality bake-off; use --model latentsync."
        )
    from talkinghead.lipsync.latentsync import LatentSync

    return LatentSync(
        profile=settings.active_profile,
        repo_dir=repo_dir,
        cache_dir=cache_dir,
        device=settings.device,
        inference_steps=steps,
        num_frames=num_frames,
    )


@app.command()
def tts(
    script: Annotated[Path, typer.Argument(help="Path to the script text file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output WAV.")] = Path(
        "out/voice.wav"
    ),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Synthesize narration only — text to a WAV in your cloned voice.

    Needs no GPU: Chatterbox runs on CPU, slower than realtime but fine for
    iterating on wording. This is the cheap half of the pipeline, so get the
    voice right here before spending GPU quota on video.
    """
    _setup_logging(verbose)
    settings = load_settings()

    from talkinghead.pipeline import stage_assemble_audio, stage_tts, Cache

    prepared = prepare_script(
        _read_script(script),
        max_chars=settings.max_chars_per_segment,
        sentence_pause_ms=settings.sentence_pause_ms,
        paragraph_pause_ms=settings.paragraph_pause_ms,
    )
    typer.echo(
        f"{len(prepared)} segments, ~{prepared.estimated_duration_s():.0f}s estimated"
    )

    provider = _build_tts(settings)
    cache = Cache(settings.work_dir / prepared.script_hash, enabled=settings.use_cache)

    segment_wavs = stage_tts(prepared, provider, settings.reference_wav, cache)
    voice = stage_assemble_audio(
        prepared, segment_wavs, out, cache,
        sample_rate=provider.sample_rate, target_lufs=settings.loudness_lufs,
    )

    info = media.probe(voice)
    typer.secho(
        f"\nwrote {voice}  ({info.duration_s:.1f}s @ {info.sample_rate}Hz)",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"segments cached in {cache.root / 'segments'}")


@app.command()
def lipsync(
    audio: Annotated[Path, typer.Option("--audio", "-a", help="Narration WAV.")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("out/synced.mp4"),
    repo_dir: Annotated[
        Path, typer.Option("--repo-dir", help="Where to clone LatentSync.")
    ] = Path("/kaggle/working/LatentSync"),
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Pre-downloaded weights; skips a 9.8GB fetch."),
    ] = None,
    steps: Annotated[
        int, typer.Option("--steps", help="Denoising steps. Lower is faster.")
    ] = 20,
    num_frames: Annotated[
        int | None,
        typer.Option("--num-frames", help="Override the profile's value."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run lipsync only — an existing WAV onto the base loop. Needs a GPU."""
    _setup_logging(verbose)
    settings = load_settings()

    from talkinghead.pipeline import (
        Cache,
        stage_lipsync,
        stage_prepare_base_video,
    )

    narration = media.probe(audio)
    typer.echo(f"narration is {narration.duration_s:.1f}s")

    cache = Cache(settings.work_dir / "lipsync", enabled=settings.use_cache)
    base = stage_prepare_base_video(
        settings.base_loop, narration.duration_s, cache.root / "base.mp4",
        cache, fps=settings.fps,
    )
    provider = _build_lipsync(settings, repo_dir, cache_dir, steps, num_frames)
    result = stage_lipsync(base, Path(audio), provider, out, cache)

    info = media.probe(result)
    typer.secho(
        f"\nwrote {result}  ({info.width}x{info.height}, {info.duration_s:.1f}s)",
        fg=typer.colors.GREEN,
    )


@app.command()
def gen(
    script: Annotated[Path, typer.Argument(help="Path to the script text file.")],
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    model: Annotated[LipsyncModel, typer.Option("--model")] = LipsyncModel.LATENTSYNC,
    repo_dir: Annotated[
        Path, typer.Option("--repo-dir")
    ] = Path("/kaggle/working/LatentSync"),
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Pre-downloaded weights; skips a 9.8GB fetch."),
    ] = None,
    steps: Annotated[
        int, typer.Option("--steps", help="Denoising steps. Lower is faster.")
    ] = 20,
    num_frames: Annotated[
        int | None, typer.Option("--num-frames", help="Override the profile's value.")
    ] = None,
    pingpong: Annotated[
        bool, typer.Option("--pingpong", help="Ping-pong the base loop to hide a seam.")
    ] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full pipeline: text in, video out. Needs a GPU session."""
    _setup_logging(verbose)
    settings = load_settings(lipsync_model=model, use_cache=not no_cache)

    from talkinghead.pipeline import generate

    profile = settings.active_profile
    typer.echo(
        f"profile {profile.name} — {profile.output_width}x{profile.output_height}, "
        f"{profile.face_crop}px crop, num_frames="
        f"{num_frames if num_frames is not None else profile.num_frames}"
    )

    artifacts = generate(
        script_text=_read_script(script),
        settings=settings,
        tts_provider=_build_tts(settings),
        lipsync_provider=_build_lipsync(
            settings, repo_dir, cache_dir, steps, num_frames
        ),
        out_path=out,
        pingpong=pingpong,
    )

    info = media.probe(artifacts.output)
    typer.secho(f"\nwrote {artifacts.output}", fg=typer.colors.GREEN)
    typer.echo(f"  {info.width}x{info.height}, {info.duration_s:.1f}s")
    typer.echo(f"  {artifacts.describe_reuse()}")


if __name__ == "__main__":
    app()
