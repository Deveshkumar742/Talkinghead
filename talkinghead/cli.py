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
        typer.echo(f"    output     {profile.output_width}x{profile.output_height}")
        typer.echo(f"    face crop  {profile.face_crop}px")
        typer.echo(f"    notes      {profile.notes}")


@app.command()
def tts(
    script: Annotated[Path, typer.Argument(help="Path to the script text file.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output WAV.")] = Path(
        "out/voice.wav"
    ),
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Synthesize narration only (Phase 2). Runs locally on CPU.

    This is the stage you iterate on while writing a script -- it needs no GPU
    session.
    """
    _setup_logging(verbose)
    typer.secho(
        "Phase 2 not implemented yet: the Chatterbox provider "
        "(talkinghead/tts/chatterbox_local.py) has not been built.\n"
        "Phase 1 delivered script prep, ffmpeg assembly, and the CLI. "
        "Run 'talkinghead prep' and 'talkinghead check' in the meantime.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(2)


@app.command()
def lipsync(
    audio: Annotated[Path, typer.Option("--audio", "-a", help="Narration WAV.")],
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("out/synced.mp4"),
    model: Annotated[LipsyncModel, typer.Option("--model")] = LipsyncModel.LATENTSYNC,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run lipsync only (Phase 3). Requires a GPU session.

    Intended to be invoked from the Kaggle notebook, where the narration WAV has
    been uploaded alongside the base loop.
    """
    _setup_logging(verbose)
    typer.secho(
        "Phase 3 not implemented yet: the lipsync providers "
        "(talkinghead/lipsync/latentsync.py, musetalk.py) have not been built.\n"
        "Phase 0 must confirm LatentSync 1.6 @ 512 fits 16GB VRAM first -- "
        "see notebooks/phase0_vram_spike.ipynb.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(2)


@app.command()
def gen(
    script: Annotated[Path, typer.Argument(help="Path to the script text file.")],
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    model: Annotated[LipsyncModel, typer.Option("--model")] = LipsyncModel.LATENTSYNC,
    pingpong: Annotated[
        bool, typer.Option("--pingpong", help="Ping-pong the base loop to hide a seam.")
    ] = False,
    no_cache: Annotated[bool, typer.Option("--no-cache")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the full pipeline: text in, video out.

    Requires both providers, so it needs a GPU session. Blocked until Phases 2
    and 3 land.
    """
    _setup_logging(verbose)
    typer.secho(
        "Full pipeline needs both the TTS provider (Phase 2) and a lipsync "
        "provider (Phase 3); neither is built yet.\n"
        "Phase 1 is complete: try 'talkinghead check', 'talkinghead prep "
        "<script>', or 'talkinghead licenses'.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(2)


if __name__ == "__main__":
    app()
