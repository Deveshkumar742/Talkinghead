"""Pipeline orchestration with a resumable artifact cache.

Six stages, of which only lipsync needs a GPU:

    1. script_prep   text -> segments                      local
    2. tts           segments -> per-segment WAVs           local CPU
    3. media         WAVs + pauses -> voice.wav             local
    4. media         base loop -> base.mp4 at right length  local
    5. lipsync       base.mp4 + voice.wav -> synced.mp4     GPU (Kaggle)
    6. media         mux, provenance, encode -> out.mp4     local

Providers are injected rather than constructed here, which keeps this module
importable and testable on a laptop with no torch installed -- the whole reason
Phase 1 can be verified before any model exists.

The cache is what makes a Kaggle session timeout survivable: every stage writes
a named artifact under ``work/<script_hash>/``, and a rerun skips any stage whose
output is already present. Segment WAVs are additionally keyed by a hash of their
own text, so editing one sentence re-synthesizes one segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from talkinghead import media, provenance
from talkinghead.config import Settings
from talkinghead.lipsync.base import LipsyncProvider
from talkinghead.script_prep import PreparedScript, Segment, prepare_script
from talkinghead.tts.base import TTSProvider

log = logging.getLogger("talkinghead")


class PipelineError(RuntimeError):
    """A stage could not complete."""


@dataclass
class Artifacts:
    """Paths to everything a run produces, plus which stages were reused."""

    root: Path
    voice_wav: Path
    base_video: Path
    synced_video: Path
    output: Path
    segment_wavs: list[Path] = field(default_factory=list)
    reused: set[str] = field(default_factory=set)

    def describe_reuse(self) -> str:
        if not self.reused:
            return "no cached stages reused"
        return f"reused from cache: {', '.join(sorted(self.reused))}"


class Cache:
    """Stage-level artifact cache rooted at ``work/<script_hash>/``."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "segments").mkdir(exist_ok=True)
        self.reused: set[str] = set()

    def is_fresh(self, path: Path, stage: str) -> bool:
        """True if ``path`` is a usable cached artifact.

        A zero-byte file is treated as absent -- that is what a killed ffmpeg
        leaves behind, and silently reusing it would corrupt the run.
        """
        if not self.enabled:
            return False
        if not path.exists() or path.stat().st_size == 0:
            return False
        self.reused.add(stage)
        log.info("cache hit: %s (%s)", stage, path.name)
        return True

    def segment_path(self, segment: Segment) -> Path:
        """Per-segment WAV path, keyed by the segment's own text hash."""
        return self.root / "segments" / f"{segment.slug}_{segment.cache_key()}.wav"


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_tts(
    prepared: PreparedScript,
    provider: TTSProvider,
    reference_wav: Path,
    cache: Cache,
) -> list[Path]:
    """Synthesize each segment against the same reference clip.

    Warms the provider only if there is at least one segment to generate, so a
    fully cached rerun does not pay a multi-gigabyte model load.
    """
    if not reference_wav.exists():
        raise PipelineError(
            f"Voice reference not found: {reference_wav}\n"
            f"Record a 15-20s clean speech sample -- see assets/README.md."
        )

    paths = [cache.segment_path(s) for s in prepared.segments]
    todo = [
        (seg, path)
        for seg, path in zip(prepared.segments, paths, strict=True)
        if not cache.is_fresh(path, "tts")
    ]

    if todo:
        log.info("synthesizing %d/%d segments", len(todo), len(paths))
        provider.warmup()
        for seg, path in todo:
            log.info("  [%s] %s", seg.slug, seg.text[:60])
            provider.synthesize(seg.text, reference_wav, path)
            if not path.exists() or path.stat().st_size == 0:
                raise PipelineError(
                    f"TTS produced an empty file for {seg.slug}: {seg.text[:80]!r}"
                )
    return paths


def stage_assemble_audio(
    prepared: PreparedScript,
    segment_wavs: list[Path],
    out_path: Path,
    cache: Cache,
    sample_rate: int,
    target_lufs: float,
) -> Path:
    """Concatenate segments with their pauses, then normalize loudness."""
    if cache.is_fresh(out_path, "assemble_audio"):
        return out_path

    clips = [
        (path, seg.pause_after_ms)
        for seg, path in zip(prepared.segments, segment_wavs, strict=True)
    ]
    raw = out_path.with_name(f"{out_path.stem}_raw.wav")
    media.concat_audio(clips, raw, sample_rate=sample_rate)
    media.normalize_loudness(raw, out_path, target_lufs=target_lufs)
    return out_path


def stage_prepare_base_video(
    base_loop: Path,
    duration_s: float,
    out_path: Path,
    cache: Cache,
    fps: int,
    pingpong: bool = False,
) -> Path:
    """Fit the recorded base loop to the narration length."""
    if cache.is_fresh(out_path, "base_video"):
        return out_path

    if not base_loop.exists():
        raise PipelineError(
            f"Base loop video not found: {base_loop}\n"
            f"Record a 60-90s silent mid-shot loop -- see assets/README.md."
        )

    info = media.probe(base_loop)
    if not info.has_video:
        raise PipelineError(f"{base_loop.name} contains no video stream.")

    # Loud warning rather than a hard failure: a short loop still works, it just
    # repeats more visibly, and that is the user's call to make.
    if info.duration_s < 20:
        log.warning(
            "Base loop is only %.1fs. Short loops repeat visibly over a "
            "multi-minute narration -- 60-90s is recommended.",
            info.duration_s,
        )
    return media.fit_video_to_duration(
        base_loop, duration_s, out_path, fps=fps, pingpong=pingpong
    )


def stage_lipsync(
    base_video: Path,
    voice_wav: Path,
    provider: LipsyncProvider,
    out_path: Path,
    cache: Cache,
) -> Path:
    """Repaint the mouth to match the narration.

    Verifies the provider preserves resolution before running, because a
    full-frame generative model silently downscaling to 720p would otherwise
    only become apparent in the finished file.
    """
    if cache.is_fresh(out_path, "lipsync"):
        return out_path

    if not provider.preserves_resolution:
        raise PipelineError(
            f"Lipsync provider {provider.model_key!r} does not preserve input "
            f"resolution, so it cannot be used to render at source resolution. "
            f"Use a mouth-inpainting provider (latentsync, musetalk)."
        )

    before = media.probe(base_video)
    result = provider.sync(base_video, voice_wav, out_path)

    after = media.probe(result)
    if (after.width, after.height) != (before.width, before.height):
        raise PipelineError(
            f"Lipsync changed resolution from {before.width}x{before.height} to "
            f"{after.width}x{after.height}. Expected the mouth region to be "
            f"repainted in place."
        )
    return result


def stage_finalize(
    synced_video: Path,
    voice_wav: Path,
    out_path: Path,
    settings: Settings,
    script_hash: str,
    created: str | None = None,
) -> Path:
    """Mux, tag provenance, and encode the deliverable."""
    profile = settings.active_profile
    metadata = provenance.build_metadata(
        script_hash=script_hash,
        lipsync_model=str(settings.lipsync_model),
        profile=profile.name,
        created=created,
    )
    return media.mux_and_encode(
        video_path=synced_video,
        audio_path=voice_wav,
        out_path=out_path,
        width=profile.output_width,
        height=profile.output_height,
        fps=settings.fps,
        crf=settings.video_crf,
        audio_bitrate=settings.audio_bitrate,
        metadata=metadata,
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate(
    script_text: str,
    settings: Settings,
    tts_provider: TTSProvider,
    lipsync_provider: LipsyncProvider,
    out_path: Path | None = None,
    pingpong: bool = False,
    created: str | None = None,
) -> Artifacts:
    """Run the full text-to-video pipeline.

    Args:
        script_text: the final script. Not rewritten or polished -- what you
            pass is what gets spoken.
        settings: resolution profile, pause timings, encoding options.
        tts_provider: injected so this is testable without torch installed.
        lipsync_provider: likewise.
        out_path: destination MP4. Defaults to ``out/<script_hash>.mp4``.
        pingpong: append a reversed copy of the base loop before looping, for
            footage that does not start and end in the same pose.
        created: ISO timestamp for provenance metadata.

    Raises:
        PipelineError: if any stage fails or an asset is missing.
    """
    prepared = prepare_script(
        script_text,
        max_chars=settings.max_chars_per_segment,
        sentence_pause_ms=settings.sentence_pause_ms,
        paragraph_pause_ms=settings.paragraph_pause_ms,
    )
    log.info(
        "prepared %d segments (~%.0fs estimated)",
        len(prepared),
        prepared.estimated_duration_s(),
    )

    cache = Cache(settings.work_dir / prepared.script_hash, enabled=settings.use_cache)
    out_path = out_path or (settings.out_dir / f"{prepared.script_hash}.mp4")

    segment_wavs = stage_tts(
        prepared, tts_provider, settings.reference_wav, cache
    )

    voice_wav = stage_assemble_audio(
        prepared,
        segment_wavs,
        cache.root / "voice.wav",
        cache,
        sample_rate=tts_provider.sample_rate,
        target_lufs=settings.loudness_lufs,
    )

    # Video length follows the *actual* narration, not the estimate.
    narration = media.probe(voice_wav)
    log.info("narration is %.1fs", narration.duration_s)

    base_video = stage_prepare_base_video(
        settings.base_loop,
        narration.duration_s,
        cache.root / "base.mp4",
        cache,
        fps=settings.fps,
        pingpong=pingpong,
    )

    synced = stage_lipsync(
        base_video, voice_wav, lipsync_provider, cache.root / "synced.mp4", cache
    )

    final = stage_finalize(
        synced, voice_wav, out_path, settings, prepared.script_hash, created
    )

    return Artifacts(
        root=cache.root,
        voice_wav=voice_wav,
        base_video=base_video,
        synced_video=synced,
        output=final,
        segment_wavs=segment_wavs,
        reused=cache.reused,
    )
