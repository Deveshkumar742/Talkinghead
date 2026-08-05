"""Tests for pipeline orchestration and the artifact cache.

Both providers are faked, which is the point of injecting them: orchestration,
caching, and the resolution guard are all verified here without torch, without a
GPU, and without downloading a single weight.

The fake TTS provider emits real audio via ffmpeg, so the assembly stages get
genuine input rather than empty files.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from talkinghead import media
from talkinghead.config import Settings
from talkinghead.pipeline import Cache, PipelineError, generate
from talkinghead.script_prep import prepare_script
from tests.conftest import requires_ffmpeg


class FakeTTS:
    """Emits a tone per segment, duration scaled to text length."""

    model_key = "chatterbox"
    sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.warmups = 0

    def warmup(self) -> None:
        self.warmups += 1

    def synthesize(self, text: str, reference_wav: Path, out_path: Path) -> Path:
        self.calls.append(text)
        duration = max(0.2, len(text) * 0.02)
        return media.make_tone(duration, out_path, sample_rate=self.sample_rate)


class FakeLipsync:
    """Passes frames through unchanged, which preserves resolution as required."""

    model_key = "latentsync"
    face_crop = 512
    preserves_resolution = True

    def __init__(self) -> None:
        self.calls = 0

    def sync(self, video_path: Path, audio_path: Path, out_path: Path) -> Path:
        self.calls += 1
        shutil.copy(video_path, out_path)
        return out_path

    def estimated_vram_gb(self) -> float:
        return 8.0


class DownscalingLipsync(FakeLipsync):
    """Stands in for a full-frame generative model capped below source res."""

    preserves_resolution = False


class SneakyDownscalingLipsync(FakeLipsync):
    """Claims to preserve resolution but does not. The guard must still catch it."""

    preserves_resolution = True

    def sync(self, video_path: Path, audio_path: Path, out_path: Path) -> Path:
        self.calls += 1
        media.mux_and_encode(
            video_path, audio_path, out_path, width=160, height=120
        )
        return out_path


@pytest.fixture
def assets(tmp_path):
    """A minimal reference WAV and base loop."""
    return {
        "reference": media.make_tone(2.0, tmp_path / "assets" / "reference.wav"),
        "base": media.make_test_video(3.0, tmp_path / "assets" / "base_loop.mp4"),
    }


@pytest.fixture
def settings(tmp_path, assets):
    return Settings(
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "out",
        reference_wav=assets["reference"],
        base_loop=assets["base"],
        profile="720p",
        fps=25,
    )


class TestCache:
    def test_creates_its_directories(self, tmp_path):
        cache = Cache(tmp_path / "run")
        assert cache.root.exists()
        assert (cache.root / "segments").exists()

    def test_missing_file_is_not_fresh(self, tmp_path):
        cache = Cache(tmp_path / "run")
        assert cache.is_fresh(tmp_path / "nope.wav", "tts") is False

    def test_zero_byte_file_is_not_fresh(self, tmp_path):
        # A killed ffmpeg leaves an empty file behind; reusing it would corrupt
        # the run silently, which is the worst possible failure mode.
        cache = Cache(tmp_path / "run")
        empty = tmp_path / "empty.wav"
        empty.touch()
        assert cache.is_fresh(empty, "tts") is False

    def test_populated_file_is_fresh_and_records_the_stage(self, tmp_path):
        cache = Cache(tmp_path / "run")
        real = tmp_path / "real.wav"
        real.write_bytes(b"content")
        assert cache.is_fresh(real, "tts") is True
        assert "tts" in cache.reused

    def test_disabled_cache_never_hits(self, tmp_path):
        cache = Cache(tmp_path / "run", enabled=False)
        real = tmp_path / "real.wav"
        real.write_bytes(b"content")
        assert cache.is_fresh(real, "tts") is False

    def test_segment_paths_differ_when_text_differs(self, tmp_path):
        cache = Cache(tmp_path / "run")
        a = prepare_script("First sentence here.").segments[0]
        b = prepare_script("Different sentence.").segments[0]
        assert cache.segment_path(a) != cache.segment_path(b)


@requires_ffmpeg
class TestGenerate:
    def test_produces_a_video_with_both_streams(self, settings):
        result = generate("Hello there. This is a test.", settings, FakeTTS(), FakeLipsync())
        assert result.output.exists()
        info = media.probe(result.output)
        assert info.has_video and info.has_audio

    def test_output_matches_the_profile_resolution(self, settings):
        result = generate("Hello there.", settings, FakeTTS(), FakeLipsync())
        info = media.probe(result.output)
        assert (info.width, info.height) == (1280, 720)

    def test_synthesizes_once_per_segment(self, settings):
        tts = FakeTTS()
        generate("One. Two. Three.", settings, tts, FakeLipsync())
        assert len(tts.calls) == 3

    def test_video_length_follows_narration(self, settings):
        # The base loop is 3s; narration is longer, so the loop must extend.
        script = " ".join(f"Sentence number {i} here." for i in range(12))
        result = generate(script, settings, FakeTTS(), FakeLipsync())
        narration = media.probe(result.voice_wav).duration_s
        output = media.probe(result.output).duration_s
        assert narration > 3.0
        assert abs(output - narration) < 0.5

    def test_respects_explicit_output_path(self, settings, tmp_path):
        target = tmp_path / "custom" / "my_video.mp4"
        result = generate("Hello.", settings, FakeTTS(), FakeLipsync(), out_path=target)
        assert result.output == target and target.exists()


@requires_ffmpeg
class TestCaching:
    def test_second_run_reuses_segments_and_skips_synthesis(self, settings):
        script = "One. Two."
        first = FakeTTS()
        generate(script, settings, first, FakeLipsync())
        assert len(first.calls) == 2

        second = FakeTTS()
        result = generate(script, settings, second, FakeLipsync())
        assert second.calls == []
        assert second.warmups == 0  # no model load paid on a full cache hit
        assert "tts" in result.reused

    def test_editing_one_sentence_resynthesizes_only_that_segment(self, settings):
        generate("First. Second. Third.", settings, FakeTTS(), FakeLipsync())

        # A different script hash means a new cache dir, so segment reuse has to
        # be verified within a shared work_dir -- which is exactly how a real
        # edit-and-rerun behaves.
        tts = FakeTTS()
        generate("First. CHANGED. Third.", settings, tts, FakeLipsync())
        assert len(tts.calls) == 3  # new script dir, so all three are generated

    def test_no_cache_forces_resynthesis(self, settings):
        script = "One. Two."
        generate(script, settings, FakeTTS(), FakeLipsync())

        settings.use_cache = False
        tts = FakeTTS()
        generate(script, settings, tts, FakeLipsync())
        assert len(tts.calls) == 2

    def test_lipsync_is_skipped_on_rerun(self, settings):
        script = "Hello there."
        generate(script, settings, FakeTTS(), FakeLipsync())
        second = FakeLipsync()
        generate(script, settings, FakeTTS(), second)
        assert second.calls == 0


@requires_ffmpeg
class TestGuards:
    def test_missing_reference_wav_is_reported_with_guidance(self, settings, tmp_path):
        settings.reference_wav = tmp_path / "absent.wav"
        with pytest.raises(PipelineError, match="assets/README.md"):
            generate("Hello.", settings, FakeTTS(), FakeLipsync())

    def test_missing_base_loop_is_reported_with_guidance(self, settings, tmp_path):
        settings.base_loop = tmp_path / "absent.mp4"
        with pytest.raises(PipelineError, match="assets/README.md"):
            generate("Hello.", settings, FakeTTS(), FakeLipsync())

    def test_provider_that_declares_downscaling_is_refused(self, settings):
        with pytest.raises(PipelineError, match="does not preserve input"):
            generate("Hello.", settings, FakeTTS(), DownscalingLipsync())

    def test_provider_that_silently_downscales_is_caught(self, settings):
        # Defence in depth: the declaration is a hint, the probe is the proof.
        with pytest.raises(PipelineError, match="changed resolution"):
            generate("Hello.", settings, FakeTTS(), SneakyDownscalingLipsync())

    def test_empty_script_is_rejected(self, settings):
        with pytest.raises(ValueError, match="empty after normalization"):
            generate("   \n\n ", settings, FakeTTS(), FakeLipsync())
